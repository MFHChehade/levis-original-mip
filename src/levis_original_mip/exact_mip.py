
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import gurobipy as gp
    from gurobipy import GRB
except Exception as exc:  # pragma: no cover
    gp = None
    GRB = None
    _GUROBI_IMPORT_ERROR = exc
else:
    _GUROBI_IMPORT_ERROR = None

try:
    import pulp
except Exception as exc:  # pragma: no cover
    pulp = None
    _PULP_IMPORT_ERROR = exc
else:
    _PULP_IMPORT_ERROR = None

from .data import get_datasets
from .models import build_model


def load_dataset_for_model(model_name: str, data_dir: Path, train: bool = False):
    train_ds, test_ds = get_datasets(model_name, data_dir)
    return train_ds if train else test_ds


def load_model(checkpoint_path: str, device: str = "cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device)
    model_name = ckpt["model_name"]
    model = build_model(model_name)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model_name, model


@torch.no_grad()
def choose_correct_sample(model: nn.Module, dataset, max_candidates: int = 1000):
    limit = min(max_candidates, len(dataset))
    for idx in range(limit):
        x, y = dataset[idx]
        logits = model(x.unsqueeze(0))
        pred = int(logits.argmax(dim=1).item())
        if pred == int(y):
            logits_1d = logits.squeeze(0)
            sorted_idx = torch.argsort(logits_1d, descending=True).tolist()
            target = next(cls for cls in sorted_idx if cls != y)
            return idx, x, int(y), int(target), logits_1d.detach().cpu()
    raise RuntimeError(f"Did not find a correctly classified sample in first {limit} points.")


def linear_interval(weight: torch.Tensor, bias: torch.Tensor, lo: torch.Tensor, hi: torch.Tensor):
    w_pos = torch.clamp(weight, min=0.0)
    w_neg = torch.clamp(weight, max=0.0)
    lower = w_pos @ lo + w_neg @ hi + bias
    upper = w_pos @ hi + w_neg @ lo + bias
    return lower, upper


def conv2d_interval(conv: nn.Conv2d, lo: torch.Tensor, hi: torch.Tensor):
    w = conv.weight.detach().cpu()
    b = conv.bias.detach().cpu()
    w_pos = torch.clamp(w, min=0.0)
    w_neg = torch.clamp(w, max=0.0)
    lower = F.conv2d(
        lo.unsqueeze(0), w_pos, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation
    ) + F.conv2d(
        hi.unsqueeze(0), w_neg, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation
    )
    upper = F.conv2d(
        hi.unsqueeze(0), w_pos, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation
    ) + F.conv2d(
        lo.unsqueeze(0), w_neg, bias=None, stride=conv.stride, padding=conv.padding, dilation=conv.dilation
    )
    lower = lower.squeeze(0)
    upper = upper.squeeze(0)
    lower = lower + b.view(-1, 1, 1)
    upper = upper + b.view(-1, 1, 1)
    return lower, upper


@torch.no_grad()
def global_bounds(model: nn.Module, input_shape):
    lo = torch.zeros(input_shape, dtype=torch.float32)
    hi = torch.ones(input_shape, dtype=torch.float32)

    bounds = []
    for module in model.layers:
        if isinstance(module, nn.Flatten):
            lo = lo.reshape(-1)
            hi = hi.reshape(-1)
            bounds.append({"type": "flatten", "post_lo": lo.clone(), "post_hi": hi.clone()})
        elif isinstance(module, nn.Linear):
            w = module.weight.detach().cpu()
            b = module.bias.detach().cpu()
            lo, hi = linear_interval(w, b, lo.reshape(-1), hi.reshape(-1))
            bounds.append({"type": "linear", "pre_lo": lo.clone(), "pre_hi": hi.clone(), "post_lo": lo.clone(), "post_hi": hi.clone()})
        elif isinstance(module, nn.Conv2d):
            lo, hi = conv2d_interval(module, lo, hi)
            bounds.append({"type": "conv2d", "pre_lo": lo.clone(), "pre_hi": hi.clone(), "post_lo": lo.clone(), "post_hi": hi.clone()})
        elif isinstance(module, nn.ReLU):
            pre_lo = lo.clone()
            pre_hi = hi.clone()
            lo = torch.clamp(lo, min=0.0)
            hi = torch.clamp(hi, min=0.0)
            bounds.append({"type": "relu", "pre_lo": pre_lo, "pre_hi": pre_hi, "post_lo": lo.clone(), "post_hi": hi.clone()})
        else:
            raise TypeError(f"Unsupported module type in bounds: {type(module)}")
    return bounds


def flatten_var_array(arr):
    return np.array(list(arr.reshape(-1)), dtype=object)


# ---------- Gurobi backend ----------

def add_var_array_gurobi(m, name, shape, lb, ub, vtype):
    arr = np.empty(shape, dtype=object)
    if np.isscalar(lb):
        lb_arr = np.full(shape, float(lb), dtype=float)
    else:
        lb_arr = np.array(lb, dtype=float).reshape(shape)
    if np.isscalar(ub):
        ub_arr = np.full(shape, float(ub), dtype=float)
    else:
        ub_arr = np.array(ub, dtype=float).reshape(shape)

    for idx in np.ndindex(shape):
        arr[idx] = m.addVar(lb=float(lb_arr[idx]), ub=float(ub_arr[idx]), vtype=vtype, name=f"{name}_{'_'.join(map(str, idx))}")
    return arr


def add_linear_layer_gurobi(m, name, prev_flat, layer: nn.Linear, lo, hi):
    out = add_var_array_gurobi(m, name, (layer.out_features,), lo, hi, GRB.CONTINUOUS)
    w = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy()
    for j in range(layer.out_features):
        expr = gp.quicksum(float(w[j, i]) * prev_flat[i] for i in range(layer.in_features)) + float(b[j])
        m.addConstr(out[j] == expr, name=f"{name}_eq_{j}")
    return out


def add_conv2d_layer_gurobi(m, name, prev_arr, layer: nn.Conv2d, lo, hi):
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    c_out, c_in, kh, kw = weight.shape
    _, h_in, w_in = prev_arr.shape
    sh, sw = layer.stride
    ph, pw = layer.padding
    dh, dw = layer.dilation
    h_out = ((h_in + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
    w_out = ((w_in + 2 * pw - dw * (kw - 1) - 1) // sw) + 1

    out = add_var_array_gurobi(m, name, (c_out, h_out, w_out), lo, hi, GRB.CONTINUOUS)
    for oc in range(c_out):
        for oh in range(h_out):
            for ow in range(w_out):
                expr = float(bias[oc])
                for ic in range(c_in):
                    for k1 in range(kh):
                        for k2 in range(kw):
                            ih = oh * sh - ph + k1 * dh
                            iw = ow * sw - pw + k2 * dw
                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                coeff = float(weight[oc, ic, k1, k2])
                                if coeff != 0.0:
                                    expr += coeff * prev_arr[ic, ih, iw]
                m.addConstr(out[oc, oh, ow] == expr, name=f"{name}_eq_{oc}_{oh}_{ow}")
    return out


def add_relu_layer_gurobi(m, name, pre_arr, lo, hi):
    post = add_var_array_gurobi(m, name, pre_arr.shape, np.maximum(lo, 0.0), np.maximum(hi, 0.0), GRB.CONTINUOUS)
    binaries = []
    unstable = 0

    for idx in np.ndindex(pre_arr.shape):
        l = float(lo[idx])
        u = float(hi[idx])
        z = pre_arr[idx]
        y = post[idx]

        if u <= 0.0:
            m.addConstr(y == 0.0, name=f"{name}_inactive_{'_'.join(map(str, idx))}")
        elif l >= 0.0:
            m.addConstr(y == z, name=f"{name}_active_{'_'.join(map(str, idx))}")
        else:
            unstable += 1
            a = m.addVar(vtype=GRB.BINARY, name=f"{name}_bin_{'_'.join(map(str, idx))}")
            binaries.append(a)
            m.addConstr(y >= z, name=f"{name}_c1_{'_'.join(map(str, idx))}")
            m.addConstr(y >= 0.0, name=f"{name}_c2_{'_'.join(map(str, idx))}")
            m.addConstr(y <= z - l * (1.0 - a), name=f"{name}_c3_{'_'.join(map(str, idx))}")
            m.addConstr(y <= u * a, name=f"{name}_c4_{'_'.join(map(str, idx))}")
    return post, binaries, unstable


def build_exact_targeted_linf_mip_gurobi(model: nn.Module, center: torch.Tensor, true_class: int, target_class: int):
    if gp is None:
        raise RuntimeError(f"gurobipy import failed: {_GUROBI_IMPORT_ERROR}")

    center = center.detach().cpu().float()
    input_shape = tuple(center.shape)
    bounds = global_bounds(model, input_shape)

    m = gp.Model("exact_targeted_linf")
    m.Params.OutputFlag = 0

    x = add_var_array_gurobi(m, "x", input_shape, 0.0, 1.0, GRB.CONTINUOUS)
    t = m.addVar(lb=0.0, ub=1.0, vtype=GRB.CONTINUOUS, name="t")

    for idx in np.ndindex(input_shape):
        cval = float(center[idx].item())
        m.addConstr(x[idx] - cval <= t, name=f"linf_pos_{'_'.join(map(str, idx))}")
        m.addConstr(cval - x[idx] <= t, name=f"linf_neg_{'_'.join(map(str, idx))}")

    prev = x
    binary_count = 0
    unstable_count = 0
    relu_layer_summaries = []
    bound_ptr = 0

    for layer_idx, module in enumerate(model.layers):
        info = bounds[bound_ptr]
        bound_ptr += 1

        if isinstance(module, nn.Flatten):
            prev = flatten_var_array(prev)
        elif isinstance(module, nn.Linear):
            lo = info["pre_lo"].numpy()
            hi = info["pre_hi"].numpy()
            prev = add_linear_layer_gurobi(m, f"layer{layer_idx}_linear", flatten_var_array(prev), module, lo, hi)
        elif isinstance(module, nn.Conv2d):
            lo = info["pre_lo"].numpy()
            hi = info["pre_hi"].numpy()
            prev = add_conv2d_layer_gurobi(m, f"layer{layer_idx}_conv", prev, module, lo, hi)
        elif isinstance(module, nn.ReLU):
            lo = info["pre_lo"].numpy()
            hi = info["pre_hi"].numpy()
            prev, bins, unstable = add_relu_layer_gurobi(m, f"layer{layer_idx}_relu", prev, lo, hi)
            binary_count += len(bins)
            unstable_count += unstable
            relu_layer_summaries.append({"layer_index": layer_idx, "unstable_relus": unstable, "shape": list(prev.shape)})
        else:
            raise TypeError(f"Unsupported module type in MIP builder: {type(module)}")

    logits = flatten_var_array(prev)
    m.addConstr(logits[true_class] - logits[target_class] <= 0.0, name="target_margin_violation")
    m.setObjective(t, GRB.MINIMIZE)

    build_meta = {
        "backend": "gurobi",
        "binary_count": binary_count,
        "unstable_count": unstable_count,
        "relu_layers": relu_layer_summaries,
        "input_shape": list(input_shape),
        "true_class": int(true_class),
        "target_class": int(target_class),
    }
    return m, build_meta


# ---------- PuLP/CBC backend ----------

def add_var_array_pulp(name, shape, lb, ub, cat):
    arr = np.empty(shape, dtype=object)
    if np.isscalar(lb):
        lb_arr = np.full(shape, float(lb), dtype=float)
    else:
        lb_arr = np.array(lb, dtype=float).reshape(shape)
    if np.isscalar(ub):
        ub_arr = np.full(shape, float(ub), dtype=float)
    else:
        ub_arr = np.array(ub, dtype=float).reshape(shape)

    for idx in np.ndindex(shape):
        idx_name = "_".join(map(str, idx))
        arr[idx] = pulp.LpVariable(
            f"{name}_{idx_name}",
            lowBound=float(lb_arr[idx]),
            upBound=float(ub_arr[idx]),
            cat=cat,
        )
    return arr


def add_linear_layer_pulp(prob, name, prev_flat, layer: nn.Linear, lo, hi):
    out = add_var_array_pulp(name, (layer.out_features,), lo, hi, pulp.LpContinuous)
    w = layer.weight.detach().cpu().numpy()
    b = layer.bias.detach().cpu().numpy()
    for j in range(layer.out_features):
        expr = pulp.lpSum(float(w[j, i]) * prev_flat[i] for i in range(layer.in_features)) + float(b[j])
        prob += (out[j] == expr), f"{name}_eq_{j}"
    return out


def add_conv2d_layer_pulp(prob, name, prev_arr, layer: nn.Conv2d, lo, hi):
    weight = layer.weight.detach().cpu().numpy()
    bias = layer.bias.detach().cpu().numpy()
    c_out, c_in, kh, kw = weight.shape
    _, h_in, w_in = prev_arr.shape
    sh, sw = layer.stride
    ph, pw = layer.padding
    dh, dw = layer.dilation
    h_out = ((h_in + 2 * ph - dh * (kh - 1) - 1) // sh) + 1
    w_out = ((w_in + 2 * pw - dw * (kw - 1) - 1) // sw) + 1

    out = add_var_array_pulp(name, (c_out, h_out, w_out), lo, hi, pulp.LpContinuous)
    for oc in range(c_out):
        for oh in range(h_out):
            for ow in range(w_out):
                terms = [float(bias[oc])]
                for ic in range(c_in):
                    for k1 in range(kh):
                        for k2 in range(kw):
                            ih = oh * sh - ph + k1 * dh
                            iw = ow * sw - pw + k2 * dw
                            if 0 <= ih < h_in and 0 <= iw < w_in:
                                coeff = float(weight[oc, ic, k1, k2])
                                if coeff != 0.0:
                                    terms.append(coeff * prev_arr[ic, ih, iw])
                prob += (out[oc, oh, ow] == pulp.lpSum(terms)), f"{name}_eq_{oc}_{oh}_{ow}"
    return out


def add_relu_layer_pulp(prob, name, pre_arr, lo, hi):
    post = add_var_array_pulp(name, pre_arr.shape, np.maximum(lo, 0.0), np.maximum(hi, 0.0), pulp.LpContinuous)
    unstable = 0

    for idx in np.ndindex(pre_arr.shape):
        l = float(lo[idx])
        u = float(hi[idx])
        z = pre_arr[idx]
        y = post[idx]
        idx_name = "_".join(map(str, idx))

        if u <= 0.0:
            prob += (y == 0.0), f"{name}_inactive_{idx_name}"
        elif l >= 0.0:
            prob += (y == z), f"{name}_active_{idx_name}"
        else:
            unstable += 1
            a = pulp.LpVariable(f"{name}_bin_{idx_name}", cat=pulp.LpBinary)
            prob += (y >= z), f"{name}_c1_{idx_name}"
            prob += (y >= 0.0), f"{name}_c2_{idx_name}"
            prob += (y <= z - l * (1.0 - a)), f"{name}_c3_{idx_name}"
            prob += (y <= u * a), f"{name}_c4_{idx_name}"
    return post, unstable


def build_exact_targeted_linf_mip_pulp(model: nn.Module, center: torch.Tensor, true_class: int, target_class: int):
    if pulp is None:
        raise RuntimeError(f"pulp import failed: {_PULP_IMPORT_ERROR}")

    center = center.detach().cpu().float()
    input_shape = tuple(center.shape)
    bounds = global_bounds(model, input_shape)

    prob = pulp.LpProblem("exact_targeted_linf", pulp.LpMinimize)
    x = add_var_array_pulp("x", input_shape, 0.0, 1.0, pulp.LpContinuous)
    t = pulp.LpVariable("t", lowBound=0.0, upBound=1.0, cat=pulp.LpContinuous)

    for idx in np.ndindex(input_shape):
        cval = float(center[idx].item())
        idx_name = "_".join(map(str, idx))
        prob += (x[idx] - cval <= t), f"linf_pos_{idx_name}"
        prob += (cval - x[idx] <= t), f"linf_neg_{idx_name}"

    prev = x
    unstable_count = 0
    relu_layer_summaries = []
    bound_ptr = 0

    for layer_idx, module in enumerate(model.layers):
        info = bounds[bound_ptr]
        bound_ptr += 1

        if isinstance(module, nn.Flatten):
            prev = flatten_var_array(prev)
        elif isinstance(module, nn.Linear):
            lo = info["pre_lo"].numpy()
            hi = info["pre_hi"].numpy()
            prev = add_linear_layer_pulp(prob, f"layer{layer_idx}_linear", flatten_var_array(prev), module, lo, hi)
        elif isinstance(module, nn.Conv2d):
            lo = info["pre_lo"].numpy()
            hi = info["pre_hi"].numpy()
            prev = add_conv2d_layer_pulp(prob, f"layer{layer_idx}_conv", prev, module, lo, hi)
        elif isinstance(module, nn.ReLU):
            lo = info["pre_lo"].numpy()
            hi = info["pre_hi"].numpy()
            prev, unstable = add_relu_layer_pulp(prob, f"layer{layer_idx}_relu", prev, lo, hi)
            unstable_count += unstable
            relu_layer_summaries.append({"layer_index": layer_idx, "unstable_relus": unstable, "shape": list(prev.shape)})
        else:
            raise TypeError(f"Unsupported module type in MIP builder: {type(module)}")

    logits = flatten_var_array(prev)
    prob += (logits[true_class] - logits[target_class] <= 0.0), "target_margin_violation"
    prob += t

    build_meta = {
        "backend": "cbc",
        "binary_count": unstable_count,
        "unstable_count": unstable_count,
        "relu_layers": relu_layer_summaries,
        "input_shape": list(input_shape),
        "true_class": int(true_class),
        "target_class": int(target_class),
    }
    return prob, build_meta


def solve_single_instance(checkpoint_path: str, data_dir: str, time_limit: float, mip_gap: float, max_candidates: int, solver: str):
    model_name, model = load_model(checkpoint_path, device="cpu")
    dataset = load_dataset_for_model(model_name, Path(data_dir), train=False)

    idx, x, y_true, y_target, logits = choose_correct_sample(model, dataset, max_candidates=max_candidates)
    solver = solver.lower()

    if solver == "gurobi":
        mip, meta = build_exact_targeted_linf_mip_gurobi(model, x, y_true, y_target)
        mip.Params.TimeLimit = float(time_limit)
        mip.Params.MIPGap = float(mip_gap)

        start = time.time()
        mip.optimize()
        runtime = time.time() - start

        status_map = {
            GRB.OPTIMAL: "OPTIMAL",
            GRB.TIME_LIMIT: "TIME_LIMIT",
            GRB.INTERRUPTED: "INTERRUPTED",
            GRB.INFEASIBLE: "INFEASIBLE",
            GRB.SUBOPTIMAL: "SUBOPTIMAL",
        }

        result = {
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "sample_index": idx,
            "true_class": y_true,
            "target_class": y_target,
            "nominal_logits": logits.tolist(),
            "runtime_sec": runtime,
            "solver_backend": "gurobi",
            "solver_status_code": int(mip.Status),
            "solver_status": status_map.get(mip.Status, str(mip.Status)),
            "objective_linf": float(mip.ObjVal) if mip.SolCount > 0 else None,
            "mip_gap": float(mip.MIPGap) if mip.SolCount > 0 else None,
            "sol_count": int(mip.SolCount),
            "build_meta": meta,
        }
        return result

    if solver == "cbc":
        if pulp is None:
            raise RuntimeError(f"pulp import failed: {_PULP_IMPORT_ERROR}")
        prob, meta = build_exact_targeted_linf_mip_pulp(model, x, y_true, y_target)

        cbc = pulp.PULP_CBC_CMD(msg=False, timeLimit=float(time_limit), gapRel=float(mip_gap))
        start = time.time()
        status_code = prob.solve(cbc)
        runtime = time.time() - start
        status_text = pulp.LpStatus.get(status_code, str(status_code))
        objective = float(pulp.value(prob.objective)) if pulp.value(prob.objective) is not None else None

        result = {
            "checkpoint_path": checkpoint_path,
            "model_name": model_name,
            "sample_index": idx,
            "true_class": y_true,
            "target_class": y_target,
            "nominal_logits": logits.tolist(),
            "runtime_sec": runtime,
            "solver_backend": "cbc",
            "solver_status_code": int(status_code),
            "solver_status": status_text,
            "objective_linf": objective,
            "mip_gap": None,
            "sol_count": 1 if status_text in {"Optimal", "Integer Feasible"} else 0,
            "build_meta": meta,
        }
        return result

    raise ValueError(f"Unsupported solver: {solver}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data_dir", default="data")
    parser.add_argument("--time_limit", type=float, default=15.0)
    parser.add_argument("--mip_gap", type=float, default=0.05)
    parser.add_argument("--max_candidates", type=int, default=1000)
    parser.add_argument("--solver", choices=["cbc", "gurobi"], default="gurobi")
    parser.add_argument("--out_json", default="")
    args = parser.parse_args()

    result = solve_single_instance(
        checkpoint_path=args.checkpoint,
        data_dir=args.data_dir,
        time_limit=args.time_limit,
        mip_gap=args.mip_gap,
        max_candidates=args.max_candidates,
        solver=args.solver,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.out_json:
        Path(args.out_json).write_text(text)


if __name__ == "__main__":
    main()
