from pathlib import Path
import re

path = Path("train_classifier.py")
text = path.read_text(encoding="utf-8")

pattern = re.compile(
    r'requested_device = str\(args\.device\)\.lower\(\)\s*\n'
    r'\s*if requested_device\.startswith\("cuda"\)\s*-and\s*\(-not torch\.cuda\.is_available\(\)\):\s*\n'
    r'\s*print\("\[warn\] CUDA was requested but this PyTorch build has no CUDA\. Falling back to CPU\."\)\s*\n'
    r'\s*device = torch\.device\("cpu"\)\s*\n'
    r'\s*else:\s*\n'
    r'\s*device = torch\.device\(args\.device\)',
    re.MULTILINE,
)

replacement = '''requested_device = str(args.device).lower()
    if requested_device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[warn] CUDA was requested but this PyTorch build has no CUDA. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)'''

new_text, n = pattern.subn(replacement, text)
if n == 0:
    fallback_old = 'device = torch.device(args.device)\n    model = build_model(args.model).to(device)'
    fallback_new = '''requested_device = str(args.device).lower()
    if requested_device.startswith("cuda") and (not torch.cuda.is_available()):
        print("[warn] CUDA was requested but this PyTorch build has no CUDA. Falling back to CPU.")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    model = build_model(args.model).to(device)'''
    if fallback_old not in text:
        raise SystemExit("Could not find a patchable device block in train_classifier.py")
    new_text = text.replace(fallback_old, fallback_new)

path.write_text(new_text, encoding="utf-8")
print("Patched train_classifier.py")
