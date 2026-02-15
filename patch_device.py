
import os

def patch_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    # Add device definition if not present
    header = 'import torch\ndevice = torch.device("mps" if torch.backends.mps.is_available() else "cpu")\n'
    
    if 'device =' not in content:
        # insert after imports (heuristic: after the last "import " or "from " line)
        lines = content.split('\n')
        last_import_idx = -1
        for i, line in enumerate(lines):
            if line.startswith('import ') or line.startswith('from '):
                last_import_idx = i
        
        if last_import_idx != -1:
            lines.insert(last_import_idx + 1, 'device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")')
            content = '\n'.join(lines)
        else:
            content = header + content

    # Replace .to("cuda") with .to(device)
    new_content = content.replace('.to("cuda")', '.to(device)')
    
    with open(filepath, 'w') as f:
        f.write(new_content)
    print(f"Patched {filepath}")

files_to_patch = [
    'infer/base.py',
    'models/hovernet/run_desc.py',
    'run_train.py', # might as well
]

for f in files_to_patch:
    if os.path.exists(f):
        patch_file(f)
