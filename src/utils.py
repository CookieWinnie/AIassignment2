import csv
import json
import os
import platform
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def configure_paths():
    for name in ['src', 'data', 'models', 'results', 'report_material', '.cache/tmp', '.cache/matplotlib', '.cache/pip']:
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    for key, value in {'TEMP': '.cache/tmp', 'TMP': '.cache/tmp', 'MPLCONFIGDIR': '.cache/matplotlib', 'PIP_CACHE_DIR': '.cache/pip', 'TORCH_HOME': '.cache/torch'}.items():
        os.environ[key] = str(ROOT / value)


def seed_all(seed):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError('Cannot write an empty result table')
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2), encoding='utf-8')


def environment(device):
    import torch
    import torchvision
    import numpy as np
    import matplotlib
    import sklearn
    import sys
    return dict(python=platform.python_version(),
                pytorch=torch.__version__, torchvision=torchvision.__version__,
                numpy=np.__version__, matplotlib=matplotlib.__version__, sklearn=sklearn.__version__,
                operating_system=platform.platform(), device=str(device),
                cuda_available=torch.cuda.is_available(), compiled_cuda=torch.version.cuda,
                gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')
