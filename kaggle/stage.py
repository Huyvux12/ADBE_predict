"""Stage the Kaggle dataset (code + data + submitted models) and kernel (notebook).

python kaggle/stage.py  ->  kaggle/build/dataset, kaggle/build/kernel
Then upload with the Kaggle CLI (see kaggle/README.md).
"""
import json
import shutil
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = HERE / 'build'
NOTEBOOK = 'ADBE_REL301m_end_to_end.ipynb'
DATASET_FILES = ['config.json', 'requirements.txt', 'README.md', 'data/ADBE_daily.csv', 'data/provenance.json',
                 *[p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'adbe_rl').glob('*.py'))],
                 'adbe_rl/report_template.md',
                 *[p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'outputs/consecutive/models').glob('*.json'))]]


def stage_dataset() -> Path:
    target = BUILD / 'dataset'
    for name in DATASET_FILES:
        (target / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target / name)
    shutil.copy2(HERE / 'dataset-metadata.json', target)
    return target


def stage_kernel() -> Path:
    """Notebook with outputs cleared; Kaggle stores the outputs of its own run."""
    target = BUILD / 'kernel'
    target.mkdir(parents=True)
    notebook = json.loads((ROOT / NOTEBOOK).read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            cell['outputs'], cell['execution_count'] = [], None
    (target / NOTEBOOK).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
    shutil.copy2(HERE / 'kernel-metadata.json', target)
    return target


if __name__ == '__main__':
    shutil.rmtree(BUILD, ignore_errors=True)
    print('dataset:', stage_dataset())
    print('kernel: ', stage_kernel())
