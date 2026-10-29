"""Headless notebook runner: executes code cells and stores text, tables and figures.

Works without Jupyter installed. Like a kernel, the value of a cell's last
expression is displayed (HTML for DataFrames) and open matplotlib figures are
embedded as PNG.
"""
from __future__ import annotations

import ast
import base64
import contextlib
import io
import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt

from .config import ROOT


def run_cell(source: str, namespace: dict, name: str):
    """Execute a cell; return (printed text, value of the trailing expression or None)."""
    tree = ast.parse(source)
    last = tree.body.pop() if tree.body and isinstance(tree.body[-1], ast.Expr) else None
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        exec(compile(tree, name, 'exec'), namespace)
        value = eval(compile(ast.Expression(last.value), name, 'eval'), namespace) if last else None
    return buffer.getvalue(), value


def figure_outputs() -> list[dict]:
    outputs = []
    for number in plt.get_fignums():
        buffer = io.BytesIO()
        plt.figure(number).savefig(buffer, format='png', dpi=110, bbox_inches='tight')
        outputs.append({'output_type': 'display_data', 'metadata': {},
                        'data': {'image/png': base64.b64encode(buffer.getvalue()).decode(),
                                 'text/plain': ['<Figure>']}})
    plt.close('all')
    return outputs


def execute_notebook(path: Path) -> None:
    """Run every code cell in order from the project root and save outputs in place."""
    notebook = json.loads(Path(path).read_text(encoding='utf-8'))
    namespace, count, cwd = {'__name__': '__main__'}, 0, os.getcwd()
    plt.switch_backend('Agg')
    warnings.filterwarnings('ignore', message='FigureCanvasAgg is non-interactive')  # plt.show() in cells
    os.chdir(ROOT)
    try:
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] != 'code':
                continue
            count += 1
            text, value = run_cell(''.join(cell['source']), namespace, f'{Path(path).name}:cell{i}')
            outputs = [{'output_type': 'stream', 'name': 'stdout', 'text': text.splitlines(True)}] if text else []
            outputs += figure_outputs()
            if value is not None:
                data = {'text/plain': repr(value).splitlines(True)}
                if hasattr(value, '_repr_html_'):
                    data['text/html'] = value._repr_html_().splitlines(True)
                outputs.append({'output_type': 'execute_result', 'execution_count': count,
                                'metadata': {}, 'data': data})
            cell['execution_count'], cell['outputs'] = count, outputs
            print(f'  cell {count} done', flush=True)
    finally:
        os.chdir(cwd)
    Path(path).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
