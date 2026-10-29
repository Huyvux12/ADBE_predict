"""REL301m ADBE: python main.py {train,evaluate,replay,verify,report,all,notebook}.

`all` runs every step after training: evaluate -> replay -> verify -> report.
`notebook` executes the end-to-end notebook headlessly (trains into outputs/end_to_end).
"""
import argparse
from pathlib import Path

from adbe_rl.config import OUTPUT_DIR, ROOT, load_config


def main() -> None:
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['train', 'evaluate', 'replay', 'verify', 'report', 'all', 'notebook'])
    parser.add_argument('--track', choices=[*config.tracks, 'all'], default='all')
    parser.add_argument('--quick', action='store_true', help='small CEM budget for a smoke test')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    if args.quick:
        config = config.quick()
    tracks = list(config.tracks) if args.track == 'all' else [args.track]
    steps = ['evaluate', 'replay', 'verify', 'report'] if args.command == 'all' else [args.command]

    for step in steps:
        if step == 'train':
            from adbe_rl.training import train_track
            for track in tracks:
                train_track(config, track, args.output)
        elif step == 'evaluate':
            from adbe_rl.evaluation import evaluate_track
            for track in tracks:
                evaluate_track(config, track, args.output)
        elif step == 'replay':
            from adbe_rl.evaluation import replay_windows
            replay_windows(config, args.output)
        elif step == 'verify':
            from adbe_rl.verify import verify
            verify(config, args.output, tracks)
        elif step == 'report':
            from adbe_rl.report import build_report
            build_report(config, args.output)
        elif step == 'notebook':
            from adbe_rl.notebook import execute_notebook
            execute_notebook(ROOT / 'ADBE_REL301m_end_to_end.ipynb')


if __name__ == '__main__':
    main()
