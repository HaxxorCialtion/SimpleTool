#!/usr/bin/env python3
"""Copy exact local benchmark fixtures; never include previous model predictions."""
import argparse
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simpletool_eval.data import describe_pair
from simpletool_eval.protocol import DATASETS_TO_EVAL


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True, help='Original benchmarks directory')
    parser.add_argument('--output', type=Path, default=Path('data'))
    args = parser.parse_args()
    source = args.source / 'Berkeley-Function-Calling-Leaderboard'
    (args.output / 'possible_answer').mkdir(parents=True, exist_ok=True)
    manifest = {'source': 'benchmarks/Berkeley-Function-Calling-Leaderboard', 'datasets': {}}
    for name in DATASETS_TO_EVAL:
        info = describe_pair(source, name)
        for rel in [Path(name), Path('possible_answer') / name]:
            dst = args.output / rel
            if dst.exists() and dst.read_bytes() != (source / rel).read_bytes():
                raise FileExistsError(f'Refusing to overwrite differing data: {dst}')
            shutil.copyfile(source / rel, dst)
        manifest['datasets'][Path(name).stem] = info
        print(f'{name}: input={info["input_count"]}, scored={info["scored_count"]}, filtered={len(info["filtered_ids"])}, missing_gt={len(info["missing_ground_truth_ids"])}')
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False)+'\n')

if __name__ == '__main__':
    main()
