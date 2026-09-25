#!/usr/bin/env python3
"""Download only the requested SimpleTool variant at a pinned HF revision."""
import argparse
import json
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = 'Cialtion/SimpleTool'
REVISION = 'aca7673dd2d90a919159613261ea374332e9d32f'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=Path('models'))
    p.add_argument('--variant', choices=['RT-Qwen3-4B', 'RT-Qwen3-4B-AWQ', 'RT-Qwen3-4B-AWQ-v2'], default='RT-Qwen3-4B')
    a = p.parse_args()
    snapshot_download(REPO, revision=REVISION, allow_patterns=[f'{a.variant}/*'],
                      ignore_patterns=['*/.ipynb_checkpoints/*'], local_dir=a.output, max_workers=4)
    (a.output/a.variant/'download_provenance.json').write_text(json.dumps({
        'repo_id': REPO, 'revision': REVISION, 'variant': a.variant}, indent=2)+'\n')
    print(a.output/a.variant)

if __name__ == '__main__':
    main()
