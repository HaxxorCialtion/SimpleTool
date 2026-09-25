#!/usr/bin/env python3
"""Verify local HF files against pinned remote LFS SHA256 / Git blob IDs."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simpletool_eval.data import sha256
from huggingface_hub import HfApi
from download_model import REPO, REVISION


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('reports/model_verification.json'))
    a = p.parse_args()
    variant = a.model.name
    entries = HfApi().list_repo_tree(REPO, path_in_repo=variant, revision=REVISION, recursive=False)
    report = {'repo_id': REPO, 'revision': REVISION, 'variant': variant, 'files': {}}
    for entry in entries:
        if not hasattr(entry, 'blob_id'):
            continue
        path = a.model/Path(entry.path).name
        digest = sha256(path)
        if entry.lfs:
            expected = entry.lfs.sha256
            actual = digest
            kind = 'sha256'
        else:
            data = path.read_bytes()
            actual = hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            expected = entry.blob_id
            kind = 'git_blob_sha1'
        if actual != expected or path.stat().st_size != entry.size:
            raise ValueError(f'Content mismatch: {entry.path}')
        report['files'][path.name] = {'size': path.stat().st_size, 'sha256': digest,
                                    'remote_hash_kind': kind, 'remote_hash': expected, 'verified': True}
        print(f'Verified {path.name}', flush=True)
    if not any(n.endswith('.safetensors') for n in report['files']):
        raise ValueError('No weights verified')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2)+'\n')

if __name__ == '__main__':
    main()
