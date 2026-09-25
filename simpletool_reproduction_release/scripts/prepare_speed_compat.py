#!/usr/bin/env python3
"""Create a separate metadata-only compatibility view for compressed-tensors 0.11."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    src=a.source.resolve(); dst=a.output.resolve()
    config=json.loads((src/'config.json').read_text())
    changes=[]
    for name,group in config['quantization_config']['config_groups'].items():
        weights=group['weights']
        for key in ['scale_dtype','zp_dtype']:
            if key in weights:
                if weights[key] is not None:
                    raise ValueError(f'Refusing to remove meaningful {key}')
                del weights[key]; changes.append(f'quantization_config.config_groups.{name}.weights.{key}')
    dst.mkdir(parents=True,exist_ok=False)
    for f in src.iterdir():
        if f.is_file() and f.name!='config.json': (dst/f.name).symlink_to(f)
    (dst/'config.json').write_text(json.dumps(config,indent=2)+'\n')
    report={'source':str(src),'removed_null_fields':changes,'weights_modified':False,
            'source_config_sha256':hashlib.sha256((src/'config.json').read_bytes()).hexdigest(),
            'compat_config_sha256':hashlib.sha256((dst/'config.json').read_bytes()).hexdigest()}
    (dst/'compatibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
