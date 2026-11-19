"""Download one explicitly selected, revision-pinned model. No automatic bulk downloads."""
import argparse,json
from pathlib import Path

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--output',required=True);a=p.parse_args()
 registry=json.loads(Path(__file__).with_name('registry.json').read_text())
 matches=[m for m in registry['models'] if m['repo_id']==a.repo]
 if len(matches)!=1:p.error('Choose an exact repo_id from registry.json')
 m=matches[0];print('Weight size only: %.2f GiB; additional disk space needed.'%(m['weight_bytes']/2**30))
 from huggingface_hub import snapshot_download
 snapshot_download(repo_id=m['repo_id'],revision=m['revision'],local_dir=a.output)
if __name__=='__main__':main()
