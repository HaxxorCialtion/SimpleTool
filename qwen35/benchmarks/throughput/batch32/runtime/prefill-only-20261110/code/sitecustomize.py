import os
if os.environ.get('SL_PREFILL_ONLY') == '1':
    import sys, json, importlib.abc, importlib.util
    from pathlib import Path
    manifest = json.loads((Path(__file__).resolve().parents[1] / 'overlay/manifest.json').read_text())
    class Finder(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in manifest:
                return importlib.util.spec_from_file_location(fullname, manifest[fullname]['path'])
    sys.meta_path.insert(0, Finder())
