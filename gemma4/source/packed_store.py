import json, bisect, hashlib
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch

class Packs:
    def __init__(self, directory):
        self.directory=Path(directory)
        index=json.loads((self.directory/'INDEX.json').read_text())
        self.shards=index['shards'];self.ends=[];self.cache=OrderedDict();self.order=None
        for s in self.shards:self.ends.append((self.ends[-1] if self.ends else 0)+s['packs'])
        self.length=self.ends[-1]
    def __len__(self):return self.length
    def __getitem__(self,i):
        if isinstance(i,slice):return [self[j] for j in range(*i.indices(len(self)))]
        if self.order is not None:i=int(self.order[i])
        j=bisect.bisect_right(self.ends,i);k=i-(self.ends[j-1] if j else 0)
        if j not in self.cache:
            p=self.directory/self.shards[j]['name']
            self.cache[j]=(np.load(p/'ids.npy',mmap_mode='r'),np.load(p/'mask.npy',mmap_mode='r'),json.loads((p/'packs.json').read_text()))
            if len(self.cache)>8:self.cache.popitem(last=False)
        self.cache.move_to_end(j)
        ids,mask,meta=self.cache[j];p=meta[k];a,b=p['range']
        return {'input_ids':torch.from_numpy(ids[a:b].copy()),'loss_mask':torch.from_numpy(mask[a:b].copy()),
                'cu_seqlens':torch.tensor(p['cuts'],dtype=torch.int32),'sources':p['sources']}
    def verify(self):
        for s in self.shards:
            for name,expected in s['sha256'].items():
                h=hashlib.sha256()
                with (self.directory/s['name']/name).open('rb') as f:
                    for block in iter(lambda:f.read(16<<20),b''):h.update(block)
                assert h.hexdigest()==expected,(s['name'],name)
