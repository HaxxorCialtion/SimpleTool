"""Bounded-memory, mmap-backed packed data; compatible with common8 batches."""
import json
from pathlib import Path
from collections import OrderedDict
import numpy as np
import torch

class Packs:
    def __init__(self, directory, split):
        self.directory=Path(directory)
        index=json.loads((self.directory/'INDEX.json').read_text())
        self.entries=[(s['name'],i) for s in index['shards'] if s['split']==split for i in range(s['packs'])]
        self.cache=OrderedDict()
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        if isinstance(i,slice):return [self[j] for j in range(*i.indices(len(self)))]
        name,k=self.entries[i]
        if name not in self.cache:
            p=self.directory/name
            self.cache[name]=(np.load(p/'ids.npy',mmap_mode='r'),np.load(p/'mask.npy',mmap_mode='r'),
                              json.loads((p/'packs.json').read_text()))
            while len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        ids,mask,meta=self.cache[name]; pack=meta[k]; a,b=pack['range']
        return {'input_ids':torch.from_numpy(ids[a:b].copy()),'loss_mask':torch.from_numpy(mask[a:b].copy()),
                'cu_seqlens':torch.tensor(pack['cuts'],dtype=torch.int32),'sources':pack['sources']}

def load(directory,split):
    return {'packs':Packs(directory,split),'pack_length':16384}
