"""Bounded-memory, mmap-backed packed data; compatible with common8 batches."""
import json
import resource
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
        self.metadata={}
        # Keep array mappings across shuffled accesses. The OS page cache is
        # shared between ranks; retaining an mmap does not copy a shard to RAM.
        # Reserve handles for distributed training and use a conservative
        # allowance of four handles per cached pair of arrays.
        fd_limit=resource.getrlimit(resource.RLIMIT_NOFILE)[0]
        shard_count=sum(s['split']==split for s in index['shards'])
        self.cache_limit=(shard_count if fd_limit==resource.RLIM_INFINITY else
                          min(shard_count,max(1,(fd_limit-256)//4)))
    def __len__(self):return len(self.entries)
    def __getitem__(self,i):
        if isinstance(i,slice):return [self[j] for j in range(*i.indices(len(self)))]
        name,k=self.entries[i]
        if name not in self.cache:
            p=self.directory/name
            if name not in self.metadata:
                self.metadata[name]=json.loads((p/'packs.json').read_text())
            self.cache[name]=(np.load(p/'ids.npy',mmap_mode='r'),np.load(p/'mask.npy',mmap_mode='r'))
            while len(self.cache)>self.cache_limit:self.cache.popitem(last=False)
        self.cache.move_to_end(name)
        ids,mask=self.cache[name]; pack=self.metadata[name][k]; a,b=pack['range']
        return {'input_ids':torch.from_numpy(ids[a:b].copy()),'loss_mask':torch.from_numpy(mask[a:b].copy()),
                'cu_seqlens':torch.tensor(pack['cuts'],dtype=torch.int32),'sources':pack['sources']}

def load(directory,split):
    return {'packs':Packs(directory,split),'pack_length':int(json.loads((Path(directory)/'INDEX.json').read_text()).get('pack_length',16384))}
