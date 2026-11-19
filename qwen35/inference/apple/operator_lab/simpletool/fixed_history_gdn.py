"""Fixed-capacity exact FP32 GDN history with runtime replay count.

The immutable base and rounded arithmetic match deferred_gdn. Old slots after
rebase are retained but never read until overwritten. No borrowed buffer write.
"""
import mlx.core as mx
from .deferred_gdn import DeferredState, SETUP, REPLAY, STEP, MATERIALIZE

FIXED_REPLAY=REPLAY.replace('t<P','t<uint(pending)').replace('b*P+t','b*C+t')
FIXED_STEP=SETUP+FIXED_REPLAY+STEP[len(SETUP+REPLAY):]
FIXED_MATERIALIZE=SETUP+FIXED_REPLAY+MATERIALIZE[len(SETUP+REPLAY):]
HISTORY_COPY=r'''
    if constexpr(!WRITE) {
        const uint phase=uint(pending);
        if(local==0) {
            for(uint t=0;t<C;++t) {
                const uint index=((b*C+t)*Hv+hv)*Dv+dv;
                history_delta_out[index]=t==phase ? delta : history_delta[index];
            }
            if(dv==0) {
                for(uint t=0;t<C;++t) {
                    const uint index=(b*C+t)*Hv+hv;
                    history_g_out[index]=t==phase ? g[n] : history_g[index];
                }
            }
        }
        if(dv==0 && hv%(Hv/Hk)==0) {
            for(uint t=0;t<C;++t) {
                const uint offset=((b*C+t)*Hk+hk)*128+local*32;
                for(uint e=0;e<32;++e)
                    history_k_out[offset+e]=t==phase ? kl[e] : history_k[offset+e];
            }
        }
    } else if(n==0 && dv==0 && local==0) {
        history_k_out[0]=0.0f;history_g_out[0]=0.0f;history_delta_out[0]=0.0f;
    }
'''
FUSED_STEP=FIXED_STEP+HISTORY_COPY
_STEP=mx.fast.metal_kernel(name='simpletool_fixed_history_gdn',
    input_names=['q','k','v','g','beta','base','history_k','history_g','history_delta','pending'],
    output_names=['y','delta_out','state_out'],source=FIXED_STEP)
_FUSED_STEP=mx.fast.metal_kernel(name='simpletool_fixed_history_gdn_fused',
    input_names=['q','k','v','g','beta','base','history_k','history_g','history_delta','pending'],
    output_names=['y','delta_out','state_out','history_k_out','history_g_out','history_delta_out'],source=FUSED_STEP)
_MATERIALIZE=mx.fast.metal_kernel(name='simpletool_fixed_history_materialize',
    input_names=['base','history_k','history_g','history_delta','pending'],
    output_names=['state_out'],source=FIXED_MATERIALIZE)

class FixedHistoryState(DeferredState):
    step_kernel=staticmethod(_STEP)
    fused_step_kernel=staticmethod(_FUSED_STEP)
    materialize_kernel=staticmethod(_MATERIALIZE)
    def dispatch(self):
        return (32,self.dv//8,self.batch*self.hv),(32,2,1)
    def __init__(self,base,batch,key_heads,limit=4,pending=0,write=None,fused=False):
        if limit not in (2,4,8,16):raise ValueError('Fixed history interval must be 2/4/8/16')
        if isinstance(pending,int) and not 0<=pending<limit:raise ValueError('Invalid pending history count')
        super().__init__(base,batch,key_heads,limit)
        self._pending=pending;self.force_write=write;self.fused=fused
    def clear(self):
        self.keys=mx.zeros((self.batch,self.limit-1,self.hk,128),mx.float32)
        self.gates=mx.zeros((self.batch,self.limit-1,self.hv),mx.float32)
        self.deltas=mx.zeros((self.batch,self.limit-1,self.hv,self.dv),mx.float32)
        self._pending=0
    @property
    def pending(self):return self._pending
    def step(self,q,k,v,g,beta):
        B=self.batch;Hk=self.hk;Hv=self.hv;Dv=self.dv
        if (q.shape!=(B,1,Hk,128) or k.shape!=q.shape or v.shape!=(B,1,Hv,Dv)
            or g.shape!=(B,1,Hv) or beta.shape!=g.shape or any(a.dtype!=mx.float32 for a in (q,k,v,g,beta))):
            raise ValueError('Fixed history GDN step input contract')
        pending=mx.array(self.pending,mx.int32) if isinstance(self.pending,int) else self.pending
        write=self.force_write if self.force_write is not None else self.pending+1==self.limit
        if not isinstance(write,bool):raise ValueError('Tracing requires explicit write specialization')
        shapes=[v.shape,(B,Hv,Dv),(B,Hv,Dv,128) if write else (1,)]
        if self.fused:shapes += [(1,)]*3 if write else [self.keys.shape,self.gates.shape,self.deltas.shape]
        grid,threadgroup=self.dispatch()
        result=(self.fused_step_kernel if self.fused else self.step_kernel)(inputs=[q,k,v,g,beta]+self.arrays+[pending],
            template=[('SB',self.base.shape[0]),('C',self.limit-1),('Hk',Hk),('Hv',Hv),('Dv',Dv),('WRITE',write)],
            grid=grid,threadgroup=threadgroup,
            output_shapes=shapes,output_dtypes=[mx.float32]*len(shapes))
        y,delta,state=result[:3]
        if write:self.base=state
        elif self.fused:self.keys,self.gates,self.deltas=result[3:]
        else:
            self.keys=mx.slice_update(self.keys,k,pending.reshape(1),axes=(1,))
            self.gates=mx.slice_update(self.gates,g,pending.reshape(1),axes=(1,))
            self.deltas=mx.slice_update(self.deltas,delta[:,None],pending.reshape(1),axes=(1,))
        self._pending=(self.pending+1)%self.limit
        return y
    def materialize(self):
        pending=mx.array(self.pending,mx.int32) if isinstance(self.pending,int) else self.pending
        grid,threadgroup=self.dispatch()
        return self.materialize_kernel(inputs=self.arrays+[pending],
            template=[('SB',self.base.shape[0]),('C',self.limit-1),('Hk',self.hk),('Hv',self.hv),('Dv',self.dv)],
            grid=grid,threadgroup=threadgroup,
            output_shapes=[(self.batch,self.hv,self.dv,128)],output_dtypes=[mx.float32])[0]
