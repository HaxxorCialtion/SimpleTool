"""Exact-order GDN replay candidate: immutable base plus short rank-one history.

All data/arithmetic FP32. This replays each rounded update, not a closed-form
low-rank sum. Packed readout reduction is Apple MLX-LM MIT arithmetic; see
third_party/MLX_LM_LICENSE. No borrowed buffer or input is mutated.
"""
import mlx.core as mx

REPLAY=r'''
    for(uint t=0;t<P;++t) {
        const float past_g=history_g[(b*P+t)*Hv+hv];
        const float past_delta=history_delta[((b*P+t)*Hv+hv)*Dv+dv];
        auto past_k=history_k+((b*P+t)*Hk+hk)*128+local*32;
        for(uint e=0;e<32;++e) {
            carry[e]=carry[e]*past_g;
            carry[e]=carry[e]+past_k[e]*past_delta;
        }
    }
'''
SETUP=r'''
    const uint n=thread_position_in_grid.z,b=n/Hv,hv=n%Hv,hk=hv/(Hv/Hk);
    const uint local=thread_index_in_simdgroup&3;
    const uint dv=thread_position_in_grid.y*8+thread_index_in_simdgroup/4;
    if(dv>=Dv) return;
    const device float* source=base+(((SB==1?0:b)*Hv+hv)*Dv+dv)*128+local*32;
    float carry[32];
    for(uint e=0;e<32;++e) carry[e]=source[e];
'''
STEP=SETUP+REPLAY+r'''
    const device float* ql=q+(b*Hk+hk)*128+local*32;
    const device float* kl=k+(b*Hk+hk)*128+local*32;
    float part[8];
    for(uint pb=0;pb<8;++pb) {
        float acc=0.0f;
        for(uint i=0;i<4;++i) {
            const uint e=pb*4+i;carry[e]=carry[e]*g[n];acc+=carry[e]*kl[e];
        }
        part[pb]=acc;
    }
    float kv=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
    kv+=simd_shuffle_xor(kv,1);kv+=simd_shuffle_xor(kv,2);
    const float delta=(v[n*Dv+dv]-kv)*beta[n];
    for(uint pb=0;pb<8;++pb) {
        float acc=0.0f;
        for(uint i=0;i<4;++i) {
            const uint e=pb*4+i;carry[e]=carry[e]+kl[e]*delta;acc+=carry[e]*ql[e];
        }
        part[pb]=acc;
    }
    float out=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
    out+=simd_shuffle_xor(out,1);out+=simd_shuffle_xor(out,2);
    if(local==0) {y[n*Dv+dv]=out;delta_out[n*Dv+dv]=delta;}
    if constexpr(WRITE) {
        device float* target=state_out+(n*Dv+dv)*128+local*32;
        for(uint e=0;e<32;++e) target[e]=carry[e];
    } else if(n==0 && dv==0 && local==0) state_out[0]=0.0f;
'''
MATERIALIZE=SETUP+REPLAY+r'''
    device float* target=state_out+(n*Dv+dv)*128+local*32;
    for(uint e=0;e<32;++e) target[e]=carry[e];
'''
_STEP=mx.fast.metal_kernel(name='simpletool_deferred_gdn_step',
    input_names=['q','k','v','g','beta','base','history_k','history_g','history_delta'],
    output_names=['y','delta_out','state_out'],source=STEP)
_MATERIALIZE=mx.fast.metal_kernel(name='simpletool_deferred_gdn_materialize',
    input_names=['base','history_k','history_g','history_delta'],output_names=['state_out'],source=MATERIALIZE)

class DeferredState:
    def __init__(self,base,batch,key_heads,limit=4):
        if (base.dtype!=mx.float32 or base.ndim!=4 or base.shape[0] not in (1,batch)
            or base.shape[-1]!=128 or base.shape[-2]%8 or not 1<=batch<=4
            or key_heads<=0 or base.shape[1]%key_heads or limit not in (1,2,4,8,16)):
            raise ValueError('Deferred GDN requires FP32, B1..4, Dk128, scalar gates and valid group heads')
        self.base=base;self.batch=batch;self.hk=key_heads;self.hv=base.shape[1];self.dv=base.shape[2];self.limit=limit
        self.clear()
    def clear(self):
        self.keys=mx.zeros((self.batch,0,self.hk,128),mx.float32)
        self.gates=mx.zeros((self.batch,0,self.hv),mx.float32)
        self.deltas=mx.zeros((self.batch,0,self.hv,self.dv),mx.float32)
    @property
    def pending(self):return self.keys.shape[1]
    @property
    def arrays(self):return [self.base,self.keys,self.gates,self.deltas]
    def step(self,q,k,v,g,beta):
        B=self.batch;P=self.pending;Hk=self.hk;Hv=self.hv;Dv=self.dv
        if (q.shape!=(B,1,Hk,128) or k.shape!=q.shape or v.shape!=(B,1,Hv,Dv)
            or g.shape!=(B,1,Hv) or beta.shape!=g.shape or any(a.dtype!=mx.float32 for a in (q,k,v,g,beta))):
            raise ValueError('Deferred GDN step input contract')
        write=P+1==self.limit
        y,delta,state=_STEP(inputs=[q,k,v,g,beta]+self.arrays,
            template=[('SB',self.base.shape[0]),('P',P),('Hk',Hk),('Hv',Hv),('Dv',Dv),('WRITE',write)],
            grid=(32,Dv//8,B*Hv),threadgroup=(32,2,1),
            output_shapes=[v.shape,(B,Hv,Dv),(B,Hv,Dv,128) if write else (1,)],output_dtypes=[mx.float32]*3)
        if write:self.base=state;self.clear()
        else:
            self.keys=mx.concatenate([self.keys,k],axis=1)
            self.gates=mx.concatenate([self.gates,g],axis=1)
            self.deltas=mx.concatenate([self.deltas,delta[:,None]],axis=1)
        return y
    def materialize(self):
        if self.pending==0:return mx.broadcast_to(self.base,(self.batch,self.hv,self.dv,128))
        return _MATERIALIZE(inputs=self.arrays,template=[('SB',self.base.shape[0]),('P',self.pending),('Hk',self.hk),('Hv',self.hv),('Dv',self.dv)],
            grid=(32,self.dv//8,self.batch*self.hv),threadgroup=(32,2,1),
            output_shapes=[(self.batch,self.hv,self.dv,128)],output_dtypes=[mx.float32])[0]
    def filter(self,indices):
        if self.base.shape[0]!=1:self.base=mx.take(self.base,indices,axis=0)
        self.keys=mx.take(self.keys,indices,axis=0);self.gates=mx.take(self.gates,indices,axis=0)
        self.deltas=mx.take(self.deltas,indices,axis=0);self.batch=len(indices)
