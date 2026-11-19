"""Copy-on-write GDN fork: read one prefix matrix, write B owned matrices.

Packed arithmetic adapted from installed MLX-LM (Apple Inc., MIT license in
third_party/MLX_LM_LICENSE). Existing packed arithmetic is credited upstream;
the changed contract is prefix batch1 -> independent output batch B.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.gated_delta import normalize_qk,compute_g


_FORK=mx.fast.metal_kernel(name='simpletool_gdn_read_shared_seed',
    input_names=['q','k','v','g','beta','prefix'],output_names=['y','state'],source=r'''
        const uint n=thread_position_in_grid.z,b=n/Hv,hv=n%Hv,hk=hv/(Hv/Hk);
        const uint lane=thread_index_in_simdgroup,local=lane&3;
        const uint dv=thread_position_in_grid.y*8+lane/4;
        const device float* ql=q+(b*Hk+hk)*128+local*32;
        const device float* kl=k+(b*Hk+hk)*128+local*32;
        const device float* source=prefix+(hv*Dv+dv)*128+local*32;
        device float* target=state+(n*Dv+dv)*128+local*32;
        float carry[32],part[8];
        for (uint i=0;i<32;++i) carry[i]=source[i];
        for (uint pb=0;pb<8;++pb) {
            float acc=0.0f;
            for (uint i=0;i<4;++i) {
                const uint e=pb*4+i;carry[e]=carry[e]*g[n];acc+=carry[e]*kl[e];
            }
            part[pb]=acc;
        }
        float kv=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
        kv+=simd_shuffle_xor(kv,1);kv+=simd_shuffle_xor(kv,2);
        const float delta=(v[n*Dv+dv]-kv)*beta[n];
        for (uint pb=0;pb<8;++pb) {
            float acc=0.0f;
            for (uint i=0;i<4;++i) {
                const uint e=pb*4+i;carry[e]=carry[e]+kl[e]*delta;acc+=carry[e]*ql[e];
            }
            part[pb]=acc;
        }
        float out=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
        out+=simd_shuffle_xor(out,1);out+=simd_shuffle_xor(out,2);
        if (local==0) y[n*Dv+dv]=out;
        for (uint i=0;i<32;++i) target[i]=carry[i];
    ''')


def fork_update(q,k,v,g,beta,prefix):
    B,S,Hk,Dk=q.shape;Hv,Dv=v.shape[-2:]
    if (S!=1 or not 1<=B<=4 or Dk!=128 or Dv%8 or Hv%Hk
        or prefix.shape!=(1,Hv,Dv,Dk) or k.shape!=q.shape
        or g.shape!=(B,1,Hv) or beta.shape!=g.shape):
        raise ValueError('Shared GDN seed requires S1, B1..4, Dk128, scalar gates and one prefix matrix')
    if any(t.dtype!=mx.float32 for t in (q,k,v,g,beta,prefix)):
        raise ValueError('Shared GDN seed uses FP32 activations/state only')
    return _FORK(inputs=[q,k,v,g,beta,prefix],template=[('Hk',Hk),('Hv',Hv),('Dv',Dv)],
        grid=(32,Dv//8,B*Hv),threadgroup=(32,2,1),
        output_shapes=[v.shape,(B,Hv,Dv,Dk)],output_dtypes=[mx.float32,mx.float32])


class SharedSeedGDN(GatedDeltaNet):
    def __call__(self,inputs,mask=None,cache=None):
        B,S,_=inputs.shape
        shared=(cache is not None and cache[1] is not None and cache[1].shape[0]==1 and B>1)
        supported=(S==1 and inputs.dtype==mx.float32 and self.head_k_dim==128 and self.head_v_dim%8==0
            and mask is None and cache is not None and cache[0] is not None and cache.lengths is None
            and not self.training and self.sharding_group is None)
        if not shared or not supported:
            if shared:cache[1]=mx.repeat(cache[1],B,axis=0)
            out=super().__call__(inputs,mask,cache)
            if cache is not None:
                cache.share_gdn_seed=(not self.training and self.sharding_group is None
                    and self.head_k_dim==128 and self.head_v_dim%8==0 and inputs.dtype==mx.float32)
            return out
        qkv=self.in_proj_qkv(inputs)
        z=self.in_proj_z(inputs).reshape(B,S,self.num_v_heads,self.head_v_dim)
        b=self.in_proj_b(inputs);a=self.in_proj_a(inputs)
        conv_input=mx.concatenate([cache[0],qkv],axis=1)
        cache[0]=mx.contiguous(conv_input[:,-(self.conv_kernel_size-1):,:])
        conv_out=nn.silu(self.conv1d(conv_input))
        q,k,v=[t.reshape(B,S,h,d) for t,h,d in zip(
            mx.split(conv_out,[self.key_dim,2*self.key_dim],axis=-1),
            [self.num_k_heads,self.num_k_heads,self.num_v_heads],
            [self.head_k_dim,self.head_k_dim,self.head_v_dim])]
        q,k=normalize_qk(q,k,inv_scale=self.head_k_dim**-.5,eps=1e-6)
        out,state=fork_update(q,k,v,compute_g(self.A_log,a,self.dt_bias),mx.sigmoid(b),cache[1])
        cache[1]=state;cache.advance(S);cache.share_gdn_seed=True
        return self.out_proj(self.norm(out,z).reshape(B,S,-1))


def install_shared_gdn_seed(model):
    for layer in model.layers:
        if layer.is_linear:
            if type(layer.linear_attn) is not GatedDeltaNet:
                raise ValueError('Shared GDN seed requires original unfused GDN layers')
            layer.linear_attn.__class__=SharedSeedGDN
