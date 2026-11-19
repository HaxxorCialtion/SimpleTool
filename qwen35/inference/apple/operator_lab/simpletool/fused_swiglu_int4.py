"""Decode-only two int4 projections + SiLU/product, without intermediate rows.

Keep packed tensors separate and immutable; no concatenation or requantization.
The two dot products follow row_reuse's FP32 per-weight accumulation order.
"""
import math
import mlx.core as mx
from mlx_lm.models.qwen3_next import Qwen3NextMLP
from .compact_metadata import CompactInt4Linear

_SWIGLU=mx.fast.metal_kernel(name='simpletool_int4_fused_swiglu',
    input_names=['x','gw','gs','gb','uw','us','ub'],output_names=['y'],source=r'''
        const uint row0=(thread_position_in_grid.x/32)*ROWS;
        const uint lane=thread_index_in_simdgroup;
        if (row0>=N) return;
        float gate[ROWS][M],up[ROWS][M];
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) { gate[r][m]=0.0f;up[r][m]=0.0f; }
        for (uint word=lane;word<K/8;word+=32) {
            uint gp[ROWS],upack[ROWS];float gscale[ROWS],gbias[ROWS],uscale[ROWS],ubias[ROWS];
            for (uint r=0;r<ROWS;++r) {
                const uint row=row0+r;
                gp[r]=row<N?gw[row*(K/8)+word]:0u;upack[r]=row<N?uw[row*(K/8)+word]:0u;
                gscale[r]=row<N?float(gs[row*(K/64)+word/8]):0.0f;
                gbias[r]=row<N?float(gb[row*(K/64)+word/8]):0.0f;
                uscale[r]=row<N?float(us[row*(K/64)+word/8]):0.0f;
                ubias[r]=row<N?float(ub[row*(K/64)+word/8]):0.0f;
            }
            for (uint j=0;j<8;++j) {
                float input[M];for (uint m=0;m<M;++m) input[m]=x[m*K+word*8+j];
                for (uint r=0;r<ROWS;++r) {
                    const float gv=float((gp[r]>>(4*j))&15)*gscale[r]+gbias[r];
                    const float uv=float((upack[r]>>(4*j))&15)*uscale[r]+ubias[r];
                    for (uint m=0;m<M;++m) { gate[r][m]+=gv*input[m];up[r][m]+=uv*input[m]; }
                }
            }
        }
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) {
            const float g=simd_sum(gate[r][m]),u=simd_sum(up[r][m]);
            if (lane==0 && row0+r<N) {
                const float t=1.0f/(1.0f+metal::precise::exp(metal::abs(g)));
                const float sig=g<0.0f?t:1.0f-t;
                y[m*N+row0+r]=(g*sig)*u;
            }
        }
    ''')

def fused_swiglu_int4(x,gate,up,rows=4,threads=128):
    k=x.shape[-1];m=x.size//k;n=gate.weight.shape[0]
    if (x.dtype!=mx.float32 or not 2<=m<=4 or k%64 or rows not in (1,2,4,8)
        or threads not in (64,128,256) or 'bias' in gate or 'bias' in up
        or gate.weight.shape!=up.weight.shape or gate.weight.shape[1]*8!=k
        or any((p.bits,p.group_size,p.mode)!=(4,64,'affine') for p in (gate,up))
        or gate.scales.dtype!=gate.biases.dtype or up.scales.dtype!=up.biases.dtype
        or gate.scales.dtype!=up.scales.dtype or gate.scales.dtype not in (mx.float16,mx.float32)):
        raise ValueError('Fused SwiGLU requires matching bias-free int4/group64 and FP32 B2..4')
    return _SWIGLU(inputs=[x,gate.weight,gate.scales,gate.biases,up.weight,up.scales,up.biases],
        template=[('M',m),('N',n),('K',k),('ROWS',rows)],
        grid=(math.ceil(n/(rows*(threads//32)))*threads,1,1),threadgroup=(threads,1,1),
        output_shapes=[(*x.shape[:-1],n)],output_dtypes=[mx.float32])[0]

class FusedSwiGLUMLP(Qwen3NextMLP):
    def __call__(self,x):
        if x.dtype==mx.float32 and 2<=x.size//x.shape[-1]<=4:
            batch=x.size//x.shape[-1]
            return self.down_proj(fused_swiglu_int4(x,self.gate_proj,self.up_proj,*self._swiglu_config[batch]))
        return super().__call__(x)

def install_fused_swiglu_int4(model,rows=None,threads=None):
    for layer in model.layers:
        m=layer.mlp
        if type(m) is not Qwen3NextMLP or not all(isinstance(p,CompactInt4Linear) for p in (m.gate_proj,m.up_proj)):
            raise ValueError('Fused SwiGLU requires original MLP with compact int4 projections')
    for layer in model.layers:
        layer.mlp.__class__=FusedSwiGLUMLP
        gate=layer.mlp.gate_proj
        measured4b=gate.weight.shape==(9216,320)
        layer.mlp._swiglu_config={b:((1,256) if measured4b and b in (2,4) else (4,128)) for b in (2,3,4)}
        if rows is not None or threads is not None:
            layer.mlp._swiglu_config={b:(rows or r,threads or t) for b,(r,t) in layer.mlp._swiglu_config.items()}
