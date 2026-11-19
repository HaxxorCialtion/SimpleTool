// Independent int4/group64 fused gate/up + SwiGLU; FP32 arithmetic.
#include <metal_stdlib>
using namespace metal;
constant uint M [[function_constant(0)]];
constant uint ROWS [[function_constant(1)]];
constant bool COMPACT [[function_constant(2)]];
struct SwiGLUArguments { uint n,k; };
inline float load_meta(device const void* p,uint i) {
    return COMPACT ? float(((device const half*)p)[i]) : ((device const float*)p)[i];
}
kernel void simpletool_int4_swiglu(
    device const float* x [[buffer(0)]],device const uint* gw [[buffer(1)]],
    device const void* gs [[buffer(2)]],device const void* gb [[buffer(3)]],
    device const uint* uw [[buffer(4)]],device const void* us [[buffer(5)]],
    device const void* ub [[buffer(6)]],device float* y [[buffer(7)]],
    constant SwiGLUArguments& a [[buffer(8)]],uint tid [[thread_position_in_grid]],
    uint simdLane [[thread_index_in_simdgroup]]) {
    const uint N=a.n,K=a.k;

        const uint row0=(tid/32)*ROWS;
        const uint lane=simdLane;
        if (row0>=N) return;
        float gate[8][4],up[8][4];
        for (uint r=0;r<ROWS;++r) for (uint m=0;m<M;++m) { gate[r][m]=0.0f;up[r][m]=0.0f; }
        for (uint word=lane;word<K/8;word+=32) {
            uint gp[8],upack[8];float gscale[8],gbias[8],uscale[8],ubias[8];
            for (uint r=0;r<ROWS;++r) {
                const uint row=row0+r;
                gp[r]=row<N?gw[row*(K/8)+word]:0u;upack[r]=row<N?uw[row*(K/8)+word]:0u;
                gscale[r]=row<N?load_meta(gs,row*(K/64)+word/8):0.0f;
                gbias[r]=row<N?load_meta(gb,row*(K/64)+word/8):0.0f;
                uscale[r]=row<N?load_meta(us,row*(K/64)+word/8):0.0f;
                ubias[r]=row<N?load_meta(ub,row*(K/64)+word/8):0.0f;
            }
            for (uint j=0;j<8;++j) {
                float input[4];for (uint m=0;m<M;++m) input[m]=x[m*K+word*8+j];
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
    
}
