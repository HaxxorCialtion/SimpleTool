// Independent affine-int4 group64 implementation. No dense weight expansion.
#include <metal_stdlib>
using namespace metal;
constant uint batch [[function_constant(0)]];
constant uint outputs [[function_constant(1)]];
constant uint reuse_rows [[function_constant(2)]];
constant bool compact_metadata [[function_constant(3)]];
struct Int4Arguments { uint n, k, c0, c1, c2, c3; };

inline float load_metadata(device const void* data,uint index) {
    // A storage-format specialization only. Every arithmetic value is float.
    return compact_metadata ? float(((device const half*)data)[index])
                            : ((device const float*)data)[index];
}

kernel void simpletool_int4(
    device const float* x [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const void* scales [[buffer(2)]],
    device const void* biases [[buffer(3)]],
    device float* y0 [[buffer(4)]], device float* y1 [[buffer(5)]],
    device float* y2 [[buffer(6)]], device float* y3 [[buffer(7)]],
    constant Int4Arguments& a [[buffer(8)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row=tid/32;
    if (row>=a.n) return; // whole SIMD group takes the same branch
    float acc[4]={0.0f,0.0f,0.0f,0.0f};
    for (uint word=lane;word<a.k/8;word+=32) {
        const uint packed=w[row*(a.k/8)+word];
        const uint group=word/8;
        const float s=load_metadata(scales,row*(a.k/64)+group);
        const float b=load_metadata(biases,row*(a.k/64)+group);
        for (uint j=0;j<8;++j) {
            const float value=float((packed>>(4*j))&15)*s+b;
            for (uint m=0;m<batch;++m) acc[m]+=value*x[m*a.k+word*8+j];
        }
    }
    for (uint m=0;m<batch;++m) {
        const float sum=simd_sum(acc[m]);
        if (lane==0) {
            if (outputs==1 || row<a.c0) y0[m*a.c0+row]=sum;
            else if (row<a.c0+a.c1) y1[m*a.c1+row-a.c0]=sum;
            else if (row<a.c0+a.c1+a.c2) y2[m*a.c2+row-a.c0-a.c1]=sum;
            else y3[m*a.c3+row-a.c0-a.c1-a.c2]=sum;
        }
    }
}

// Each SIMD group owns several output rows. Input loads are shared through
// registers while every row retains the original word/j/accumulate order.
kernel void simpletool_int4_row_reuse(
    device const float* x [[buffer(0)]],
    device const uint* w [[buffer(1)]],
    device const void* scales [[buffer(2)]],
    device const void* biases [[buffer(3)]],
    device float* y0 [[buffer(4)]], device float* y1 [[buffer(5)]],
    device float* y2 [[buffer(6)]], device float* y3 [[buffer(7)]],
    constant Int4Arguments& a [[buffer(8)]],
    uint tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint row0=(tid/32)*reuse_rows;
    if (row0>=a.n) return;
    float acc[8][4];
    for (uint r=0;r<reuse_rows;++r)
        for (uint m=0;m<batch;++m) acc[r][m]=0.0f;
    for (uint word=lane;word<a.k/8;word+=32) {
        uint packed[8];float s[8],b[8];
        for (uint r=0;r<reuse_rows;++r) {
            const uint row=row0+r;
            packed[r]=row<a.n?w[row*(a.k/8)+word]:0u;
            s[r]=row<a.n?load_metadata(scales,row*(a.k/64)+word/8):0.0f;
            b[r]=row<a.n?load_metadata(biases,row*(a.k/64)+word/8):0.0f;
        }
        for (uint j=0;j<8;++j) {
            float input[4];
            for (uint m=0;m<batch;++m) input[m]=x[m*a.k+word*8+j];
            for (uint r=0;r<reuse_rows;++r) {
                const float value=float((packed[r]>>(4*j))&15)*s[r]+b[r];
                for (uint m=0;m<batch;++m) acc[r][m]+=value*input[m];
            }
        }
    }
    for (uint r=0;r<reuse_rows;++r)
        for (uint m=0;m<batch;++m) {
            const float sum=simd_sum(acc[r][m]);
            const uint row=row0+r;
            if (lane==0 && row<a.n) {
                if (outputs==1 || row<a.c0) y0[m*a.c0+row]=sum;
                else if (row<a.c0+a.c1) y1[m*a.c1+row-a.c0]=sum;
                else if (row<a.c0+a.c1+a.c2) y2[m*a.c2+row-a.c0-a.c1]=sum;
                else y3[m*a.c3+row-a.c0-a.c1-a.c2]=sum;
            }
        }
}
