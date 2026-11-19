// Adapted from mlx_lm.models.gated_delta, Copyright 2025 Apple Inc.
// MIT license: third_party/MLX_LM_LICENSE. The existing packed reduction
// layout is ported here; it is not an original optimization of this project.
#include <metal_stdlib>
using namespace metal;
struct GDNArguments { uint batch, state_batch, key_heads, value_heads, value_width; };
constant bool GDN_OWNED [[function_constant(0)]];

// One normalized q/k/v step; scalar decay g and sigmoid beta are precomputed.
// All tensors and all arithmetic are float32. Dk is fixed at128.
kernel void simpletool_gdn_step(
    device const float* q [[buffer(0)]], device const float* k [[buffer(1)]],
    device const float* v [[buffer(2)]], device const float* g [[buffer(3)]],
    device const float* beta [[buffer(4)]], device float* state_in [[buffer(5)]],
    device float* y [[buffer(6)]], device float* state_out [[buffer(7)]],
    constant GDNArguments& a [[buffer(8)]],
    uint3 tid [[thread_position_in_grid]], uint lane [[thread_index_in_simdgroup]]) {
    const uint b=tid.z/a.value_heads, hv=tid.z%a.value_heads;
    const uint hk=hv/(a.value_heads/a.key_heads);
    const uint lane_in_row=lane&3, dv=tid.y*8+lane/4;
    if (dv>=a.value_width) return;
    const uint state_b=a.state_batch==1?0:b;
    const device float* ql=q+(b*a.key_heads+hk)*128+lane_in_row*32;
    const device float* kl=k+(b*a.key_heads+hk)*128+lane_in_row*32;
    const device float* source=state_in+((state_b*a.value_heads+hv)*a.value_width+dv)*128+lane_in_row*32;
    device float* target=(GDN_OWNED ? state_in : state_out)+((b*a.value_heads+hv)*a.value_width+dv)*128+lane_in_row*32;
    float state[32],part[8];
    for (uint i=0;i<32;++i) state[i]=source[i];
    const float decay=g[b*a.value_heads+hv];
    for (uint pb=0;pb<8;++pb) {
        float acc=0.0f;
        for (uint i=0;i<4;++i) {
            const uint e=pb*4+i;state[e]=state[e]*decay;acc+=state[e]*kl[e];
        }
        part[pb]=acc;
    }
    float kv=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
    kv+=simd_shuffle_xor(kv,1);kv+=simd_shuffle_xor(kv,2);
    const float delta=(v[(b*a.value_heads+hv)*a.value_width+dv]-kv)*beta[b*a.value_heads+hv];
    for (uint pb=0;pb<8;++pb) {
        float acc=0.0f;
        for (uint i=0;i<4;++i) {
            const uint e=pb*4+i;state[e]=state[e]+kl[e]*delta;acc+=state[e]*ql[e];
        }
        part[pb]=acc;
    }
    float out=((part[0]+part[1])+(part[2]+part[3]))+((part[4]+part[5])+(part[6]+part[7]));
    out+=simd_shuffle_xor(out,1);out+=simd_shuffle_xor(out,2);
    if (lane_in_row==0) y[(b*a.value_heads+hv)*a.value_width+dv]=out;
    // Each lane owns its32 elements and retains them until both readouts are
    // complete. An owned branch matrix can therefore be updated in place.
    for (uint i=0;i<32;++i) target[i]=state[i];
}
