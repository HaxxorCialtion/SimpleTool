"""Experimental vLLM GDN endpoint export. GPU validation required before use.
One complete prefill output, one FP32 snapshot per row at floor(P/64)*64,
and <=63 recurrent updates over already-projected normalized K/V/g/beta.
"""
import torch
from vllm.triton_utils import triton,tl

@triton.jit
def replay_endpoint(KP,VP,GP,BP,SP,EP,CU,
                    H:tl.constexpr,HG:tl.constexpr,K:tl.constexpr,V:tl.constexpr,
                    BK:tl.constexpr,BV:tl.constexpr,BT:tl.constexpr):
    iv=tl.program_id(0); ih=tl.program_id(1)%H; row=tl.program_id(1)//H
    p=tl.load(EP+row)
    if p<=0:return
    start=tl.load(CU+row); begin=start+(p//BT)*BT; end=start+p
    ok=tl.arange(0,BK); ov=iv*BV+tl.arange(0,BV)
    mask=(ov[:,None]<V)&(ok[None,:]<K)
    ptr=SP+((row*H+ih)*V+ov[:,None])*K+ok[None,:]
    state=tl.load(ptr,mask=mask,other=0).to(tl.float32)
    for t in range(begin,end):
        key=tl.load(KP+(t*HG+ih//(H//HG))*K+ok,mask=ok<K,other=0).to(tl.float32)
        value=tl.load(VP+(t*H+ih)*V+ov,mask=ov<V,other=0).to(tl.float32)
        gate=tl.load(GP+t*H+ih).to(tl.float32)
        beta=tl.load(BP+t*H+ih).to(tl.float32)
        state*=tl.exp(gate)
        value=(value-tl.sum(state*key[None,:],axis=1))*beta
        state+=value[:,None]*key[None,:]
    tl.store(ptr,state,mask=mask)


def chunk_with_endpoint(q,k,v,g,beta,initial_state,cu_seqlens,
                        chunk_indices,chunk_offsets,endpoints,backend='triton',host_lengths=None,host_endpoints=None,plan_cache=None):
    from vllm.third_party.flash_linear_attention.ops.utils import FLA_CHUNK_SIZE
    BT=FLA_CHUNK_SIZE
    assert BT==64 and q.shape[0]==1 and k.is_contiguous() and v.is_contiguous()
    assert beta.ndim==3 and g.ndim==3
    n=endpoints.numel();H=v.shape[-2];HG=k.shape[-2];V=v.shape[-1];K=k.shape[-1]
    snap=torch.empty((n,H,V,K),device=k.device,dtype=torch.float32)
    if backend=='triton':
        from vllm.third_party.flash_linear_attention.ops.cumsum import chunk_local_cumsum
        from vllm.third_party.flash_linear_attention.ops.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
        from vllm.third_party.flash_linear_attention.ops.solve_tril import solve_tril
        from vllm.third_party.flash_linear_attention.ops.wy_fast import recompute_w_u_fwd
        from vllm.third_party.flash_linear_attention.ops.chunk_delta_h import chunk_gated_delta_rule_fwd_h
        from vllm.third_party.flash_linear_attention.ops.chunk_o import chunk_fwd_o
        gc=chunk_local_cumsum(g,chunk_size=BT,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices)
        A=chunk_scaled_dot_kkt_fwd(k=k,beta=beta,g=gc,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,output_dtype=torch.float32)
        A=solve_tril(A=A,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,output_dtype=k.dtype)
        w,u=recompute_w_u_fwd(k=k,v=v,beta=beta,A=A,g_cumsum=gc,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices)
        h,vnew,final=chunk_gated_delta_rule_fwd_h(k=k,w=w,u=u,g=gc,initial_state=initial_state,
            output_final_state=True,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices,chunk_offsets=chunk_offsets,
            endpoint_snapshot=snap,endpoint_offsets=endpoints)
        out=chunk_fwd_o(q=q,k=k,v=vnew,h=h,g=gc,scale=K**-.5,cu_seqlens=cu_seqlens,chunk_indices=chunk_indices)
    elif backend=='flashinfer':
        from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import fi_chunk_gated_delta_rule
        assert host_lengths is not None and host_endpoints is not None
        plan=getattr(plan_cache,'_exact_fi_plan',None) if plan_cache is not None else None
        if plan is None:
            import math
            every=math.gcd(*(p//BT*BT for p in host_endpoints)) or BT
            starts=[0]
            for length in host_lengths:starts.append(starts[-1]+length//every)
            indices=[starts[i]+(p//BT*BT)//every-1 if p>=BT else -1 for i,p in enumerate(host_endpoints)]
            cs=torch.tensor(starts,device=k.device,dtype=torch.int64)
            rows=torch.tensor(indices,device=k.device,dtype=torch.int64)
            plan=(starts[-1],cs,rows,every)
            if plan_cache is not None:setattr(plan_cache,'_exact_fi_plan',plan)
        count,cs,rows,every=plan
        checkpoints=torch.empty((max(count,1),H,V,K),device=k.device,dtype=torch.float32)
        out,final=fi_chunk_gated_delta_rule(q=q,k=k,v=v,g=g,beta=beta,initial_state=initial_state,
            output_final_state=True,cu_seqlens=cu_seqlens,use_qk_l2norm_in_kernel=False,
            state_checkpoints=checkpoints[:count] if count else None,
            checkpoint_cu_starts=cs if count else None,checkpoint_every_n_tokens=every if count else 0)
        gather_base[(triton.cdiv(H*V*K,256),n)](checkpoints,initial_state,snap,rows,H*V*K,256)
    else:
        raise NotImplementedError(backend)
    replay_endpoint[(triton.cdiv(V,32),n*H)](k,v,g,beta,snap,endpoints,cu_seqlens,
        H,HG,K,V,triton.next_power_of_2(K),32,BT,num_warps=4)
    return out.to(q.dtype),final,snap

@triton.jit
def gather_base(CP,H0,OUT,ROWS,S:tl.constexpr,B:tl.constexpr):
    x=tl.program_id(0)*B+tl.arange(0,B);row=tl.program_id(1)
    idx=tl.load(ROWS+row)
    if idx>=0:state=tl.load(CP+idx*S+x,x<S,other=0).to(tl.float32)
    else:state=tl.load(H0+row*S+x,x<S,other=0).to(tl.float32)
    tl.store(OUT+row*S+x,state,x<S)
