"""Experimental fixed-group shared-prefix decode. FP16 KV; no quantized weight changes.
Prefix tiles load a KV head once and multiply B*GQA query rows together.
This is logical kernel load reuse, not a measurement of physical DRAM traffic.
"""
import math
import torch
import triton
import triton.language as tl

@triton.jit
def prefix_kernel(Q,KV,BT,LENS,PO,PL,
 B:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
 N:tl.constexpr,PAGE:tl.constexpr,SPLITS:tl.constexpr,
 QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,
 KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
 BTS0:tl.constexpr,SCALE:tl.constexpr,BM:tl.constexpr,BN:tl.constexpr):
 kh=tl.program_id(0); part=tl.program_id(1)
 r=tl.arange(0,BM); d=tl.arange(0,D); t=part*BN+tl.arange(0,BN)
 G:tl.constexpr=HQ//HK
 branch=r//G; qh=kh*G+r%G
 q=tl.load(Q+branch[:,None]*QS0+qh[:,None]*QS1+d[None,:]*QS2,branch[:,None]<B,0)
 # During graph capture seq_len may be shorter than the configured prefix.
 valid_n=tl.minimum(N,tl.load(LENS))
 pages=tl.load(BT+t//PAGE,t<valid_n,0)
 off=pages[:,None]*KS0+kh*KS1+(t%PAGE)[:,None]*KS2+d[None,:]*KS3
 k=tl.load(KV+off,t[:,None]<valid_n,0)
 v=tl.load(KV+off+D*KS3,t[:,None]<valid_n,0)
 scores=tl.dot(q,tl.trans(k))*SCALE
 scores=tl.where((t[None,:]<valid_n)&(branch[:,None]<B),scores,-float('inf'))
 m=tl.maximum(tl.max(scores,1),-1.e30)
 p=tl.exp(scores-m[:,None]);l=tl.sum(p,1)
 o=tl.dot(p.to(v.dtype),v)/tl.maximum(l[:,None],1.e-20)
 out_index=(part*B*HQ+branch*HQ+qh)
 tl.store(PO+out_index[:,None]*D+d[None,:],o,branch[:,None]<B)
 tl.store(PL+out_index,m+tl.log(l),branch<B)

@triton.jit
def merge_tail_kernel(Q,KV,BT,LENS,PO,PL,O,
 B:tl.constexpr,HQ:tl.constexpr,HK:tl.constexpr,D:tl.constexpr,
 N:tl.constexpr,PAGE:tl.constexpr,SPLITS:tl.constexpr,RS:tl.constexpr,TAIL:tl.constexpr,
 QS0:tl.constexpr,QS1:tl.constexpr,QS2:tl.constexpr,
 KS0:tl.constexpr,KS1:tl.constexpr,KS2:tl.constexpr,KS3:tl.constexpr,
 BTS0:tl.constexpr,OS0:tl.constexpr,OS1:tl.constexpr,OS2:tl.constexpr,SCALE:tl.constexpr):
    branch=tl.program_id(0);qh=tl.program_id(1);kh=qh//(HQ//HK)
    d=tl.arange(0,D); s=tl.arange(0,RS)
    q=tl.load(Q+branch*QS0+qh*QS1+d*QS2)
    length=tl.load(LENS+branch)
    prefix_lse=tl.load(PL+s*B*HQ+branch*HQ+qh,s<SPLITS,-float('inf'))
    m=tl.maximum(tl.max(prefix_lse,0),-1.e30)
    weights=tl.exp(prefix_lse-m)
    po=tl.load(PO+(s[:,None]*B*HQ+branch*HQ+qh)*D+d[None,:],s[:,None]<SPLITS,0)
    acc=tl.sum(po*weights[:,None],0);denom=tl.sum(weights,0)
    # A 256xD FP32 private tail spills registers badly for D=256. Stream 16
    # tokens per step, using an online softmax merge with the prefix result.
    for begin in range(N,tl.minimum(length,N+TAIL),16):
        t=begin+tl.arange(0,16)
        pages=tl.load(BT+branch*BTS0+t//PAGE,t<length,0)
        off=pages[:,None]*KS0+kh*KS1+(t%PAGE)[:,None]*KS2+d[None,:]*KS3
        k=tl.load(KV+off,t[:,None]<length,0).to(tl.float32)
        score=tl.sum(k*q[None,:].to(tl.float32),1)*SCALE
        score=tl.where(t<length,score,-float('inf'))
        new_m=tl.maximum(m,tl.max(score,0));alpha=tl.exp(m-new_m)
        tail_weights=tl.exp(score-new_m)
        v=tl.load(KV+off+D*KS3,t[:,None]<length,0).to(tl.float32)
        acc=acc*alpha+tl.sum(v*tail_weights[:,None],0)
        denom=denom*alpha+tl.sum(tail_weights,0);m=new_m
    tl.store(O+branch*OS0+qh*OS1+d*OS2,acc/tl.maximum(denom,1.e-20))

def shared_decode(query,kv,block_table,seq_lens,prefix_n,output=None,tile=64,max_tail=256,workspace=None,phase='both'):
    b,hq,d=query.shape;hk=kv.shape[1];page=kv.shape[2]
    assert query.dtype==kv.dtype==torch.float16 and hq%hk==0
    assert b in (1,2,4,8) and d in (64,128,256)
    splits=triton.cdiv(prefix_n,tile)
    if output is None:output=torch.empty_like(query)
    output3=output.view(b,hq,d)
    if workspace is None:
        po=torch.empty((splits,b,hq,d),device=query.device,dtype=torch.float32)
        pl=torch.empty((splits,b,hq),device=query.device,dtype=torch.float32)
    else:po,pl=workspace
    if phase in ('both','prefix'):
        prefix_kernel[(hk,splits)](query,kv,block_table,seq_lens,po,pl,
          b,hq,hk,d,prefix_n,page,splits,*query.stride(),*kv.stride(),block_table.stride(0),
          d**-.5,max(16,triton.next_power_of_2(b*(hq//hk))),tile,num_warps=4)
    if phase in ('both','tail'):
        merge_tail_kernel[(b,hq)](query,kv,block_table,seq_lens,po,pl,output3,
          b,hq,hk,d,prefix_n,page,splits,triton.next_power_of_2(splits),max_tail,
          *query.stride(),*kv.stride(),block_table.stride(0),*output3.stride(),d**-.5,num_warps=4)
    return output

def install(prefix_n=700,max_tail=256):
    """Dedicated-process opt-in; caller guarantees ONE group with this prefix.
    General traffic, non-singleton queries and unsupported batch sizes use FA.
    Inputs sharing only a numerical length are NOT sufficient authorization.
    """
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
    original=FlashAttentionImpl.forward
    stats={'custom_calls':0,'fallback_calls':0,'prefix_n':prefix_n,'max_tail':max_tail}
    def forward(self,layer,query,key,value,kv_cache,attn_metadata,output,output_scale=None,output_block_scale=None):
        if (attn_metadata is not None and attn_metadata.max_query_len==1
            and attn_metadata.num_actual_tokens in (1,2,4,8)
            and self.head_size==256 and self.num_kv_heads==4
            and kv_cache.dtype==torch.float16 and query.dtype==torch.float16
            and self.dcp_world_size==1 and self.alibi_slopes is None
            and self.sliding_window==(-1,-1) and not self.logits_soft_cap
            and output_scale is None and output_block_scale is None):
            b=attn_metadata.num_actual_tokens
            q=query[:b].view(b,self.num_heads,self.head_size)
            stats['custom_calls']+=1
            return shared_decode(q,kv_cache,attn_metadata.block_table,
                attn_metadata.seq_lens,prefix_n,output[:b],max_tail=max_tail)
        stats['fallback_calls']+=1
        return original(self,layer,query,key,value,kv_cache,attn_metadata,output,output_scale,output_block_scale)
    FlashAttentionImpl.forward=forward
    return stats,original
