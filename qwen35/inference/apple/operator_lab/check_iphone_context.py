"""Exercise bounded context eviction and compare native tokens to full recompute."""
from pathlib import Path
import json, subprocess
import mlx.core as mx
from simpletool.loading import load_model
from simpletool.decode import HEADS, prompt_ids, prefill, batched_generate
from simpletool.stream_grouped_qmv import install_stream_grouped_qmv
from simpletool.fused_swiglu_int4 import install_fused_swiglu_int4
from simpletool.compiled import CompiledStep

root=Path(__file__).resolve().parent
fixture=json.loads((root/'reports/iphone_chat_fixture_20261008.json').read_text())
fixture.update(contextLimit=256,maxTokens=8,inputs=[
    '你好，我叫小明。','今天我有点累。','给我一点鼓励。','我想喝杯茶。',
    '外面开始下雨了。','推荐一个休息方式。','明天有什么计划？','再和我聊一会儿。'])
fp=root/'reports/iphone_chat_context_fixture_20261008.json'
out=root/'reports/iphone_chat_context_native_20261008.json'
fp.write_text(json.dumps(fixture,ensure_ascii=False,indent=2))
subprocess.run([str(root/'artifacts/swift_build/Build/Products/Release/SimpleToolSwiftCLI'),
    '--chat-check',str(root/'artifacts/2b_int4_compact'),str(fp),str(out)],check=True)
native=json.loads(out.read_text())
mx.set_cache_limit(128*1024**2)
model,tok=load_model(root/'artifacts/2b_int4_compact',kernel='custom',compact_metadata=True)
install_stream_grouped_qmv(model);install_fused_swiglu_int4(model)
step=CompiledStep(model)
history=[];checks=[]
for actual in native:
    recent=[]
    current={'role':'user','content':actual['user']}
    system={'role':'system','content':fixture['system']}
    for older in reversed(history):
        proposed=older+recent
        if len(prompt_ids(tok,[system]+proposed+[current]))>fixture['contextLimit']:break
        recent=proposed
    ids=prompt_ids(tok,[system]+recent+[current])
    cache,_=prefill(model,ids,chunk_size=64)
    result=batched_generate(model,tok,ids,fixture['maxTokens'],prefix_cache=cache,compiled_step=step)
    parity={h:result['channels'][h]['token_ids']==actual['tokenIDs'][h] for h in HEADS}
    checks.append(dict(user=actual['user'],prompt_tokens=len(ids),cache_hit=actual['cacheHitTokens'],
                       retained_turns=len(recent)//2,token_parity=parity))
    history.append([current,{'role':'assistant','content':''.join(
        f'<{h}>{actual["channels"][h]}</{h}>' for h in HEADS)}])
    print(checks[-1],flush=True)
assert any(c['retained_turns']<i for i,c in enumerate(checks)), 'No eviction exercised'
assert all(all(c['token_parity'].values()) for c in checks), 'Native token divergence'
assert all(c['prompt_tokens']<=256 for c in checks)
(root/'reports/iphone_chat_context_parity_20261008.json').write_text(json.dumps(
    dict(all_passed=True,eviction_exercised=True,checks=checks),ensure_ascii=False,indent=2)+'\n')
