"""Fresh-image bridge vs exact common-prefix KV reuse; fixed four-channel lengths."""
import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path
from simpletool.decode import batched_generate,prefill
from simpletool.loading import load_model
from simpletool.vision import image_prefix


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--model',required=True)
    p.add_argument('--image',type=Path,required=True)
    p.add_argument('--vision-gguf',type=Path,required=True)
    p.add_argument('--mmproj',type=Path,required=True)
    p.add_argument('--tokens',type=int,default=12)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    model,tok=load_model(args.model,kernel='custom')
    banned=tuple(tok.convert_tokens_to_ids(t) for t in ('<|endoftext|>','<|im_end|>'))
    messages=[{'role':'user','content':'屏幕上现在打开的是什么应用？'}]
    exporter=Path('artifacts/vision_export_official')
    def prepare(cache_dir):
        return image_prefix(model,tok,messages,args.image,args.vision_gguf,args.mmproj,exporter,cache_dir)
    warm_dir=Path(tempfile.mkdtemp(prefix='image_bench_warm_',dir='artifacts'))
    ids,emb,pos,next_pos,extra=prepare(warm_dir)
    source,_=prefill(model,ids,embeddings=emb,positions=pos,next_position=next_pos)
    def run(warm):
        start=time.perf_counter()
        directory=warm_dir if warm else Path(tempfile.mkdtemp(prefix='image_bench_cold_',dir='artifacts'))
        current_ids,emb,pos,next_pos,vision=prepare(directory)
        if warm:
            assert current_ids==ids and vision['vision_encoding_calls']==0
            result=batched_generate(model,tok,ids,args.tokens,prefix_cache=source,
                                    ignore_stops=True,suppress_tokens=banned)
        else:
            assert vision['vision_encoding_calls']==1
            result=batched_generate(model,tok,current_ids,args.tokens,embeddings=emb,positions=pos,
                                    next_position=next_pos,ignore_stops=True,suppress_tokens=banned)
        wall=time.perf_counter()-start
        assert all(len(c['token_ids'])==args.tokens and not set(c['token_ids']).intersection(banned)
                   for c in result['channels'].values())
        result['stats'].update(vision)
        result['stats']['wall_s']=wall
        result['stats']['total_s']+=vision['vision_preparation_s']
        result['stats']['reused_prefix_tokens']=len(ids) if warm else 0
        return result['stats']
    run(False);run(True)
    samples={'fresh_image':[],'warm_image_and_prefix':[]}
    for i in range(3):
        order=[('fresh_image',False),('warm_image_and_prefix',True)]
        if i%2:order.reverse()
        for name,warm in order:
            row=run(warm);samples[name].append(row)
            print(name,i,row['wall_s'],'encoding calls',row['vision_encoding_calls'],flush=True)
    summary={name:{k:statistics.median(row[k] for row in rows) for k in
        ('wall_s','prefill_s','decode_s','vision_bridge_s','vision_preparation_s','vision_encoding_calls')}
             for name,rows in samples.items()}
    result={'model':str(Path(args.model).resolve()),'image':str(args.image.resolve()),
            'vision_exporter':str(exporter.resolve()),'tokens_per_head':args.tokens,
            'compute':'float32','kernel':'custom','suppressed_token_ids':list(banned),
            'prompt_token_count':len(ids),'summaries':summary,'samples':samples,
            'competing_workloads_uncontrolled':True,
            'limitations':'Warm case repeats the exact same common prompt, not an added history turn. Cold bridge includes process/projector loading; official server keeps projector resident.'}
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
