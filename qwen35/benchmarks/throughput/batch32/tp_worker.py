"""Private worker extension. TP ranks use their own GDN/conv and CUDA graphs.
Only control-plane receipts use collective_rpc; no RPC per decode token.
"""
import os,types,time

class SimpleToolWorker:
    def init_worker_extension(self):
        import prefill_runtime as rt
        rt.install()
        original=self.load_model
        def load(worker,*args,**kwargs):
            out=original(*args,**kwargs)
            from fast_prefill import PublicPrefillGraphs
            runner=worker.model_runner
            worker._sl_fast=PublicPrefillGraphs(runner,rt,max_graphs=int(os.getenv('SL_MAX_PREFILL_GRAPHS','24')))
            worker._sl_fast.enabled=False
            worker._sl_fast.allow_capture=False
            original_execute=runner.execute_model
            def execute(scheduler_output,*args,**kwargs):
                for rid in scheduler_output.finished_req_ids:
                    rt.PRIMES.discard(rid)
                for req in scheduler_output.scheduled_new_reqs:
                    params=req.sampling_params
                    m=(params.extra_args or {}).get('simpletool_prefill') if params else None
                    if m and m['role']=='prime':rt.PRIMES.add(req.req_id)
                rt.CURRENT_STEP+=1
                return original_execute(scheduler_output,*args,**kwargs)
            runner.execute_model=execute
            return out
        self.load_model=types.MethodType(load,self)

    def sl_control(self,enabled=False,capture=False,verify=False,trace=False):
        import prefill_runtime as rt
        rt.ACTIVE=trace
        rt.EVENTS.clear();rt.GPU_SPANS.clear()
        f=self._sl_fast;f.enabled=enabled;f.allow_capture=capture;f.verify=verify
        return {'rank':self.rank,'graphs':len(f.graphs),'enabled':enabled,'capture':capture,'verify':verify}

    def sl_receipt(self):
        import torch,prefill_runtime as rt
        torch.cuda.synchronize()
        out={'rank':self.rank,'fast_prefill':self._sl_fast.receipt(),
             'events':list(rt.EVENTS),
             'forward_spans':[{'step':i,'ms':x.elapsed_time(y),'scheduled_tokens':t} for i,x,y,t in rt.GPU_SPANS],
             'gpu_memory':{'allocated':torch.cuda.memory_allocated(),'reserved':torch.cuda.memory_reserved(),
                           'max_allocated':torch.cuda.max_memory_allocated(),
                           'free':torch.cuda.mem_get_info()[0]}}
        rt.EVENTS.clear();rt.GPU_SPANS.clear()
        return out
