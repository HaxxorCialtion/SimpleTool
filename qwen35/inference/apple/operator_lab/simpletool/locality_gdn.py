"""Experimental dispatch permutations; exact arithmetic, immutable shared seed.

head: neighbouring z planes read the same head across branches.
value: neighbouring y SIMD groups read the same head/value across branches.
Owned states use the old dispatch. No cache locality benefit is assumed.
"""
import mlx.core as mx
from .fixed_history_gdn import FixedHistoryState,FIXED_STEP,FUSED_STEP,FIXED_MATERIALIZE
from .deferred_gdn import SETUP

HEAD_SETUP=SETUP.replace(
    'const uint n=thread_position_in_grid.z,b=n/Hv,hv=n%Hv,hk=hv/(Hv/Hk);',
    'const uint z=thread_position_in_grid.z;\n'
    '    const uint b=SB==1 ? z%B : z/Hv,hv=SB==1 ? z/B : z%Hv;\n'
    '    const uint n=b*Hv+hv,hk=hv/(Hv/Hk);')
VALUE_SETUP=SETUP.replace(
    'const uint n=thread_position_in_grid.z,b=n/Hv,hv=n%Hv,hk=hv/(Hv/Hk);',
    'const uint z=thread_position_in_grid.z;\n'
    '    const uint b=SB==1 ? thread_position_in_grid.y%B : z/Hv,hv=SB==1 ? z : z%Hv;\n'
    '    const uint n=b*Hv+hv,hk=hv/(Hv/Hk);').replace(
    'thread_position_in_grid.y*8',
    '(SB==1 ? thread_position_in_grid.y/B : thread_position_in_grid.y)*8')

def make_state(order):
    if order not in ('head','value'):raise ValueError('dispatch order must be head/value')
    setup=HEAD_SETUP if order=='head' else VALUE_SETUP
    sources={name:setup+body[len(SETUP):] for name,body in
             [('step',FIXED_STEP),('fused',FUSED_STEP),('materialize',FIXED_MATERIALIZE)]}
    inputs=['q','k','v','g','beta','base','history_k','history_g','history_delta','pending']
    kernels={}
    for name,body in sources.items():
        outputs=['state_out'] if name=='materialize' else ['y','delta_out','state_out']
        if name=='fused':outputs+=['history_k_out','history_g_out','history_delta_out']
        kernels[name]=mx.fast.metal_kernel(name=f'simpletool_locality_{order}_{name}',
            input_names=inputs[5:] if name=='materialize' else inputs,
            output_names=outputs,source=body)
    class LocalityState(FixedHistoryState):
        step_kernel=kernels['step'];fused_step_kernel=kernels['fused'];materialize_kernel=kernels['materialize']
        def dispatch(self):
            if order=='value' and self.base.shape[0]==1:
                return (32,self.dv//8*self.batch,self.hv),(32,2,1)
            return super().dispatch()
    # B is an explicit template even for owned states; the shared kernel uses it
    # for the bijection. Append it without changing the kernel body arithmetic.
    def bind(kernel):
        def run(**kwargs):
            b=kwargs['inputs'][0].shape[0] if len(kwargs['inputs'])==10 else kwargs['output_shapes'][0][0]
            kwargs['template']=[*kwargs['template'],('B',b)]
            return kernel(**kwargs)
        return staticmethod(run)
    LocalityState.step_kernel=bind(kernels['step'])
    LocalityState.fused_step_kernel=bind(kernels['fused'])
    LocalityState.materialize_kernel=bind(kernels['materialize'])
    LocalityState.sources=sources;LocalityState.order=order
    return LocalityState

HeadLocalityState=make_state('head')
ValueLocalityState=make_state('value')
STATES={'baseline':FixedHistoryState,'head':HeadLocalityState,'value':ValueLocalityState}
