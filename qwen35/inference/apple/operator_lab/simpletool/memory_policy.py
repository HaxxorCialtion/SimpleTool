"""Configure only this process's reusable MLX buffer allocator cache."""
import mlx.core as mx


def configure_allocator_cache(limit_mib=None):
    result={'configured':limit_mib is not None,'requested_mib':limit_mib,
            'scope':'MLX free-buffer allocator cache in this process only',
            'live_tensor_limit':False,'hard_cap':False}
    if limit_mib is not None:
        if not isinstance(limit_mib,int) or limit_mib<0:raise ValueError('Allocator cache MiB must be nonnegative')
        result['previous_limit_bytes']=mx.set_cache_limit(limit_mib*1024*1024)
        result['requested_bytes']=limit_mib*1024*1024
    return result
