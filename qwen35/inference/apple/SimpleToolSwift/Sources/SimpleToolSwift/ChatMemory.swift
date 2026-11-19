import Cmlx
import MLX

/// Only call on the exclusive inference lane after dropping the old model.
/// Compiled tapes can retain weight constants independently of Swift objects.
/// Shader pipelines stay reusable; model-specific compiled tapes are discarded.
public func releaseChatRuntimeCaches() {
    Stream.gpu.synchronize()
    Stream.cpu.synchronize()
    precondition(mlx_detail_compile_clear_cache()==0,"Could not release compiled graph cache")
    Memory.clearCache()
}
