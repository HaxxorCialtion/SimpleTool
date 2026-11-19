import Cmlx
import MLX

func updateHistory(_ source:MLXArray,_ value:MLXArray,_ phase:MLXArray)->MLXArray {
    var output=mlx_array_new()
    let index=phase.reshaped(1).asType(.int32)
    let status=mlx_slice_update_dynamic(&output,source.ctx,value.ctx,index.ctx,[Int32(1)],1,StreamOrDevice.default.ctx)
    precondition(status==0,"fixed history update failed")
    return MLXArray(output)
}
extension Int4Model {
    func fixedHistoryRecurrence(_ q:MLXArray,_ k:MLXArray,_ v:MLXArray,_ g:MLXArray,_ beta:MLXArray,_ cache:LayerState)->MLXArray {
        let base=cache.matrix!,keys=cache.deferredKeys!,gates=cache.deferredGates!,deltas=cache.deferredDeltas!
        let b=q.dim(0),write=cache.writeHistory,phase=cache.historyPhase!,fused=historyPolicy == .fused
        precondition(keys.dim(1)==deferredInterval-1 && (base.dim(0)==1 || base.dim(0)==b))
        var shapes=[v.shape,[b,c.valueHeads,c.valueDim],write ? [b,c.valueHeads,c.valueDim,128] : [1]]
        if fused {shapes += write ? [[1],[1],[1]] : [keys.shape,gates.shape,deltas.shape]}
        let r=(fused ? bodies.fusedHistory : bodies.fixedHistory)([q,k,v,g,beta,base,keys,gates,deltas,phase],
            template:[("SB",base.dim(0)),("C",deferredInterval-1),("Hk",c.keyHeads),("Hv",c.valueHeads),("Dv",c.valueDim),("WRITE",write)],
            grid:(32,c.valueDim/8,b*c.valueHeads),threadGroup:(32,2,1),outputShapes:shapes,
            outputDTypes:Array(repeating:.float32,count:shapes.count))
        if write {cache.matrix=r[2]}
        else if fused {cache.deferredKeys=r[3];cache.deferredGates=r[4];cache.deferredDeltas=r[5]}
        else {
            cache.deferredKeys=updateHistory(keys,k,phase)
            cache.deferredGates=updateHistory(gates,g,phase)
            cache.deferredDeltas=updateHistory(deltas,r[1].expandedDimensions(axis:1),phase)
        }
        return r[0]
    }
}
