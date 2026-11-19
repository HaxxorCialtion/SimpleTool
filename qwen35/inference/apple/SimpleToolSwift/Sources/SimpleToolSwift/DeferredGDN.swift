// Immutable FP32 base and exact-order rounded update history. Never mutate a
// borrowed MLX buffer; materialize only when rebasing or exporting dense state.
import MLX

extension LayerState {
    func clearDeferred(_ config:EngineConfig,capacity:Int=0) {
        let batch=conv!.dim(0)
        deferredKeys=MLXArray.zeros([batch,capacity,config.keyHeads,128],dtype:.float32)
        deferredGates=MLXArray.zeros([batch,capacity,config.valueHeads],dtype:.float32)
        deferredDeltas=MLXArray.zeros([batch,capacity,config.valueHeads,config.valueDim],dtype:.float32)
        historyPhase=capacity>0 ? MLXArray(Int32(0)) : nil
    }
    var compiledArrays:[MLXArray] {
        if prefixK != nil {return [k!,v!]}
        if let keys=deferredKeys {return [conv!,matrix!,keys,deferredGates!,deferredDeltas!]}
        return [conv!,matrix!]
    }
    func commitCompiled(_ arrays:ArraySlice<MLXArray>) {
        let a=Array(arrays)
        if prefixK != nil {k=a[0];v=a[1]}
        else {
            conv=a[0];matrix=a[1]
            if a.count==5 {deferredKeys=a[2];deferredGates=a[3];deferredDeltas=a[4]}
        }
    }
}

extension Int4Model {
    func deferredRecurrence(_ q:MLXArray,_ k:MLXArray,_ v:MLXArray,_ g:MLXArray,_ beta:MLXArray,_ cache:LayerState)->MLXArray {
        if historyPolicy != .dynamic {return fixedHistoryRecurrence(q,k,v,g,beta,cache)}
        let base=cache.matrix!,keys=cache.deferredKeys!,gates=cache.deferredGates!,deltas=cache.deferredDeltas!
        let b=q.dim(0),pending=keys.dim(1),write=pending+1==deferredInterval
        precondition(pending<deferredInterval && (base.dim(0)==1 || base.dim(0)==b))
        let r=bodies.deferred([q,k,v,g,beta,base,keys,gates,deltas],
            template:[("SB",base.dim(0)),("P",pending),("Hk",c.keyHeads),("Hv",c.valueHeads),("Dv",c.valueDim),("WRITE",write)],
            grid:(32,c.valueDim/8,b*c.valueHeads),threadGroup:(32,2,1),
            outputShapes:[v.shape,[b,c.valueHeads,c.valueDim],write ? [b,c.valueHeads,c.valueDim,128] : [1]],
            outputDTypes:[.float32,.float32,.float32])
        if write {cache.matrix=r[2];cache.clearDeferred(c)}
        else {
            cache.deferredKeys=concatenated([keys,k],axis:1)
            cache.deferredGates=concatenated([gates,g],axis:1)
            cache.deferredDeltas=concatenated([deltas,r[1].expandedDimensions(axis:1)],axis:1)
        }
        return r[0]
    }
    func denseMatrix(_ cache:LayerState)->MLXArray {
        let base=cache.matrix!
        guard let keys=cache.deferredKeys else {return base}
        let b=cache.conv!.dim(0),p=keys.dim(1),shape=[b,c.valueHeads,c.valueDim,128]
        if historyPolicy != .dynamic {
            return bodies.fixedMaterialize([base,keys,cache.deferredGates!,cache.deferredDeltas!,cache.historyPhase!],
                template:[("SB",base.dim(0)),("C",deferredInterval-1),("Hk",c.keyHeads),("Hv",c.valueHeads),("Dv",c.valueDim)],
                grid:(32,c.valueDim/8,b*c.valueHeads),threadGroup:(32,2,1),outputShapes:[shape],outputDTypes:[.float32])[0]
        }
        if p==0 {return broadcast(base,to:shape)}
        return bodies.materialize([base,keys,cache.deferredGates!,cache.deferredDeltas!],
            template:[("SB",base.dim(0)),("P",p),("Hk",c.keyHeads),("Hv",c.valueHeads),("Dv",c.valueDim)],
            grid:(32,c.valueDim/8,b*c.valueHeads),threadGroup:(32,2,1),outputShapes:[shape],outputDTypes:[.float32])[0]
    }
    func canonicalState(_ caches:[LayerState])->[MLXArray] {
        caches.flatMap{$0.prefixK==nil ? [$0.conv!,denseMatrix($0)] : [$0.k!,$0.v!]}
    }
}
