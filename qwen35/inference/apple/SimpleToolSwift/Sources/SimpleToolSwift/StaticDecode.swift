import Foundation
import Cmlx
import MLX

public enum DecodeExecution:String {case dynamic,staticCache,compiled}

// Expose the pinned MLX-C dynamic slice-update primitive without changing the
// dependency. Functional MLX ownership applies: never write a borrowed buffer.
func updateTail(_ source:MLXArray,_ value:MLXArray,_ offset:MLXArray)->MLXArray {
    var output=mlx_array_new()
    let indices=offset.reshaped(1).asType(.int32)
    let status=mlx_slice_update_dynamic(&output,source.ctx,value.ctx,indices.ctx,[Int32(2)],1,StreamOrDevice.default.ctx)
    precondition(status==0,"dynamic KV update failed")
    return MLXArray(output)
}

final class CompiledDecode {
    let caches:[LayerState]
    private let eagerModel:Int4Model?
    private let evaluationStride:Int
    let function:@Sendable ([MLXArray])->[MLXArray]
    let markerFunction:@Sendable ([MLXArray])->[MLXArray]
    let rawFunction:@Sendable ([MLXArray])->[MLXArray]
    let writeFunction:@Sendable ([MLXArray])->[MLXArray]
    let writeMarkerFunction:@Sendable ([MLXArray])->[MLXArray]
    private var blocks=[Int:@Sendable ([MLXArray])->[MLXArray]]()
    private var fixedBlocks=[Int:@Sendable ([MLXArray])->[MLXArray]]()
    private(set) var used=0
    let capacity:Int
    let config:EngineConfig
    let deferredInterval:Int
    let historyPolicy:HistoryPolicy
    init(_ model:Int4Model,_ caches:[LayerState],compileStep:Bool=true,evaluationStride:Int=0) {
        precondition(!model.traceEnabled,"whole-step compilation excludes retaining diagnostic traces")
        precondition([0,4,8,16].contains(evaluationStride))
        precondition(evaluationStride==0 || (!compileStep && model.deferredInterval==1))
        self.caches=caches
        self.eagerModel=evaluationStride>0 ? model : nil
        self.evaluationStride=evaluationStride
        config=model.c;deferredInterval=model.deferredInterval;historyPolicy=model.historyPolicy
        if deferredInterval>1 {
            for state in caches where state.prefixK==nil {state.clearDeferred(model.c,capacity:model.historyPolicy == .dynamic ? 0 : model.deferredInterval-1)}
        }
        capacity=caches.first(where:{$0.prefixK != nil})!.k!.dim(2)
        let prefix=caches.map{($0.prefixK,$0.prefixV)}
        let widths=caches.map{$0.compiledArrays.count}
        let offsets=widths.indices.map{2+widths.prefix($0).reduce(0,+)}
        func makeFunction(project:Bool,write:Bool=false)->@Sendable ([MLXArray])->[MLXArray] {
          { (arguments:[MLXArray])->[MLXArray] in
            let states=(0..<model.c.layers).map{i -> LayerState in
                let state=LayerState()
                let offset=offsets[i]
                if let pk=prefix[i].0,let pv=prefix[i].1 {
                    state.prefixK=pk;state.prefixV=pv
                    state.k=arguments[offset];state.v=arguments[offset+1]
                    state.tailUsed=arguments[1]
                } else {
                    state.conv=arguments[offset];state.matrix=arguments[offset+1]
                    if widths[i]==5 {
                        state.deferredKeys=arguments[offset+2];state.deferredGates=arguments[offset+3];state.deferredDeltas=arguments[offset+4]
                        if model.historyPolicy != .dynamic {
                            state.historyPhase=arguments[1]%Int32(model.deferredInterval);state.writeHistory=write
                        }
                    }
                }
                return state
            }
            let hidden=model.hidden(arguments[0],cache:states)
            let prediction=project ? model.linear(hidden[0...,-1,0...],"embed_tokens").argMax(axis:-1).asType(.int32)
                : MLXArray.zeros([arguments[0].dim(0)],dtype:.int32)
            return [prediction]+states.flatMap{$0.compiledArrays}
        }
        }
        rawFunction=makeFunction(project:true)
        function=compileStep ? compile(shapeless:false,rawFunction) : rawFunction
        let marker=makeFunction(project:false),write=makeFunction(project:true,write:true),writeMarker=makeFunction(project:false,write:true)
        markerFunction=compileStep ? compile(shapeless:false,marker) : marker
        writeFunction=compileStep ? compile(shapeless:false,write) : write
        writeMarkerFunction=compileStep ? compile(shapeless:false,writeMarker) : writeMarker
    }
    private func flatState()->[MLXArray] {
        caches.flatMap{$0.compiledArrays}
    }
    private func commit(_ states:[MLXArray],advances:Int) {
        used+=advances
        let phase=historyPolicy == .dynamic ? nil : MLXArray(Int32(used%deferredInterval))
        var offset=0
        for state in caches {
            let width=state.compiledArrays.count
            state.commitCompiled(states[offset..<(offset+width)]);offset+=width
            if state.prefixK != nil {state.tailUsed=MLXArray(Int32(used))}
            else if historyPolicy != .dynamic {state.historyPhase=phase}
        }
    }
    // Returns [count,B]. Last prediction remains pending, so the terminal
    // token is never required to be consumed into cache. Intermediate tokens
    // and the corresponding state transitions stay in the GPU graph.
    func advancePending(_ prediction:MLXArray,count:Int)->MLXArray {
        precondition((1...8).contains(count) && used+count-1<=capacity)
        if count==1 {return prediction.reshaped(1,prediction.dim(0))}
        if historyPolicy != .dynamic {return advanceFixed(prediction,count:count)}
        if blocks[count]==nil {
            let raw=rawFunction
            blocks[count]=compile(shapeless:false) { (arguments:[MLXArray])->[MLXArray] in
                var current=arguments[0],states=Array(arguments.dropFirst(2)),predictions=[current]
                for i in 0..<(count-1) {
                    let result=raw([current.reshaped(current.dim(0),1),arguments[1]+Int32(i)]+states)
                    current=result[0];states=Array(result.dropFirst());predictions.append(current)
                }
                return [stacked(predictions,axis:0)]+states
            }
        }
        let result=blocks[count]!([prediction,MLXArray(Int32(used))]+flatState())
        commit(Array(result.dropFirst()),advances:count-1)
        return result[0]
    }
    func forward(_ tokens:MLXArray,project:Bool=true)->MLXArray {
        precondition(used<capacity,"fixed KV capacity exceeded")
        if let model=eagerModel {
            // Avoid flatState()/arguments retaining every previous-layer state
            // until an entire forward graph returns. This is still one true B4
            // forward, with the same single shared prefix and per-head GDN.
            let hidden=model.hidden(tokens,cache:caches,evaluateEvery:evaluationStride)
            let prediction=project ? model.linear(hidden[0...,-1,0...],"embed_tokens").argMax(axis:-1).asType(.int32)
                : MLXArray.zeros([tokens.dim(0)],dtype:.int32)
            used+=1
            return prediction
        }
        let write=historyPolicy != .dynamic && used%deferredInterval==deferredInterval-1
        let f=write ? (project ? writeFunction : writeMarkerFunction) : (project ? function : markerFunction)
        let outputs=f([tokens,MLXArray(Int32(used))]+flatState())
        commit(Array(outputs.dropFirst()),advances:1)
        // No persistent array snapshot of the initial B1 seed or B4 state.
        return outputs[0]
    }
    func reset(_ initial:[LayerState]) {
        used=0
        for (state,seed) in zip(caches,initial) {
            state.conv=seed.conv;state.matrix=seed.matrix;state.k=seed.k;state.v=seed.v
            state.tailUsed=seed.tailUsed;state.offset=seed.offset
            if deferredInterval>1 && state.prefixK==nil {state.clearDeferred(config,capacity:historyPolicy == .dynamic ? 0 : deferredInterval-1)}
        }
    }
    private func advanceFixed(_ prediction:MLXArray,count:Int)->MLXArray {
        var remaining=count-1,step=used,current=prediction,states=flatState()
        var pieces=[prediction.reshaped(1,prediction.dim(0))]
        while remaining>0 {
            let phase=step%deferredInterval,n:Int,part:MLXArray
            if phase==deferredInterval-1 {
                let result=writeFunction([current.reshaped(current.dim(0),1),MLXArray(Int32(step))]+states)
                current=result[0];states=Array(result.dropFirst());n=1;part=current.reshaped(1,current.dim(0))
            } else {
                n=min(remaining,deferredInterval-1-phase)
                if fixedBlocks[n]==nil {
                    let raw=rawFunction
                    fixedBlocks[n]=compile(shapeless:false) { (arguments:[MLXArray])->[MLXArray] in
                        var prediction=arguments[0],states=Array(arguments.dropFirst(2)),predictions=[MLXArray]()
                        for i in 0..<n {
                            let result=raw([prediction.reshaped(prediction.dim(0),1),arguments[1]+Int32(i)]+states)
                            prediction=result[0];states=Array(result.dropFirst());predictions.append(prediction)
                        }
                        return [stacked(predictions,axis:0)]+states
                    }
                }
                let result=fixedBlocks[n]!([current,MLXArray(Int32(step))]+states)
                part=result[0];current=part[-1];states=Array(result.dropFirst())
            }
            pieces.append(part);remaining-=n;step+=n
        }
        commit(states,advances:count-1)
        return concatenated(pieces,axis:0)
    }
}
