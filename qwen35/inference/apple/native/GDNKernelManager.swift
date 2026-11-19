import Foundation
import Metal

struct GDNArguments { var batch: UInt32, stateBatch: UInt32, keyHeads: UInt32, valueHeads: UInt32, valueWidth: UInt32 }
enum GDNError: Error { case invalidShape, shortBuffer, unsafeAlias, unavailable, unsupportedSIMD }

/// One-step FP32 recurrence, independent of Python/MLX. Buffers must not
/// overlap, except an explicitly owned source matrix may also be stateOut.
/// The normalized q/k and precomputed scalar gates must match model semantics.
final class GDNKernelManager {
    private let pipelines: [MTLComputePipelineState]
    init(device: MTLDevice, library: MTLLibrary) throws {
        pipelines=try [false,true].map { owned in
            let constants=MTLFunctionConstantValues();var value=owned
            constants.setConstantValue(&value,type:.bool,index:0)
            let function=try library.makeFunction(name:"simpletool_gdn_step",constantValues:constants)
            let pipeline=try device.makeComputePipelineState(function:function)
            guard pipeline.threadExecutionWidth==32, pipeline.maxTotalThreadsPerThreadgroup>=64
            else { throw GDNError.unsupportedSIMD };return pipeline
        }
    }

    func encode(commandBuffer: MTLCommandBuffer,batch: Int,sourceBatch: Int,
                keyHeads: Int,valueHeads: Int,valueWidth: Int=128,
                q: MTLBuffer,k: MTLBuffer,v: MTLBuffer,g: MTLBuffer,beta: MTLBuffer,
                source: MTLBuffer,y: MTLBuffer,stateOut: MTLBuffer,
                sourceIsOwned: Bool=false,simdGroups: Int=2) throws {
        guard let encoder=commandBuffer.makeComputeCommandEncoder() else { throw GDNError.unavailable }
        defer { encoder.endEncoding() }
        try encode(encoder:encoder,batch:batch,sourceBatch:sourceBatch,keyHeads:keyHeads,valueHeads:valueHeads,
                   valueWidth:valueWidth,q:q,k:k,v:v,g:g,beta:beta,source:source,y:y,stateOut:stateOut,
                   sourceIsOwned:sourceIsOwned,simdGroups:simdGroups)
    }

    /// Append disjoint layer dispatches to one caller-owned encoder. The
    /// caller must add a barrier before a dispatch reads a previous write.
    func encode(encoder: MTLComputeCommandEncoder,batch: Int,sourceBatch: Int,
                keyHeads: Int,valueHeads: Int,valueWidth: Int=128,
                q: MTLBuffer,k: MTLBuffer,v: MTLBuffer,g: MTLBuffer,beta: MTLBuffer,
                source: MTLBuffer,y: MTLBuffer,stateOut: MTLBuffer,
                sourceIsOwned: Bool=false,simdGroups: Int=2) throws {
        guard (1...4).contains(batch), sourceBatch==1 || sourceBatch==batch,
              0<keyHeads,keyHeads<=valueHeads,valueHeads<=4096,valueHeads%keyHeads==0,
              0<valueWidth,valueWidth<=4096,valueWidth%32==0,
              valueHeads<=Int(UInt32.max)/(batch*valueWidth*128),[2,4].contains(simdGroups)
        else { throw GDNError.invalidShape }
        let inputCount=batch*keyHeads*128, outputCount=batch*valueHeads*valueWidth
        guard q.length>=inputCount*4,k.length>=inputCount*4,v.length>=outputCount*4,
              g.length>=batch*valueHeads*4,beta.length>=batch*valueHeads*4,
              source.length>=sourceBatch*valueHeads*valueWidth*128*4,
              y.length>=outputCount*4,stateOut.length>=outputCount*128*4
        else { throw GDNError.shortBuffer }
        if source === stateOut && (!sourceIsOwned || sourceBatch != batch) { throw GDNError.unsafeAlias }
        let owned=source === stateOut
        if y === source || y === stateOut { throw GDNError.unsafeAlias }
        for buffer in [q,k,v,g,beta] {
            if buffer === y || buffer === stateOut { throw GDNError.unsafeAlias }
        }
        let pipeline=pipelines[owned ? 1 : 0]
        guard pipeline.maxTotalThreadsPerThreadgroup>=32*simdGroups else { throw GDNError.unsupportedSIMD }
        encoder.label="SimpleTool GDN FP32 B=\(batch) sourceB=\(sourceBatch) owned=\(sourceIsOwned)"
        encoder.setComputePipelineState(pipeline)
        for (i,buffer) in [q,k,v,g,beta,source,y].enumerated() { encoder.setBuffer(buffer,offset:0,index:i) }
        // Owned specialization writes through buffer5; its unused buffer7
        // is eliminated at specialization, avoiding a duplicate state binding.
        encoder.setBuffer(owned ? nil : stateOut,offset:0,index:7)
        var args=GDNArguments(batch:UInt32(batch),stateBatch:UInt32(sourceBatch),keyHeads:UInt32(keyHeads),
                              valueHeads:UInt32(valueHeads),valueWidth:UInt32(valueWidth))
        encoder.setBytes(&args,length:MemoryLayout<GDNArguments>.stride,index:8)
        encoder.dispatchThreadgroups(MTLSize(width:1,height:valueWidth/(8*simdGroups),depth:batch*valueHeads),
                                     threadsPerThreadgroup:MTLSize(width:32,height:simdGroups,depth:1))
    }
}
