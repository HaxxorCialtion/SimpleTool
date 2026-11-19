import Foundation
import Metal

struct Int4Arguments {
    var n: UInt32, k: UInt32, c0: UInt32, c1: UInt32, c2: UInt32, c3: UInt32
}

enum Int4Error: Error { case invalidShape, shortBuffer, unsupportedSIMD, unavailable }

/// Independent of Python and MLX. Reuse one manager on the engine's GPU queue.
/// Weights: uint32, eight low-to-high nibbles; scale/bias: float32 or lossless
/// float16 storage. Input, output and all arithmetic stay float32.
final class Int4KernelManager {
    private let device: MTLDevice
    private let library: MTLLibrary
    private var pipelines: [Int: MTLComputePipelineState] = [:]
    private let lock=NSLock()
    init(device: MTLDevice, library: MTLLibrary) {
        self.device=device; self.library=library
    }

    func pipeline(batch: Int, outputs: Int, rows: Int=1, compactMetadata: Bool=false) throws -> MTLComputePipelineState {
        guard (1...4).contains(batch), [1,2,4].contains(outputs), [1,2,4,8].contains(rows)
        else { throw Int4Error.invalidShape }
        lock.lock(); defer { lock.unlock() }
        let key=batch*100+outputs*10+rows+(compactMetadata ? 1000 : 0)
        if let existing=pipelines[key] { return existing }
        let constants=MTLFunctionConstantValues()
        var m=UInt32(batch), count=UInt32(outputs)
        constants.setConstantValue(&m,type:.uint,index:0)
        constants.setConstantValue(&count,type:.uint,index:1)
        var compact=compactMetadata
        constants.setConstantValue(&compact,type:.bool,index:3)
        if rows>1 {
            var r=UInt32(rows)
            constants.setConstantValue(&r,type:.uint,index:2)
        }
        let name=rows==1 ? "simpletool_int4" : "simpletool_int4_row_reuse"
        let function=try library.makeFunction(name:name,constantValues:constants)
        let result=try device.makeComputePipelineState(function:function)
        guard result.threadExecutionWidth==32, result.maxTotalThreadsPerThreadgroup>=128
        else { throw Int4Error.unsupportedSIMD }
        pipelines[key]=result
        return result
    }

    /// Twelve pipelines cover M=1/2/3/4 and ordinary, two-output, four-output
    /// projections. Dimensions are runtime arguments, not extra PSO keys.
    func warmPipelines(rowSets: [Int]=[1], compactMetadata: Bool=false) throws {
        for rows in rowSets {
            for m in 1...4 { for count in [1,2,4] { _=try pipeline(batch:m,outputs:count,rows:rows,compactMetadata:compactMetadata) } }
        }
    }

    func encode(commandBuffer: MTLCommandBuffer, batch: Int, inputWidth: Int,
                outputWidths: [Int], x: MTLBuffer, packed: MTLBuffer,
                scales: MTLBuffer, biases: MTLBuffer, result: [MTLBuffer],
                rows: Int=1, threads: Int=128, compactMetadata: Bool=false) throws {
        guard (1...4).contains(batch), inputWidth>0, inputWidth%64==0,
              [1,2,4].contains(outputWidths.count), outputWidths.allSatisfy({$0>0}),
              result.count==outputWidths.count, [1,2,4,8].contains(rows),
              [64,128,256,512].contains(threads) else { throw Int4Error.invalidShape }
        let n=outputWidths.reduce(0,+)
        guard n<=Int(UInt32.max)/32, inputWidth<=Int(UInt32.max),
              n<=Int(UInt32.max)/(inputWidth/8),
              inputWidth<=Int(UInt32.max)/batch else { throw Int4Error.invalidShape }
        let metadataBytes=compactMetadata ? 2 : 4
        guard x.length>=batch*inputWidth*4, packed.length>=n*(inputWidth/8)*4,
              scales.length>=n*(inputWidth/64)*metadataBytes, biases.length>=n*(inputWidth/64)*metadataBytes,
              zip(result,outputWidths).allSatisfy({$0.0.length>=batch*$0.1*4})
        else { throw Int4Error.shortBuffer }
        let p=try pipeline(batch:batch,outputs:outputWidths.count,rows:rows,compactMetadata:compactMetadata)
        guard p.maxTotalThreadsPerThreadgroup>=threads else { throw Int4Error.unsupportedSIMD }
        guard let encoder=commandBuffer.makeComputeCommandEncoder() else { throw Int4Error.unavailable }
        encoder.label="SimpleTool int4 M=\(batch) outputs=\(outputWidths.count) rows=\(rows)"
        encoder.setComputePipelineState(p)
        for (i,buffer) in [x,packed,scales,biases].enumerated() { encoder.setBuffer(buffer,offset:0,index:i) }
        for i in 0..<4 { encoder.setBuffer(result[min(i,result.count-1)],offset:0,index:4+i) }
        let widths=outputWidths+Array(repeating:0,count:4-outputWidths.count)
        var args=Int4Arguments(n:UInt32(n),k:UInt32(inputWidth),c0:UInt32(widths[0]),
                              c1:UInt32(widths[1]),c2:UInt32(widths[2]),c3:UInt32(widths[3]))
        encoder.setBytes(&args,length:MemoryLayout<Int4Arguments>.stride,index:8)
        let rowsPerGroup=rows*(threads/32)
        encoder.dispatchThreadgroups(MTLSize(width:(n+rowsPerGroup-1)/rowsPerGroup,height:1,depth:1),
                                     threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
        encoder.endEncoding()
    }
}
