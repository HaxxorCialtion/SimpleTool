import Foundation
import Metal

struct SwiGLUArguments { var n:UInt32,k:UInt32 }
enum SwiGLUError:Error { case unavailable,invalidShape,shortBuffer,unsafeAlias,unsupportedSIMD }

final class SwiGLUKernelManager {
    private let device:MTLDevice,library:MTLLibrary
    private var pipelines=[Int:MTLComputePipelineState]()
    private let lock=NSLock()
    init(device:MTLDevice,library:MTLLibrary) { self.device=device;self.library=library }
    func pipeline(batch:Int,rows:Int,compact:Bool) throws -> MTLComputePipelineState {
        guard (2...4).contains(batch),[1,2,4,8].contains(rows) else { throw SwiGLUError.invalidShape }
        let key=batch*100+rows+(compact ? 1000 : 0)
        lock.lock();defer { lock.unlock() }
        if let p=pipelines[key] { return p }
        let c=MTLFunctionConstantValues();var b=UInt32(batch),r=UInt32(rows),narrow=compact
        c.setConstantValue(&b,type:.uint,index:0);c.setConstantValue(&r,type:.uint,index:1)
        c.setConstantValue(&narrow,type:.bool,index:2)
        let f=try library.makeFunction(name:"simpletool_int4_swiglu",constantValues:c)
        let p=try device.makeComputePipelineState(function:f)
        guard p.threadExecutionWidth==32 else { throw SwiGLUError.unsupportedSIMD }
        pipelines[key]=p;return p
    }
    // Buffers must be disjoint in underlying storage, not only object identity.
    func encode(command:MTLCommandBuffer,batch:Int,n:Int,k:Int,x:MTLBuffer,
                gate:[MTLBuffer],up:[MTLBuffer],output:MTLBuffer,rows:Int=4,
                threads:Int=128,compact:Bool=true) throws {
        guard (2...4).contains(batch),n>0,k>0,k%64==0,gate.count==3,up.count==3,
              n<=Int(UInt32.max)/32,k<=Int(UInt32.max)/batch,n<=Int(UInt32.max)/(k/8),
              [1,2,4,8].contains(rows),[64,128,256].contains(threads) else { throw SwiGLUError.invalidShape }
        let bytes=compact ? 2 : 4
        guard x.length>=batch*k*4,output.length>=batch*n*4,
              [gate,up].allSatisfy({$0[0].length>=n*k/8*4 && $0[1].length>=n*k/64*bytes && $0[2].length>=n*k/64*bytes})
        else { throw SwiGLUError.shortBuffer }
        for buffer in [x]+gate+up { if buffer === output { throw SwiGLUError.unsafeAlias } }
        let p=try pipeline(batch:batch,rows:rows,compact:compact)
        guard p.maxTotalThreadsPerThreadgroup>=threads else { throw SwiGLUError.unsupportedSIMD }
        guard let e=command.makeComputeCommandEncoder() else { throw SwiGLUError.unavailable }
        defer { e.endEncoding() };e.setComputePipelineState(p)
        e.label="int4 SwiGLU B=\(batch) R=\(rows) compact=\(compact)"
        for (i,b) in ([x]+gate+up+[output]).enumerated() { e.setBuffer(b,offset:0,index:i) }
        var args=SwiGLUArguments(n:UInt32(n),k:UInt32(k))
        e.setBytes(&args,length:MemoryLayout<SwiGLUArguments>.stride,index:8)
        let perGroup=rows*threads/32
        e.dispatchThreadgroups(MTLSize(width:(n+perGroup-1)/perGroup,height:1,depth:1),
                               threadsPerThreadgroup:MTLSize(width:threads,height:1,depth:1))
    }
}
