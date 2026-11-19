import Foundation
import Metal

@main struct VerifySwiGLU {
    static func main() throws {
        let root=URL(fileURLWithPath:CommandLine.arguments[1])
        guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else { throw SwiGLUError.unavailable }
        let library=try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[2]))
        let manager=SwiGLUKernelManager(device:device,library:library)
        var records=[[String:Any]]()
        for (label,n,k) in [("n79_k256",79,256),("n512_k2560",512,2560),("n256_k2048",256,2048),("actual4b",9216,2560)] {
            let folder=root.appendingPathComponent(label)
            func load(_ name:String) throws -> MTLBuffer {
                let d=try Data(contentsOf:folder.appendingPathComponent(name+".bin"))
                return try d.withUnsafeBytes { raw in
                    guard let b=device.makeBuffer(bytes:raw.baseAddress!,length:d.count,options:.storageModeShared)
                    else { throw SwiGLUError.unavailable };return b
                }
            }
            let gw=try load("gw"),uw=try load("uw")
            for compact in [false,true] {
                let suffix=compact ? "half" : "float"
                let gate=try [gw,load("gs_"+suffix),load("gb_"+suffix)]
                let up=try [uw,load("us_"+suffix),load("ub_"+suffix)]
                for batch in 2...4 {
                    let x=try load("x\(batch)")
                    let expected=try Data(contentsOf:folder.appendingPathComponent("expected\(batch).bin"))
                    for rows in [1,2,4,8] { for threads in [64,128,256] {
                        guard let out=device.makeBuffer(length:batch*n*4,options:.storageModeShared),
                              let command=queue.makeCommandBuffer() else { throw SwiGLUError.unavailable }
                        try manager.encode(command:command,batch:batch,n:n,k:k,x:x,gate:gate,up:up,output:out,
                                           rows:rows,threads:threads,compact:compact)
                        command.commit();command.waitUntilCompleted();if let error=command.error { throw error }
                        let actual=Data(bytes:out.contents(),count:expected.count)
                        guard actual==expected else {
                            let error=actual.withUnsafeBytes { a in expected.withUnsafeBytes { e in
                                let av=a.bindMemory(to:Float.self),ev=e.bindMemory(to:Float.self)
                                return (0..<av.count).map { abs(av[$0]-ev[$0]) }.max() ?? 0
                            } }
                            FileHandle.standardError.write(Data("mismatch \(label) B\(batch) R\(rows) T\(threads) compact=\(compact) maxAbs=\(error)\n".utf8))
                            throw SwiGLUError.invalidShape
                        }
                        records.append(["label":label,"batch":batch,"rows":rows,"threads":threads,
                            "compact_metadata":compact,"bitwise_equal":true,"actual_model_weights":label=="actual4b"])
                    } }
                }
            }
        }
        print(String(data:try JSONSerialization.data(withJSONObject:["device":device.name,"cases":records,
            "scope":"fused int4 gate/up and SwiGLU; actual 4B first-layer tensors and structural fixtures",
            "iPhone_runtime_tested":false],options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
