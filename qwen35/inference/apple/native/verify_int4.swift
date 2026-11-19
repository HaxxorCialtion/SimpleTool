import Foundation
import Metal

@main struct VerifyInt4 {
    static func main() throws {
        let root=URL(fileURLWithPath:CommandLine.arguments[1])
        let compact=CommandLine.arguments.contains("--compact-metadata")
        guard let device=MTLCreateSystemDefaultDevice(), let queue=device.makeCommandQueue()
        else { throw Int4Error.unavailable }
        let library=try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[2]))
        let manager=Int4KernelManager(device:device,library:library)
        let start=Date(); try manager.warmPipelines(rowSets:[1,2,4,8],compactMetadata:compact)
        let warmSeconds=Date().timeIntervalSince(start)
        func buffer(_ name:String) throws -> MTLBuffer {
            let data=try Data(contentsOf:root.appendingPathComponent(name))
            return try data.withUnsafeBytes { bytes in
                guard let b=device.makeBuffer(bytes:bytes.baseAddress!,length:data.count,options:.storageModeShared)
                else { throw Int4Error.unavailable }; return b
            }
        }
        let packed=try buffer("packed.bin"), scales=try buffer("scales.bin"), biases=try buffer("biases.bin")
        var results=[[String:Any]]()
        for m in 1...4 { for widths in [[79],[37,42],[37,23,7,12]] { for rows in [1,2,4,8] {
            let x=try buffer("x\(m).bin")
            let outputs=try widths.map { width -> MTLBuffer in
                guard let b=device.makeBuffer(length:m*width*4,options:.storageModeShared)
                else { throw Int4Error.unavailable }; return b
            }
            guard let command=queue.makeCommandBuffer() else { throw Int4Error.unavailable }
            try manager.encode(commandBuffer:command,batch:m,inputWidth:256,outputWidths:widths,
                               x:x,packed:packed,scales:scales,biases:biases,result:outputs,rows:rows,compactMetadata:compact)
            command.commit(); command.waitUntilCompleted()
            if let error=command.error { throw error }
            let expectedData=try Data(contentsOf:root.appendingPathComponent("expected\(m).bin"))
            let expected=expectedData.withUnsafeBytes { Array($0.bindMemory(to:Float.self)) }
            var offset=0, maxError:Float=0, bitwise=true
            for (buffer,width) in zip(outputs,widths) {
                let actual=buffer.contents().bindMemory(to:Float.self,capacity:m*width)
                for b in 0..<m { for row in 0..<width {
                    let a=actual[b*width+row], e=expected[b*79+offset+row]
                    maxError=max(maxError,abs(a-e)); bitwise=bitwise && a.bitPattern==e.bitPattern
                } }; offset+=width
            }
            guard bitwise && maxError==0 else { throw Int4Error.invalidShape }
            results.append(["batch":m,"widths":widths,"rows_per_simd":rows,"max_absolute_error":maxError,
                            "bitwise_equal_MLX_custom":bitwise])
        } } }
        let report:[String:Any]=["device":device.name,"pipeline_warm_s":warmSeconds,"compact_metadata":compact,
                                "results":results,"iPhone_runtime_tested":false]
        print(String(data:try JSONSerialization.data(withJSONObject:report,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
