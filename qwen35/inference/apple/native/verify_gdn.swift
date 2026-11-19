import Foundation
import Metal

@main struct VerifyGDN {
    static func main() throws {
        let root=URL(fileURLWithPath:CommandLine.arguments[1])
        guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else { throw GDNError.unavailable }
        let library=try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[2]))
        let manager=try GDNKernelManager(device:device,library:library)
        func buffer(_ path:URL) throws -> MTLBuffer {
            let data=try Data(contentsOf:path)
            return try data.withUnsafeBytes { bytes in
                guard let value=device.makeBuffer(bytes:bytes.baseAddress!,length:data.count,options:.storageModeShared)
                else { throw GDNError.unavailable };return value
            }
        }
        func allocate(_ bytes:Int) throws -> MTLBuffer {
            guard let value=device.makeBuffer(length:bytes,options:.storageModeShared) else { throw GDNError.unavailable };return value
        }
        func equal(_ b:MTLBuffer,_ expected:URL) throws -> Bool {
            let data=try Data(contentsOf:expected)
            return data.withUnsafeBytes { raw in
                let ref=raw.bindMemory(to:UInt32.self),got=b.contents().bindMemory(to:UInt32.self,capacity:ref.count)
                return (0..<ref.count).allSatisfy { ref[$0]==got[$0] }
            }
        }
        func error(_ b:MTLBuffer,_ expected:URL) throws -> Float {
            let data=try Data(contentsOf:expected)
            return data.withUnsafeBytes { raw in
                let ref=raw.bindMemory(to:Float.self),got=b.contents().bindMemory(to:Float.self,capacity:ref.count)
                return (0..<ref.count).map { abs(ref[$0]-got[$0]) }.max() ?? 0
            }
        }
        var records=[[String:Any]]()
        for (hk,hv) in [(16,32),(16,16),(8,16)] {
            let folder=root.appendingPathComponent("h\(hk)_\(hv)")
            let prefix=try buffer(folder.appendingPathComponent("prefix.bin"))
            for batch in 1...4 { for inplace in [false,true] { for simdGroups in [2,4] {
                var state=try allocate(batch*hv*128*128*4)
                var outputsEqual=true,statesEqual=true
                for step in 1...2 {
                    let stem="b\(batch)_s\(step)_"
                    let args=try ["q","k","v","g","beta"].map { try buffer(folder.appendingPathComponent(stem+$0+".bin")) }
                    let y=try allocate(batch*hv*128*4)
                    let target=step==1 || inplace ? state : try allocate(batch*hv*128*128*4)
                    guard let command=queue.makeCommandBuffer() else { throw GDNError.unavailable }
                    try manager.encode(commandBuffer:command,batch:batch,sourceBatch:step==1 ? 1 : batch,
                        keyHeads:hk,valueHeads:hv,q:args[0],k:args[1],v:args[2],g:args[3],beta:args[4],
                        source:step==1 ? prefix : state,y:y,stateOut:target,sourceIsOwned:step==2 && inplace,simdGroups:simdGroups)
                    command.commit();command.waitUntilCompleted();if let error=command.error { throw error }
                    let yEqual=try equal(y,folder.appendingPathComponent(stem+"y.bin"))
                    let stateEqual=try equal(target,folder.appendingPathComponent(stem+"state.bin"))
                    if !yEqual || !stateEqual {
                        let ye=try error(y,folder.appendingPathComponent(stem+"y.bin"))
                        let se=try error(target,folder.appendingPathComponent(stem+"state.bin"))
                        FileHandle.standardError.write(Data("mismatch B=\(batch) HK=\(hk) HV=\(hv) step=\(step) inPlace=\(inplace) y=\(ye) state=\(se)\n".utf8))
                    }
                    outputsEqual=outputsEqual && yEqual
                    statesEqual=statesEqual && stateEqual
                    state=target
                }
                let unchanged=try equal(prefix,folder.appendingPathComponent("prefix.bin"))
                guard outputsEqual,statesEqual,unchanged else { throw GDNError.invalidShape }
                records.append(["batch":batch,"key_heads":hk,"value_heads":hv,"steps":2,
                                "second_step_in_place":inplace,"simdgroups":simdGroups,"output_bitwise_equal":outputsEqual,
                                "state_bitwise_equal":statesEqual,"prefix_unchanged":unchanged])
            } } }
        }
        print(String(data:try JSONSerialization.data(withJSONObject:["device":device.name,"cases":records,
            "iPhone_runtime_tested":false,"scope":"FP32 recurrence only; no model weights or end-to-end token parity"],
            options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
