import Foundation
import Metal

// Isolated recurrence workload: actual 4B head dimensions and 24 independent
// layers, but fixed fixture q/k/v/g/beta, no projections, attention or model.
@main struct BenchGDN {
    static func main() throws {
        let root=URL(fileURLWithPath:CommandLine.arguments[1]).appendingPathComponent("h16_32")
        guard let device=MTLCreateSystemDefaultDevice(),let queue=device.makeCommandQueue() else { throw GDNError.unavailable }
        let library=try device.makeLibrary(URL:URL(fileURLWithPath:CommandLine.arguments[2]))
        let manager=try GDNKernelManager(device:device,library:library)
        let layers=24,batch=4,hk=16,hv=32,width=128,steps=32,rounds=5
        func load(_ name:String) throws -> MTLBuffer {
            let data=try Data(contentsOf:root.appendingPathComponent(name+".bin"))
            return try data.withUnsafeBytes { raw in
                guard let b=device.makeBuffer(bytes:raw.baseAddress!,length:data.count,options:.storageModeShared)
                else { throw GDNError.unavailable };return b
            }
        }
        func allocate(_ bytes:Int) throws -> MTLBuffer {
            guard let b=device.makeBuffer(length:bytes,options:.storageModeShared) else { throw GDNError.unavailable };return b
        }
        func median(_ values:[Double]) -> Double { values.sorted()[values.count/2] }
        var records=[[String:Any]]()
        for inplace in [false,true] { for combined in [false,true] { for sg in [2,4] {
            // Separate buffers for each layer: combining encoders introduces
            // no producer/consumer dependencies between these dispatches.
            let inputs=try (0..<layers).map { _ in try ["q","k","v","g","beta"].map { try load("b4_s2_"+$0) } }
            var states=try (0..<layers).map { _ in try load("b4_s1_state") }
            var spare=inplace ? [] : try (0..<layers).map { _ in try allocate(batch*hv*width*128*4) }
            let outputs=try (0..<layers).map { _ in try allocate(batch*hv*width*4) }
            let totalBytes=(inputs.flatMap{$0}+states+spare+outputs).reduce(0){$0+$1.length}
            func run() throws -> Double {
                guard let command=queue.makeCommandBuffer() else { throw GDNError.unavailable }
                let encoder=combined ? command.makeComputeCommandEncoder() : nil
                if combined && encoder == nil { throw GDNError.unavailable }
                for i in 0..<layers {
                    let x=inputs[i],target=inplace ? states[i] : spare[i]
                    if let encoder=encoder {
                        try manager.encode(encoder:encoder,batch:batch,sourceBatch:batch,keyHeads:hk,valueHeads:hv,
                            q:x[0],k:x[1],v:x[2],g:x[3],beta:x[4],source:states[i],y:outputs[i],stateOut:target,
                            sourceIsOwned:inplace,simdGroups:sg)
                    } else {
                        try manager.encode(commandBuffer:command,batch:batch,sourceBatch:batch,keyHeads:hk,valueHeads:hv,
                            q:x[0],k:x[1],v:x[2],g:x[3],beta:x[4],source:states[i],y:outputs[i],stateOut:target,
                            sourceIsOwned:inplace,simdGroups:sg)
                    }
                }
                encoder?.endEncoding();command.commit();command.waitUntilCompleted()
                if let error=command.error { throw error }
                if !inplace { swap(&states,&spare) }
                return command.gpuEndTime-command.gpuStartTime
            }
            // Confirm this exact scheduling path against step-two fixture
            // before its recurrent inputs begin evolving in the benchmark.
            _=try run()
            let expected=try Data(contentsOf:root.appendingPathComponent("b4_s2_state.bin"))
            let expectedY=try Data(contentsOf:root.appendingPathComponent("b4_s2_y.bin"))
            for i in 0..<layers {
                guard Data(bytes:states[i].contents(),count:expected.count)==expected,
                      Data(bytes:outputs[i].contents(),count:expectedY.count)==expectedY
                else { throw GDNError.invalidShape }
            }
            for _ in 0..<8 { _=try run() }
            var gpuSamples=[Double](),wallSamples=[Double]()
            for _ in 0..<rounds {
                var gpu=0.0;let started=ProcessInfo.processInfo.systemUptime
                for _ in 0..<steps { gpu += try run() }
                wallSamples.append((ProcessInfo.processInfo.systemUptime-started)/Double(steps))
                gpuSamples.append(gpu/Double(steps))
            }
            records.append(["in_place":inplace,"single_encoder":combined,"simdgroups":sg,
                "allocated_buffer_bytes":totalBytes,"state_buffer_bytes":(states+spare).reduce(0){$0+$1.length},
                "first_step_output_and_state_bitwise_equal":true,"median_gpu_s":median(gpuSamples),
                "median_wall_s":median(wallSamples),"gpu_samples_s":gpuSamples,"wall_samples_s":wallSamples])
        } } }
        print(String(data:try JSONSerialization.data(withJSONObject:["device":device.name,"batch":batch,
            "layers":layers,"key_heads":hk,"value_heads":hv,"width":width,"rounds":rounds,"steps_per_round":steps,
            "scope":"isolated FP32 GDN recurrence; fixed fixtures; no full model or int4 weights",
            "iPhone_runtime_tested":false,"competing_workloads_uncontrolled":true,"results":records],
            options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    }
}
