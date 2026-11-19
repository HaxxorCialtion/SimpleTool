// Empirical FP32 scalar-FMA capacity; not model throughput or hardware peak.
import Foundation
import Metal
import Darwin

enum CapacityError:Error {case invalid(String)}
guard CommandLine.arguments.count==2 else {throw CapacityError.invalid("output.json")}
let device=MTLCreateSystemDefaultDevice()!,queue=device.makeCommandQueue()!
let chainCounts=[1,4,8,16],iterations=4096
var source="#include <metal_stdlib>\nusing namespace metal;\n"
for chains in chainCounts {
    source += "kernel void capacity_\(chains)(device float* out [[buffer(0)]], constant uint& iterations [[buffer(1)]],uint tid [[thread_position_in_grid]]) {\nfloat seed=float(tid%16)*0.03125f+0.5f;\n"
    for c in 0..<chains {source += "float a\(c)=seed+float(\(c))*0.0625f;\n"}
    source += "for(uint i=0;i<iterations;++i) {\n"
    for c in 0..<chains {source += "a\(c)=fma(a\(c),1.0000001192092896f,0.000001f);\n"}
    source += "}\nfloat sum=a0;\n"
    for c in 1..<chains {source += "sum+=a\(c);\n"}
    source += "out[tid]=sum;\n}\n"
}
let options=MTLCompileOptions();options.fastMathEnabled=false
let library=try device.makeLibrary(source:source,options:options)
var pipelines=[Int:MTLComputePipelineState]()
for c in chainCounts {pipelines[c]=try device.makeComputePipelineState(function:library.makeFunction(name:"capacity_\(c)")!)}
struct Setting {let chains:Int,n:Int,threads:Int;var key:String {"C\(chains)_N\(n)_T\(threads)"}}
let settings=chainCounts.flatMap{c in [65536,262144].flatMap{n in [128,256].map{Setting(chains:c,n:n,threads:$0)}}}
let output=device.makeBuffer(length:262144*4,options:.storageModeShared)!
var references=[Int:[UInt32]]()
for c in chainCounts {
    references[c]=(0..<16).map {tid in
        let seed=Float(tid)*0.03125+0.5
        var values=(0..<c).map{seed+Float($0)*0.0625}
        for _ in 0..<iterations {for j in 0..<c {values[j]=fmaf(values[j],Float(1.0000001192092896),Float(0.000001))}}
        var sum=values[0];for j in 1..<c {sum+=values[j]};return sum.bitPattern
    }
}
let clock=ContinuousClock()
func elapsed(_ duration:Duration)->Double {let v=duration.components;return Double(v.seconds)+Double(v.attoseconds)*1e-18}
func run(_ s:Setting)throws ->(Double,Double) {
    let cb=queue.makeCommandBuffer()!,start=clock.now,encoder=cb.makeComputeCommandEncoder()!
    var it=UInt32(iterations)
    encoder.setComputePipelineState(pipelines[s.chains]!)
    encoder.setBuffer(output,offset:0,index:0);encoder.setBytes(&it,length:4,index:1)
    encoder.dispatchThreads(MTLSize(width:s.n,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:s.threads,height:1,depth:1))
    encoder.endEncoding();cb.commit();cb.waitUntilCompleted()
    guard cb.status == .completed,cb.gpuEndTime>cb.gpuStartTime else {throw CapacityError.invalid("GPU failure")}
    let wall=elapsed(start.duration(to:clock.now))
    let raw=output.contents().bindMemory(to:UInt32.self,capacity:s.n),gold=references[s.chains]!
    guard (0..<s.n).allSatisfy({raw[$0]==gold[$0%16]}) else {throw CapacityError.invalid("FP32 fmaf output bit mismatch \(s.key)")}
    return (cb.gpuEndTime-cb.gpuStartTime,wall)
}
var first=[String:[Double]](),gpu=[String:[Double]](),wall=gpu
for s in settings {let r=try run(s);first[s.key]=[r.0,r.1];gpu[s.key]=[];wall[s.key]=[]}
for round in 0..<7 {
    let shift=(5*round)%settings.count
    for s in Array(settings[shift...])+Array(settings[..<shift]) {
        let r=try run(s);gpu[s.key]!.append(r.0);wall[s.key]!.append(r.1)
    }
}
let records=settings.map{s -> [String:Any] in
    let seconds=gpu[s.key]!.sorted()[3],ops=Double(s.n)*Double(iterations)*Double(s.chains)*2
    return ["chains":s.chains,"threads":s.threads,"elements":s.n,"iterations":iterations,
            "fma_count":ops/2,"counted_flops":ops,"gpu_median_s":seconds,"gpu_samples_s":gpu[s.key]!,
            "wall_median_s":wall[s.key]!.sorted()[3],"wall_samples_s":wall[s.key]!,
            "first_gpu_wall_s":first[s.key]!,"empirical_FP32_TFLOPs":ops/seconds/1e12,"all_output_bitwise_fmaf":true]
}
let result:[String:Any]=["device":device.name,"records":records,"shader_source":source,
    "compute":"float32","fast_math":false,"first_plus_rotating_rounds":8,"all_passed":true,
    "limits":"Empirical scalar register-resident FP32 FMA workload, useful FMA=2 FLOPs; loop/initialization/sum/store included in GPU duration. Not advertised physical peak, int4 unpack/convert capacity, GEMM or full-model throughput. External workloads and frequency uncontrolled."]
try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[1]))
print(device.name,"best empirical FP32 TFLOPs",records.compactMap{$0["empirical_FP32_TFLOPs"] as? Double}.max()!)
