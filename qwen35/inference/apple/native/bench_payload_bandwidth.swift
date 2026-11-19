// Native Metal calibration using the actual packed language payload as raw bits.
// No model inference, dequantization, borrowed MLX buffer or FP16 arithmetic.
import Foundation
import Metal

enum CalibrationError:Error {case invalid(String)}
let arguments=CommandLine.arguments
guard arguments.count==3 else {throw CalibrationError.invalid("checkpoint.safetensors output.json")}
let device=MTLCreateSystemDefaultDevice()!,queue=device.makeCommandQueue()!
let source="""
#include <metal_stdlib>
using namespace metal;
constant uint IPT [[function_constant(0)]];
kernel void read_payload(device const uint* data [[buffer(0)]],device uint* sums [[buffer(1)]],
 constant uint& words [[buffer(2)]],uint tid [[thread_index_in_threadgroup]],
 uint3 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]],
 uint simd [[simdgroup_index_in_threadgroup]]) {
 uint acc=0;
 const uint base=group.x*256*IPT+tid*4;
 for(uint i=0;i<IPT/4;++i) {
  const uint offset=base+i*256*4;
  if(offset+3<words) {
   const uint4 values=*((device const uint4*)(data+offset));
   acc+=values.x+values.y+values.z+values.w;
  } else {for(uint j=0;j<4;++j) if(offset+j<words) acc+=data[offset+j];}
 }
 const uint total=simd_sum(acc);
 if(lane==0) sums[group.x*8+simd]=total;
}
"""
let library=try device.makeLibrary(source:source,options:nil)
let sizes=[4,16,64,256]
var pipelines=[Int:MTLComputePipelineState]()
for n in sizes {
    let constants=MTLFunctionConstantValues();var value=UInt32(n)
    constants.setConstantValue(&value,type:.uint,index:0)
    pipelines[n]=try device.makeComputePipelineState(function:library.makeFunction(name:"read_payload",constantValues:constants))
    guard pipelines[n]!.threadExecutionWidth==32,pipelines[n]!.maxTotalThreadsPerThreadgroup>=256
    else {throw CalibrationError.invalid("requires SIMD32 and 256 threads")}
}
struct Payload {let inputs:[MTLBuffer],outputs:[MTLBuffer],destinations:[MTLBuffer],words:[Int],bytes:Int,checksum:UInt32}
func loadPayload()->Payload {
    let data=try! Data(contentsOf:URL(fileURLWithPath:arguments[1]),options:.mappedIfSafe)
    let headerSize=data.withUnsafeBytes{Int(UInt64(littleEndian:$0.loadUnaligned(as:UInt64.self)))}
    let object=try! JSONSerialization.jsonObject(with:data.subdata(in:8..<(8+headerSize))) as! [String:Any]
    precondition(object.keys.allSatisfy{$0=="__metadata__" || $0.hasPrefix("language_model.")},"language-only payload required")
    let offset=8+headerSize,total=data.count-offset,chunk=64*1024*1024
    precondition(total>0 && total%4==0)
    var inputs=[MTLBuffer](),outputs=[MTLBuffer](),destinations=[MTLBuffer](),words=[Int](),checksum:UInt32=0
    data.withUnsafeBytes {raw in
        for start in stride(from:0,to:total,by:chunk) {
            let bytes=min(chunk,total-start),pointer=raw.baseAddress!.advanced(by:offset+start),count=bytes/4
            let input=device.makeBuffer(bytes:pointer,length:bytes,options:.storageModeShared)!
            let ptr=input.contents().bindMemory(to:UInt32.self,capacity:count)
            for i in 0..<count {checksum &+= ptr[i]}
            inputs.append(input);words.append(count)
            outputs.append(device.makeBuffer(length:((count+256*4-1)/(256*4))*8*4,options:.storageModeShared)!)
            destinations.append(device.makeBuffer(length:bytes,options:.storageModeShared)!)
        }
    }
    return Payload(inputs:inputs,outputs:outputs,destinations:destinations,words:words,bytes:total,checksum:checksum)
}
let payload=loadPayload()
func run(_ key:String)->(gpu:Double,wall:Double) {
    let cb=queue.makeCommandBuffer()!,start=Date()
    if key=="copy" {
        let encoder=cb.makeBlitCommandEncoder()!
        for i in payload.inputs.indices {encoder.copy(from:payload.inputs[i],sourceOffset:0,to:payload.destinations[i],destinationOffset:0,size:payload.words[i]*4)}
        encoder.endEncoding()
    } else {
        let n=Int(key)!,encoder=cb.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(pipelines[n]!)
        for i in payload.inputs.indices {
            var count=UInt32(payload.words[i]);let groups=(Int(count)+256*n-1)/(256*n)
            encoder.setBuffer(payload.inputs[i],offset:0,index:0);encoder.setBuffer(payload.outputs[i],offset:0,index:1)
            encoder.setBytes(&count,length:4,index:2)
            encoder.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1))
        }
        encoder.endEncoding()
    }
    cb.commit();cb.waitUntilCompleted()
    precondition(cb.status == .completed && cb.gpuEndTime>cb.gpuStartTime,"GPU command failed")
    return (cb.gpuEndTime-cb.gpuStartTime,Date().timeIntervalSince(start))
}
let keys=sizes.map{String($0)}+["copy"]
var gpu=[String:[Double]](),wall=[String:[Double]](),correct=[String:Bool]()
for key in keys {gpu[key]=[];wall[key]=[];_ = run(key)}
for roundIndex in 0..<7 {
    let shift=roundIndex%keys.count
    for key in Array(keys[shift...])+Array(keys[..<shift]) {
        let sample=run(key);gpu[key]!.append(sample.gpu);wall[key]!.append(sample.wall)
        if key=="copy" {
            correct[key]=payload.inputs.indices.allSatisfy{memcmp(payload.inputs[$0].contents(),payload.destinations[$0].contents(),payload.words[$0]*4)==0}
        } else {
            let n=Int(key)!;var sum:UInt32=0
            for i in payload.inputs.indices {
                let groups=(payload.words[i]+256*n-1)/(256*n),ptr=payload.outputs[i].contents().bindMemory(to:UInt32.self,capacity:groups*8)
                for j in 0..<(groups*8) {sum &+= ptr[j]}
            }
            correct[key]=sum==payload.checksum
        }
        precondition(correct[key] == true,"payload checksum/copy mismatch")
    }
}
let medians=gpu.mapValues{$0.sorted()[$0.count/2]},wallMedians=wall.mapValues{$0.sorted()[$0.count/2]}
let gbps=Dictionary(uniqueKeysWithValues:keys.map{($0,Double(payload.bytes)*($0=="copy" ? 2.0:1.0)/medians[$0]!/1e9)})
let result:[String:Any]=["device":device.name,"payload_bytes":payload.bytes,"chunk_bytes":64*1024*1024,"chunk_count":payload.inputs.count,
 "gpu_samples_s":gpu,"wall_samples_s":wall,"gpu_median_s":medians,"wall_median_s":wallMedians,
 "useful_payload_GBps":gbps,"checksum_or_copy_equal":correct,"cpu_raw_sum_mod32":payload.checksum,
 "native_buffer_bytes":payload.inputs.reduce(0){$0+$1.length}+payload.outputs.reduce(0){$0+$1.length}+payload.destinations.reduce(0){$0+$1.length},
 "scope":"actual packed int4 language payload read as uint32; read checksum or blit copy; seven rotating rounds",
 "limitations":"useful bytes divided by command-buffer GPU duration; not DRAM counters or model kernel performance; buffers include full destination clone; external load uncontrolled"]
try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:arguments[2]))
print(device.name,payload.bytes,gbps,correct)
