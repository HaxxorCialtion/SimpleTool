// Isolated cache-hot GPU command timings. Input buffers immutable; exact output
// gold supplied by the FP32 MLX kernel. No weight/model/iPhone timing claim.
import Foundation
import Metal
let args=CommandLine.arguments
precondition(args.count==3,"manifest.json output.json")
let manifest=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:args[1]))) as! [String:Any]
let device=MTLCreateSystemDefaultDevice()!,queue=device.makeCommandQueue()!
let options=MTLCompileOptions();options.fastMathEnabled=false
let library=try device.makeLibrary(source:String(contentsOfFile:manifest["metal_source"] as! String,encoding:.utf8),options:options)
let orders=["baseline","head","value"],repeats=160,rounds=7
var results=[[String:Any]]()
for record in manifest["cases"] as! [[String:Any]] {
 let inputData=try (record["input_files"] as! [String]).map{try Data(contentsOf:URL(fileURLWithPath:$0))}
 let gold=try (record["gold_files"] as! [String]).map{try Data(contentsOf:URL(fileURLWithPath:$0))}
 let inputs=inputData.map{data in data.withUnsafeBytes{device.makeBuffer(bytes:$0.baseAddress!,length:data.count,options:.storageModeShared)!}}
 let outputs=gold.map{device.makeBuffer(length:$0.count,options:.storageModeShared)!}
 let names=record["functions"] as! [String:String],grids=record["grids"] as! [String:[Int]]
 var pipelines=[String:MTLComputePipelineState](),gpu=[String:[Double]](),wall=[String:[Double]](),equal=[String:Bool]()
 for order in orders {
  pipelines[order]=try device.makeComputePipelineState(function:library.makeFunction(name:names[order]!)!)
  precondition(pipelines[order]!.threadExecutionWidth==32 && pipelines[order]!.maxTotalThreadsPerThreadgroup>=64)
  gpu[order]=[];wall[order]=[]
 }
 func run(_ order:String,_ count:Int)->(Double,Double) {
  let start=Date(),command=queue.makeCommandBuffer()!,encoder=command.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(pipelines[order]!)
  for (index,buffer) in (inputs+outputs).enumerated(){encoder.setBuffer(buffer,offset:0,index:index)}
  let grid=grids[order]!
  for _ in 0..<count {encoder.dispatchThreads(MTLSize(width:grid[0],height:grid[1],depth:grid[2]),threadsPerThreadgroup:MTLSize(width:32,height:2,depth:1))}
  encoder.endEncoding();command.commit();command.waitUntilCompleted()
  precondition(command.status == .completed && command.gpuEndTime>command.gpuStartTime)
  let matched=zip(gold,outputs).allSatisfy{data,buffer in data.withUnsafeBytes{memcmp($0.baseAddress!,buffer.contents(),data.count)==0}}
  equal[order]=(equal[order] ?? true) && matched;precondition(matched,"native output differs from MLX gold")
  return ((command.gpuEndTime-command.gpuStartTime)/Double(count),Date().timeIntervalSince(start)/Double(count))
 }
 for order in orders {_=run(order,1)}
 for round in 0..<rounds {
  let shift=round%orders.count
  for order in Array(orders[shift...])+Array(orders[..<shift]) {
   let sample=run(order,repeats);gpu[order]!.append(sample.0);wall[order]!.append(sample.1)
  }
 }
 let unchanged=zip(inputData,inputs).allSatisfy{data,buffer in data.withUnsafeBytes{memcmp($0.baseAddress!,buffer.contents(),data.count)==0}}
 precondition(unchanged)
 let result:[String:Any]=["key":record["key"]!,"hv":record["hv"]!,"stored_batch":record["stored_batch"]!,"phase":record["phase"]!,"interval":record["interval"]!,"write":record["write"]!,"gpu_samples_s":gpu,"wall_samples_s":wall,"gpu_median_s":gpu.mapValues{$0.sorted()[$0.count/2]},"wall_median_s":wall.mapValues{$0.sorted()[$0.count/2]},"six_outputs_bitwise_equal_to_mlx":equal,"inputs_unchanged":unchanged]
 results.append(result);print(record["key"]!,result["gpu_median_s"]!)
}
let out:[String:Any]=["cases":results,"device":device.name,"rounds":rounds,"repeated_dispatches_per_command":repeats,"fast_math":false,"phone_executed":false,"scope":"One layer's inputs repeatedly read; cache-hot isolated native Metal command-buffer GPU timing per dispatch; seven rotated rounds; all six outputs compared bytewise against MLX gold each command","limits":"No full-model weights, layer sweep, DRAM counter, cold-cache or iPhone claim; native sequential repeated output writes include command scheduling/hazard cost"]
try JSONSerialization.data(withJSONObject:out,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:args[2]))
