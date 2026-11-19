// Actual all-layer packed int4 projections. GPU duration is separate from host wall time.
// Buffers belong to this private process; no borrowed MLX inputs or model state.
import Foundation
import Metal

enum BenchError: Error { case invalid(String) }
let args = CommandLine.arguments
guard args.count == 4 else { throw BenchError.invalid("fixture-directory metallib output-json") }
let root = URL(fileURLWithPath: args[1])
let manifest = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent("manifest.json"))) as! [String: Any]
guard manifest["weight_bits"] as? Int == 4, manifest["compute"] as? String == "float32",
      let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
else { throw BenchError.invalid("int4 FP32 or Metal unavailable") }
let library = try device.makeLibrary(URL: URL(fileURLWithPath: args[2]))
let checkpoint = try Data(contentsOf: URL(fileURLWithPath: manifest["checkpoint"] as! String), options: .mappedIfSafe)

func copyBuffer(_ data: Data) throws -> MTLBuffer {
    try data.withUnsafeBytes { raw in
        guard let buffer = device.makeBuffer(bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
        else { throw BenchError.invalid("buffer allocation") }
        return buffer
    }
}
struct Matrix { let path: String; let w, s, b, output: MTLBuffer }
struct Tile { let rows, threads: Int }
let tiles = [1, 2, 4, 8].flatMap { r in [128, 256].map { Tile(rows: r, threads: $0) } }
let modes = ["baseline", "j8", "scalar"]
var records = [[String: Any]]()
var totalComparisons = 0
var allocationBytes = 0
var familyAllocationBytes = [String: Int]()
var peakFamilyBytes = 0
let checkpointBytes = checkpoint.count
let clock = ContinuousClock()
func seconds(_ duration: Duration) -> Double {
    Double(duration.components.seconds) + Double(duration.components.attoseconds) / 1e18
}
func writeReport() throws {
    let result: [String: Any] = ["device": device.name, "size": manifest["size"]!,
        "weight_bits": 4, "compute": "float32", "rounds": 7,
        "checkpoint_sha256": manifest["checkpoint_sha256"]!, "records": records,
        "actual_weight_bitwise_comparisons": totalComparisons,
        "family_buffer_bytes": familyAllocationBytes, "peak_family_buffer_bytes": peakFamilyBytes,
        "mapped_checkpoint_file_bytes": checkpointBytes,
        "phone_executed": false, "full_model": false,
        "scope": "All actual gate/down matrices plus tied vocabulary; B1..4, 24 candidates each. One command buffer per family sweep; seven rotating rounds. Vocabulary repeats same matrix four times.",
        "limitations": "GPU command duration includes encoder scheduling, not isolated hardware counters; useful packed/metadata bytes are not measured DRAM traffic. Native candidates, not installed Python shader or full Swift model. Inputs are seeded finite FP32, not captured decode activations. B1 uses row-reuse gold. External load uncontrolled. Buffer accounting is not physical footprint or deployment RAM."]
    try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: args[3]))
}
for familyObject in manifest["families"] as! [[String: Any]] {
    try autoreleasepool {
        let family = familyObject["name"] as! String, n = familyObject["n"] as! Int, k = familyObject["k"] as! Int
        let descriptors = familyObject["matrices"] as! [[String: Any]]
        var matrices = [Matrix]()
        allocationBytes = 0
        for entry in descriptors {
            let buffers = entry["buffers"] as! [String: [String: Int]]
            func tensor(_ name: String) throws -> MTLBuffer {
                let info = buffers[name]!, offset = info["offset"]!, bytes = info["bytes"]!
                guard offset >= 0, offset + bytes <= checkpoint.count else { throw BenchError.invalid("tensor file range") }
                let buffer = try checkpoint.withUnsafeBytes { raw -> MTLBuffer in
                    guard let b = device.makeBuffer(bytes: raw.baseAddress!.advanced(by: offset), length: bytes, options: .storageModeShared)
                    else { throw BenchError.invalid("weight allocation") }
                    return b
                }
                allocationBytes += bytes
                return buffer
            }
            guard let output = device.makeBuffer(length: 4*n*4, options: .storageModeShared) else { throw BenchError.invalid("output allocation") }
            allocationBytes += output.length
            matrices.append(Matrix(path: entry["path"] as! String, w: try tensor("w"), s: try tensor("s"), b: try tensor("b"), output: output))
        }
        let repeats = family == "vocabulary" ? 4 : 1
        let usefulBytes = matrices.reduce(0) { $0 + $1.w.length + $1.s.length + $1.b.length } * repeats
        familyAllocationBytes[family] = allocationBytes
        peakFamilyBytes = max(peakFamilyBytes, allocationBytes)
        for batchObject in familyObject["batches"] as! [[String: Any]] {
            let m = batchObject["batch"] as! Int
            let inputData = try Data(contentsOf: root.appendingPathComponent(batchObject["x"] as! String))
            let input = try copyBuffer(inputData)
            let expected = try Data(contentsOf: root.appendingPathComponent(batchObject["expected"] as! String))
            guard input.length == m*k*4, expected.count == matrices.count*m*n*4 else { throw BenchError.invalid("fixture dimensions") }
            struct Candidate { let mode: String; let tile: Tile; let pipeline: MTLComputePipelineState; let compile: Double }
            var candidates = [Candidate]()
            for mode in modes { for tile in tiles {
                let values = MTLFunctionConstantValues()
                for (index, v) in [k, n].enumerated() {
                    var value = UInt32(v); values.setConstantValue(&value, type: .uint, index: index)
                }
                let name = "weighted_\(mode)_m\(m)_r\(tile.rows)"
                let start = clock.now
                let function = try library.makeFunction(name: name, constantValues: values)
                let pipeline = try device.makeComputePipelineState(function: function)
                let compile = seconds(start.duration(to: clock.now))
                guard pipeline.threadExecutionWidth == 32, pipeline.maxTotalThreadsPerThreadgroup >= tile.threads else { throw BenchError.invalid("SIMD/threadgroup mismatch") }
                candidates.append(Candidate(mode: mode, tile: tile, pipeline: pipeline, compile: compile))
            } }
            func run(_ c: Candidate) throws -> (gpu: Double, wall: Double) {
                let start = clock.now
                guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder() else { throw BenchError.invalid("encoder allocation") }
                encoder.setComputePipelineState(c.pipeline)
                encoder.setBuffer(input, offset: 0, index: 0)
                let groups = (n + c.tile.rows * (c.tile.threads/32)-1)/(c.tile.rows*(c.tile.threads/32))
                for _ in 0..<repeats { for matrix in matrices {
                    encoder.setBuffer(matrix.w, offset: 0, index: 1)
                    encoder.setBuffer(matrix.s, offset: 0, index: 2)
                    encoder.setBuffer(matrix.b, offset: 0, index: 3)
                    encoder.setBuffer(matrix.output, offset: 0, index: 4)
                    encoder.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1), threadsPerThreadgroup: MTLSize(width: c.tile.threads, height: 1, depth: 1))
                } }
                encoder.endEncoding(); command.commit(); command.waitUntilCompleted()
                guard command.status == .completed, command.gpuEndTime > command.gpuStartTime else { throw BenchError.invalid("GPU command failure \(String(describing: command.error))") }
                return (command.gpuEndTime-command.gpuStartTime, seconds(start.duration(to: clock.now)))
            }
            func validate() throws {
                let equal = expected.withUnsafeBytes { raw in
                    matrices.indices.allSatisfy { i in
                        memcmp(matrices[i].output.contents(), raw.baseAddress!.advanced(by: i*m*n*4), m*n*4) == 0
                    }
                }
                guard equal else { throw BenchError.invalid("MLX bitwise mismatch \(family) B\(m)") }
                totalComparisons += matrices.count
            }
            var gpu = Array(repeating: [Double](), count: candidates.count)
            var wall = gpu
            var first = [(gpu: Double, wall: Double)]()
            for candidate in candidates {
                first.append(try run(candidate)); try validate()
            }
            for round in 0..<7 {
                let shift = (round * 5) % candidates.count
                for i in (0..<candidates.count).map({ ($0+shift)%candidates.count }) {
                    let sample = try run(candidates[i]); gpu[i].append(sample.gpu); wall[i].append(sample.wall)
                    try validate()
                }
            }
            let outputs = candidates.indices.map { i -> [String: Any] in
                let c = candidates[i], medianGPU = gpu[i].sorted()[3], medianWall = wall[i].sorted()[3]
                return ["mode": c.mode, "rows": c.tile.rows, "threads": c.tile.threads,
                    "gpu_samples_s": gpu[i], "wall_samples_s": wall[i], "gpu_median_s": medianGPU,
                    "wall_median_s": medianWall, "mean_projection_gpu_s": medianGPU/Double(matrices.count*repeats),
                    "mean_projection_wall_s": medianWall/Double(matrices.count*repeats),
                    "useful_weight_metadata_GBps": Double(usefulBytes)/medianGPU/1e9,
                    "pipeline_specialize_s": c.compile, "first_gpu_s": first[i].gpu, "first_wall_s": first[i].wall,
                    "thread_execution_width": c.pipeline.threadExecutionWidth,
                    "max_total_threads": c.pipeline.maxTotalThreadsPerThreadgroup,
                    "static_threadgroup_memory_bytes": c.pipeline.staticThreadgroupMemoryLength,
                    "bitwise_equal_MLX_gold_all_rounds": true]
            }
            records.append(["family": family, "batch": m, "n": n, "k": k, "layers": matrices.count,
                "layer_paths": matrices.map { $0.path }, "repeats": repeats, "useful_weight_metadata_bytes": usefulBytes,
                "candidates": outputs])
            try writeReport()
            let fastest = outputs.min { ($0["gpu_median_s"] as! Double) < ($1["gpu_median_s"] as! Double) }!
            print(manifest["size"]!, family, "B\(m)", fastest["mode"]!, fastest["rows"]!, fastest["threads"]!, "GPU ms/projection", (fastest["mean_projection_gpu_s"] as! Double)*1000)
            fflush(stdout)
        }
    }
}
try writeReport()
