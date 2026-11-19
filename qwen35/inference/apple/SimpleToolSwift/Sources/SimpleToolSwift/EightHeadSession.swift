import Foundation
import MLX

public struct EightHeadPlan: Sendable {
    public let heads: [String]
    /// Count all declared properties (including optional ones) of all candidate
    /// tools. Never infer positional argument order by sorting JSON keys.
    public init(toolsJSON: [String], includeContent: Bool = true) throws {
        guard !toolsJSON.isEmpty else { throw EngineError.invalidPacket("Empty tool list") }
        var names = Set<String>(), maximum = 1
        for text in toolsJSON {
            guard let data = text.data(using:.utf8),
                  let tool = try JSONSerialization.jsonObject(with:data) as? [String:Any],
                  tool["type"] as? String == "function",
                  let function = tool["function"] as? [String:Any],
                  let name = function["name"] as? String, !name.isEmpty,
                  let parameters = function["parameters"] as? [String:Any],
                  parameters["type"] as? String == "object",
                  let properties = parameters["properties"] as? [String:Any],
                  properties.count <= 6, names.insert(name).inserted else {
                throw EngineError.invalidPacket("Expected unique function tools with 0...6 properties")
            }
            maximum = max(maximum,properties.count)
        }
        heads = (includeContent ? ["content"] : []) + ["function"] + (1...maximum).map { "arg\($0)" }
    }
}

public struct EightHeadOutput: Codable, Sendable {
    public let name: String
    /// Includes the terminating token; body excludes the matching close marker/EOS.
    public let tokens: [Int]
    public let body: String
    public let stop: String
    public let finishedSeconds: Double
}

public struct EightHeadResult: Codable, Sendable {
    public let heads: [EightHeadOutput]
    public let promptTokens: Int
    public let cacheHitTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public let stepSeconds: [Double]
    public let activeBatchSizes: [Int]
    public let sharedPrefix: Bool
    public let precision: String
    public let peakMLXBytes: Int
}

/// Text-only SimpleTool-Qwen3.5 eight-branch ABI, deliberately separate from
/// SimpleLove's four-channel ChatSession. Call on one exclusive inference lane.
/// Caller supplies the COMPLETE raw ChatML prompt; no template, think block,
/// persona, tool schema, history or tool execution is implicitly added here.
public final class EightHeadSession {
    public static let heads = ["content", "function"] + (1...6).map { "arg\($0)" }
    private let model: Int4Model
    public let tokenizer: ChatTokenizer
    private let opening: [String: Int], closing: [String: Int]
    private let eos: Int
    private let contextLimit: Int
    private let evaluationStride: Int
    private var prefixIDs = [Int](), prefixCache = [LayerState]()
    public let loadSeconds: Double
    public var weightBytes: Int { model.weightBytes }

    public init(directory: URL, precision: InferencePrecision = .fp32,
                contextLimit: Int = 8192, evaluationStride: Int = 4,
                reuseEightRows: Bool = true) throws {
        let start = ProcessInfo.processInfo.systemUptime
        guard contextLimit > 128, [0,4,8,16].contains(evaluationStride) else {
            throw EngineError.invalidConfiguration("Invalid eight-head memory policy")
        }
        self.contextLimit = contextLimit; self.evaluationStride = evaluationStride
        Memory.cacheLimit = 16 * 1024 * 1024
        tokenizer = try ChatTokenizer(directory: directory)
        let raw = try JSONSerialization.jsonObject(with: Data(contentsOf: directory.appendingPathComponent("simpletool_config.json"))) as? [String: Any]
        guard raw?["protocol"] as? String == "SimpleTool-Qwen3.5-8Head",
              raw?["prompt_family"] as? String == "raw-chatml",
              raw?["heads"] as? [String] == Self.heads,
              let opens = raw?["head_token_ids"] as? [String: Int],
              let closes = raw?["closing_token_ids"] as? [String: Int] else {
            throw EngineError.invalidConfiguration("Expected eight-head raw ChatML protocol")
        }
        for name in Self.heads {
            guard try tokenizer.encode("<\(name)>") == [opens[name] ?? -1],
                  try tokenizer.encode("</\(name)>") == [closes[name] ?? -1] else {
                throw EngineError.invalidConfiguration("Eight-head tokenizer mismatch: \(name)")
            }
        }
        eos = try tokenizer.tokenID("<|im_end|>")
        guard raw?["eos_token_id"] as? Int == eos,
              try tokenizer.tokenID("<|null|>") == raw?["null_token_id"] as? Int else {
            throw EngineError.invalidConfiguration("Eight-head EOS/null mismatch")
        }
        opening = opens; closing = closes
        model = try Int4Model(directory: directory, projectionPolicy: .both, precision: precision)
        model.prefillEvaluationStride = evaluationStride
        // Same per-row arithmetic at B1..B8, including after early termination.
        model.decodeProjectionMode = "reuse-consistent"
        model.decodeReuseBatchLimit = reuseEightRows ? 8 : 4
        loadSeconds = ProcessInfo.processInfo.systemUptime - start
    }

    public func resetPrefix() { prefixIDs = []; prefixCache = []; Memory.clearCache() }

    /// Branch bodies are independent; null is data, not a reason to skip closure.
    /// Serial mode is a same-weights reference. Neither mode executes tools.
    public func infer(rawPrompt: String, heads: [String] = EightHeadSession.heads,
                      maxTokens: Int = 128, sharedPrefix: Bool = true,
                      cancelled: () -> Bool = { false },
                      onUpdate: ([String: String]) -> Void = { _ in }) throws -> EightHeadResult {
        guard rawPrompt.hasSuffix("<|im_start|>assistant\n"),
              (1...4096).contains(maxTokens), !heads.isEmpty,
              Set(heads).count == heads.count, heads.allSatisfy(Self.heads.contains) else {
            throw EngineError.invalidPacket("Supply raw ChatML ending in assistant newline and unique eight-head names")
        }
        let ids = try tokenizer.encode(rawPrompt)
        guard ids.count + maxTokens + 1 <= contextLimit else {
            throw EngineError.invalidPacket("Eight-head context budget exceeded")
        }
        let clock = { ProcessInfo.processInfo.systemUptime }
        let prefillStart = clock()
        let hit = ids.starts(with: prefixIDs) ? prefixIDs.count : 0
        let seed = hit > 0 ? prefixCache.map { $0.copy() } : model.newCache()
        for i in stride(from: hit, to: ids.count, by: 64) {
            if cancelled() { throw CancellationError() }
            _ = model.prefill(Array(ids[i..<min(i+64, ids.count)]), cache: seed, chunkSize: 64)
        }
        // Saved prefix is immutable; every decode forks its own GDN/KV suffix.
        prefixIDs = ids; prefixCache = seed.map { $0.copy() }
        let prefillSeconds = clock() - prefillStart
        let begin = clock()
        var raw = Array(repeating: [Int](), count: heads.count)
        var stops = Array(repeating: "max_tokens", count: heads.count)
        var finished = Array(repeating: 0.0, count: heads.count)
        var times = [Double](), batches = [Int]()
        func body(_ index: Int) -> String {
            var tokens = raw[index]
            if let last = tokens.last, last == closing[heads[index]] || last == eos { tokens.removeLast() }
            return (try? model.decoder.decode(tokens)) ?? ""
        }
        let groups = sharedPrefix ? [Array(heads.indices)] : heads.indices.map { [$0] }
        for group in groups {
            let cache = seed.map { $0.fork(batch: group.count, capacity: maxTokens + 1) }
            var active = group, current = group.map { opening[heads[$0]]! }
            for step in 0..<maxTokens {
                if cancelled() { throw CancellationError() }
                let stepStart = clock()
                let hidden = model.hidden(MLXArray(current, [current.count,1]), cache: cache, evaluateEvery: evaluationStride)
                let logits = model.linear(hidden[0...,-1,0...], "embed_tokens")
                let prediction = argMax(logits, axis: -1)
                eval([prediction] + cache.flatMap { $0.arrays })
                let chosen = prediction.asArray(Int32.self).map(Int.init)
                times.append(clock() - stepStart); batches.append(active.count)
                var keep = [Int]()
                for (row, index) in active.enumerated() {
                    let token = chosen[row]; raw[index].append(token)
                    finished[index] = clock() - begin
                    if token == closing[heads[index]] { stops[index] = "closing_marker" }
                    else if token == eos { stops[index] = "eos_before_closing_marker" }
                    else { keep.append(row) }
                }
                onUpdate(Dictionary(uniqueKeysWithValues: heads.indices.compactMap { i in
                    (heads[i] == "content" || stops[i] != "max_tokens") ? (heads[i], body(i)) : nil
                }))
                if keep.isEmpty || step+1 == maxTokens { break }
                if keep.count != active.count {
                    let rows = MLXArray(keep)
                    let blockSize = evaluationStride > 0 ? evaluationStride : cache.count
                    for i in stride(from: 0, to: cache.count, by: blockSize) {
                        let block = Array(cache[i..<min(i+blockSize,cache.count)])
                        block.forEach { $0.filter(rows) }; model.evaluate(block)
                    }
                }
                current = keep.map { chosen[$0] }; active = keep.map { active[$0] }
            }
        }
        return EightHeadResult(heads: heads.indices.map {
            EightHeadOutput(name: heads[$0], tokens: raw[$0], body: body($0), stop: stops[$0], finishedSeconds: finished[$0])
        }, promptTokens: ids.count, cacheHitTokens: hit, prefillSeconds: prefillSeconds,
            decodeSeconds: clock() - begin, stepSeconds: times, activeBatchSizes: batches,
            sharedPrefix: sharedPrefix, precision: model.precision.rawValue, peakMLXBytes: Memory.peakMemory)
    }
}
