import Foundation
import SimpleToolSwift

struct Options {
    let model: URL
    let fixture: URL
    let output: URL
    let size: String
    let limit: Int?

    init(_ args: [String]) throws {
        func value(_ name: String) throws -> String {
            guard let i = args.firstIndex(of: name), i + 1 < args.count else {
                throw EngineError.invalidPacket("Missing \(name) <value>")
            }
            return args[i + 1]
        }
        model = URL(fileURLWithPath: try value("--model"))
        fixture = URL(fileURLWithPath: try value("--fixture"))
        output = URL(fileURLWithPath: try value("--output"))
        size = try value("--size")
        if let i = args.firstIndex(of: "--limit"), i + 1 < args.count { limit = Int(args[i + 1]) }
        else { limit = nil }
    }
}

func persist(_ report: [String: Any], to url: URL) throws {
    let data = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys, .fragmentsAllowed])
    try data.write(to: url, options: .atomic)
}

do {
    let options = try Options(Array(CommandLine.arguments.dropFirst()))
    let fixtureData = try Data(contentsOf: options.fixture)
    let raw: [[String: Any]]
    if let array = try? JSONSerialization.jsonObject(with: fixtureData) as? [[String: Any]] {
        raw = array
    } else {
        raw = try fixtureData.split(separator: 10).filter { !$0.isEmpty }.map {
            try JSONSerialization.jsonObject(with: Data($0)) as! [String: Any]
        }
    }
    let cases = Array(raw.prefix(options.limit ?? raw.count))
    guard !cases.isEmpty else { throw EngineError.invalidPacket("No evaluation cases") }
    try FileManager.default.createDirectory(at: options.output.deletingLastPathComponent(), withIntermediateDirectories: true)
    var report: [String: Any] = [
        "dataset": "google/mobile-actions",
        "model": options.size,
        "runtime": "SimpleToolSwift EightHeadSession / MLX Metal",
        "precision": "w4a16-kv8",
        "phase": "loading",
        "status": "running",
        "started_at": ISO8601DateFormatter().string(from: Date()),
        "rows": [] as [[String: Any]]
    ]
    try persist(report, to: options.output)
    try autoreleasepool {
        let session = try EightHeadSession(directory: options.model, precision: .a16kv8)
        report["load_seconds"] = session.loadSeconds
        report["weight_bytes"] = session.weightBytes
        report["phase"] = "infer"
        try persist(report, to: options.output)
        var rows = [[String: Any]]()
        for (position, item) in cases.enumerated() {
            guard let prompt = item["prompt"] as? String,
                  let heads = item["heads"] as? [String],
                  let maxTokens = item["max_tokens"] as? Int,
                  let id = item["id"] as? Int else {
                throw EngineError.invalidPacket("Malformed fixture row \(position)")
            }
            session.resetPrefix()
            let result = try session.infer(rawPrompt: prompt, heads: heads, maxTokens: maxTokens)
            let encoded = try JSONEncoder().encode(result)
            let resultJSON = try JSONSerialization.jsonObject(with: encoded) as! [String: Any]
            rows.append([
                "id": id,
                "reference_call_count": item["reference_call_count"] as? Int ?? 0,
                "reference_name": item["reference_name"] as? String ?? "",
                "reference_arguments": item["reference_arguments"] ?? [:],
                "property_order": item["property_order"] ?? [],
                "required": item["required"] ?? [],
                "schema": item["schema"] ?? [:],
                "prompt_tokens": result.promptTokens,
                "prefill_seconds": result.prefillSeconds,
                "decode_seconds": result.decodeSeconds,
                "generated_token_count": result.heads.reduce(0) { $0 + $1.tokens.count },
                "active_batch_sizes": result.activeBatchSizes,
                "peak_mlx_bytes": result.peakMLXBytes,
                "heads": resultJSON["heads"] ?? []
            ])
            report["rows"] = rows
            report["completed_cases"] = rows.count
            report["current_case_id"] = id
            try persist(report, to: options.output)
            if (position + 1).isMultiple(of: 25) || position + 1 == cases.count {
                FileHandle.standardOutput.write(Data("\(options.size) \(position + 1)/\(cases.count)\n".utf8))
            }
        }
    }
    releaseChatRuntimeCaches()
    report["status"] = "completed"
    report["phase"] = "finished"
    report["finished_at"] = ISO8601DateFormatter().string(from: Date())
    report["scope"] = "Raw ChatML; first reference call for multi-call requests; inference candidate only; no tool execution"
    try persist(report, to: options.output)
} catch {
    fputs("MobileActionsBench failed: \(error)\n", stderr)
    exit(1)
}
