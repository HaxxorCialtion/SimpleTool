import Foundation
import MLX
import SimpleToolSwift

// Run synchronously on one exclusive inference lane, never on an application's UI thread.
let args=CommandLine.arguments
guard args.count==3 else {fatalError("Usage: InfraSmoke MODEL_DIRECTORY RECEIPT.json")}
let directory=URL(fileURLWithPath:args[1]),output=URL(fileURLWithPath:args[2])
if let path=ProcessInfo.processInfo.environment["SL_EIGHT_HEAD_FIXTURE"] {
    let fixture=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:path))) as! [String:Any]
    let precision=InferencePrecision(rawValue:ProcessInfo.processInfo.environment["SL_LLM_PRECISION"] ?? "w4a32")!
    var rows=[[String:Any]](),tokenizerPassed=true
    let zeroTool = #"{"type":"function","function":{"name":"battery","parameters":{"type":"object","properties":{}}}}"#
    guard try EightHeadPlan(toolsJSON:[zeroTool]).heads == ["content","function","arg1"],
          try EightHeadPlan(toolsJSON:[zeroTool],includeContent:false).heads == ["function","arg1"] else {fatalError("Eight-head zero-argument schedule mismatch")}
    try autoreleasepool {
        let session=try EightHeadSession(directory:directory,precision:precision)
        for test in fixture["tokenizer_tests"] as? [[String:Any]] ?? [] {
            let encoded = try session.tokenizer.encode(test["text"] as! String)
            tokenizerPassed = tokenizerPassed && encoded == test["ids"] as! [Int]
        }
        for test in fixture["cases"] as! [[String:Any]] {
            let prompt=test["prompt"] as! String,heads=test["heads"] as! [String],limit=test["max_tokens"] as! Int
            session.resetPrefix()
            let parallel=try session.infer(rawPrompt:prompt,heads:heads,maxTokens:limit)
            let serial=try session.infer(rawPrompt:prompt,heads:heads,maxTokens:limit,sharedPrefix:false)
            let parity=zip(parallel.heads,serial.heads).allSatisfy{$0.name == $1.name && $0.tokens == $1.tokens}
            let prefixCachePassed=parallel.cacheHitTokens == 0 && serial.cacheHitTokens == parallel.promptTokens
            var rawPromptRejected=false
            do {_ = try session.infer(rawPrompt:prompt+"<think>",heads:heads,maxTokens:limit)}
            catch EngineError.invalidPacket {rawPromptRejected=true}
            var updates=0,cancelled=false
            do {_ = try session.infer(rawPrompt:prompt,heads:heads,maxTokens:limit,cancelled:{updates>=1},onUpdate:{_ in updates+=1})}
            catch is CancellationError {cancelled=true}
            let replay=try session.infer(rawPrompt:prompt,heads:heads,maxTokens:limit)
            let replayEqual=zip(parallel.heads,replay.heads).allSatisfy{$0.tokens == $1.tokens}
            rows.append(["case":test["name"]!,"parallel":try JSONSerialization.jsonObject(with:JSONEncoder().encode(parallel)),
                         "serial":try JSONSerialization.jsonObject(with:JSONEncoder().encode(serial)),
                         "serial_batch_token_equal":parity,"cancelled":cancelled,"replay_token_equal":replayEqual,
                         "prefix_cache_passed":prefixCachePassed,"extra_template_rejected":rawPromptRejected,
                         "load_seconds":session.loadSeconds])
            print("EIGHT_HEAD",test["name"]!,"parity",parity,"cancel",cancelled,"replay",replayEqual);fflush(stdout)
            try JSONSerialization.data(withJSONObject:["rows":rows,"tokenizer_passed":tokenizerPassed,"status":"running"],options:[.prettyPrinted,.sortedKeys]).write(to:output)
        }
    }
    releaseChatRuntimeCaches()
    let passed=tokenizerPassed && rows.allSatisfy{($0["serial_batch_token_equal"] as? Bool)==true && ($0["cancelled"] as? Bool)==true && ($0["replay_token_equal"] as? Bool)==true && ($0["prefix_cache_passed"] as? Bool)==true && ($0["extra_template_rejected"] as? Bool)==true} && Memory.activeMemory<64*1024*1024
    try JSONSerialization.data(withJSONObject:["rows":rows,"tokenizer_passed":tokenizerPassed,"all_passed":passed,
        "precision":precision.rawValue,"released_active_bytes":Memory.activeMemory,"status":"completed",
        "scope":"Mac Metal INT4 authored text fixtures; same-weights serial/batch check; no FP16 numerical parity or iPhone performance claim"],options:[.prettyPrinted,.sortedKeys]).write(to:output)
    exit(passed ? 0:2)
}
if ProcessInfo.processInfo.environment["SL_LOGITS_TEST"] == "1" {
    let checks=try validateConversationLogits()
    try JSONSerialization.data(withJSONObject:checks,options:[.prettyPrinted,.sortedKeys]).write(to:output)
    print(checks);exit(0)
}
if let fixturePath=ProcessInfo.processInfo.environment["SL_REPLAY_CONFIG"] {
    let fixture=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:fixturePath))) as! [String:Any]
    let system=fixture["system"] as! String,payload=fixture["payload"] as! String,query=fixture["query"] as! String
    let variants=fixture["variants"] as! [[String:String]],limit=fixture["maxTokens"] as! Int
    var rows=[[String:Any]]()
    for variant in variants {
        try autoreleasepool {
            let precision=InferencePrecision(rawValue:variant["precision"]!)!
            let session=try ChatSession(directory:directory,system:system,initialLayerEvaluationStride:4,allocatorCacheBytes:16*1024*1024,precision:precision,numerics:NumericalPolicy(rawValue:variant["numerics"] ?? "standard")!)
            session.configureDecodeTuning(projections:variant["projections"]!)
                session.decodingPolicy=ChatDecodingPolicy(rawValue:variant["decoding"] ?? "greedy")!
                session.temperature=Float(variant["temperature"] ?? "0.7")!
            if variant["warm_decode"] == "1" {try session.warmDecode()}
            if variant["warm"] == "1" {
                for warm in fixture["warmups"] as? [[String:Any]] ?? [] {
                    session.restore([])
                    _ = try session.reply(warm["query"] as! String,maxTokens:limit,preparedPayload:warm["payload"] as! String)
                }
            }
            session.restore([])
            let turn=try session.reply(query,maxTokens:limit,headNames:variant["batch"] == "1" ? ["speech"]:ChatSession.heads,preparedPayload:payload)
            rows.append(["variant":variant,"turn":try JSONSerialization.jsonObject(with:JSONEncoder().encode(turn))])
        }
        releaseChatRuntimeCaches()
        var record:[String:Any]=["scope":"Mac replay of supplied system/payload; INT4 weights; raw greedy, no repetition guard; not device speed acceptance","rows":rows,"released_active_bytes":Memory.activeMemory]
        record["fixture_path"]=fixturePath
        try JSONSerialization.data(withJSONObject:record,options:[.prettyPrinted,.sortedKeys]).write(to:output)
        print("replayed",variant,"released",Memory.activeMemory)
    }
    exit(0)
}
if ProcessInfo.processInfo.environment["SL_REAL_PARITY_SWEEP"] == "1" {
    let s="你叫 simplelove，是温柔机灵的小精灵。用中文自然聊天。expression 可用微笑、思考、惊讶。系统指定 speech、expression、inner 或 command 通道后，直接输出正文。"
    let q="今天有点累，陪我聊聊吧。给我一些温柔又具体的建议。"
    let p=try SimpleToolProtocol.json(["date":"2026-11-07","history":[],"query":q,"memory_records":[]] as [String:Any])
    var samples=[[String:Any]]()
    try autoreleasepool {
        let session=try ChatSession(directory:directory,system:s,precision:.a16kv8)
        for candidate in ["legacy","reuse","reuse-qmv","reuse-consistent"] {
            session.configureDecodeTuning(projections:candidate)
            for heads in [ChatSession.heads,["speech"]] {
                session.restore([])
                let turn=try session.reply(q,maxTokens:64,headNames:heads,preparedPayload:p)
                samples.append(["candidate":candidate,"batch":heads.count,"turn":try JSONSerialization.jsonObject(with:JSONEncoder().encode(turn))])
            }
        }
    }
    releaseChatRuntimeCaches()
    try JSONSerialization.data(withJSONObject:["samples":samples,"released_active_bytes":Memory.activeMemory,"scope":"Mac authored real-chat diagnostic; not iPhone speed acceptance"],options:[.prettyPrinted,.sortedKeys]).write(to:output)
    exit(0)
}
var receipt=[String:Any]()
let system="你叫 simplelove，是温柔机灵的小精灵。用中文自然聊天。expression 可用微笑、思考、惊讶。系统指定 speech、expression、inner 或 command 通道后，直接输出正文。"
let payload=try SimpleToolProtocol.json(["date":"2026-11-05","history":[],"query":"你好，介绍一下你自己。"] as [String:Any])
let start=ProcessInfo.processInfo.systemUptime
var session:ChatSession?
try autoreleasepool {
    let stride = Int(ProcessInfo.processInfo.environment["SL_DECODE_EVAL_STRIDE"] ?? "0") ?? 0
    session=try ChatSession(directory:directory,system:system,promptProfile:"infra-smoke-20261105",initialLayerEvaluationStride:stride,allocatorCacheBytes:(stride>0 ? 16 : 128)*1024*1024,precision:InferencePrecision(rawValue:ProcessInfo.processInfo.environment["SL_LLM_PRECISION"] ?? "w4a32") ?? .fp32)
    if ProcessInfo.processInfo.environment["SL_IPHONE_POLICY"] == "1" {session!.useIPhoneDecodeDefaults()}
    if let tuning=ProcessInfo.processInfo.environment["SL_PROJECTION_POLICY"] {session!.configureDecodeTuning(projections:tuning)}
    try session!.warmDecode()
    let batch=try session!.reply("smoke",maxTokens:12,preparedPayload:payload)
    var serial=[String:[Int]](),equal=true
    for head in ChatSession.heads {
        session!.restore([])
        let t=try session!.reply("smoke",maxTokens:12,headNames:[head],preparedPayload:payload)
        serial[head]=t.tokenIDs?[head] ?? []
        equal = equal && serial[head] == batch.tokenIDs?[head]
    }
    session!.restore([])
    var updates=0,cancelled=false
    do {_ = try session!.reply("smoke",maxTokens:12,preparedPayload:payload,cancelled:{updates>=1}) {_ in updates+=1}}
    catch is CancellationError {cancelled=true}
    let cancelledHistoryEmpty=session!.turns.isEmpty
    let memory=try session!.memoryReply(system:"简要概括用户提供的记录。",input:"用户喜欢薄荷，今天整理了书桌。",maxTokens:8)
    session!.releaseMemoryCache()
    receipt=["model":directory.lastPathComponent,"authored_smoke_not_training_gold":true,"batch":try JSONSerialization.jsonObject(with:JSONEncoder().encode(batch)),"serial_token_ids":serial,"batch_serial_token_equal":equal,"cancelled_between_tokens":cancelled,"cancelled_history_empty":cancelledHistoryEmpty,"memory_single_head_smoke":try JSONSerialization.jsonObject(with:JSONEncoder().encode(memory)),"load_metrics":session!.loadMetrics,"warm_active_bytes":Memory.activeMemory]
    receipt["layer_evaluation_stride"] = session!.layerEvaluationStride
    receipt["precision"] = session!.precision.rawValue
    receipt["iphone_kernel_policy"] = ProcessInfo.processInfo.environment["SL_IPHONE_POLICY"] == "1"
    receipt["projection_override"] = ProcessInfo.processInfo.environment["SL_PROJECTION_POLICY"] ?? "none"
}
autoreleasepool {session=nil;releaseChatRuntimeCaches()}
receipt["released_active_bytes"]=Memory.activeMemory
receipt["elapsed_seconds"]=ProcessInfo.processInfo.systemUptime-start
receipt["all_passed"]=(receipt["batch_serial_token_equal"] as? Bool == true) && (receipt["cancelled_between_tokens"] as? Bool == true) && (receipt["cancelled_history_empty"] as? Bool == true) && Memory.activeMemory<64*1024*1024
try JSONSerialization.data(withJSONObject:receipt,options:[.prettyPrinted,.sortedKeys]).write(to:output)
print(directory.lastPathComponent,"all_passed",receipt["all_passed"]!,"released_active_bytes",Memory.activeMemory)
if receipt["all_passed"] as? Bool != true {exit(2)}
