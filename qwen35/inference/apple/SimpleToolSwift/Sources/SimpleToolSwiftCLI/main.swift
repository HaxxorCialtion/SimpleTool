import Foundation
import SimpleToolSwift
import MLX
if CommandLine.arguments.count==5 && CommandLine.arguments[1]=="--1013-case" {
    let messages=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[3]))) as! [[String:String]]
    let session=try ChatSession(directory:URL(fileURLWithPath:CommandLine.arguments[2]),system:messages[0]["content"]!,contextLimit:8192,promptProfile:"1013-example-greedy")
    let input=messages[1]["content"]!
    let memory=CommandLine.arguments[3].contains("memory_")
    let turn=try memory ? session.memoryReply(system:messages[0]["content"]!,input:input,maxTokens:2048) : session.reply(input,maxTokens:256,preparedPayload:input)
    try JSONEncoder().encode(turn).write(to:URL(fileURLWithPath:CommandLine.arguments[4]))
    print("Case complete",turn.stops,turn.prefillSeconds,turn.decodeSeconds)
} else if CommandLine.arguments.count==4 && CommandLine.arguments[1]=="--prefill-check" {
    let session=try ChatSession(directory:URL(fileURLWithPath:CommandLine.arguments[2]),system:ChatPromptProfile.companion.text,promptProfile:"companion")
    _ = try session.reply("你好",maxTokens:8)
    let history=session.turns
    var samples=[ChatTurn]()
    for target in [256,512,1024] {
        let input=try session.benchmarkInput(newPrefillTokens:target)
        let turn=try session.benchmarkReply(input,newPrefillTokens:target,maxTokens:24)
        precondition(session.turns.map{$0.id}==history.map{$0.id},"benchmark changed chat history")
        samples.append(turn)
    }
    var cancelled=false
    do {_ = try session.benchmarkReply(session.benchmarkInput(newPrefillTokens:256),newPrefillTokens:256,cancelled:{true})}
    catch is CancellationError {cancelled=true}
    precondition(cancelled && session.turns.map{$0.id}==history.map{$0.id},"benchmark cancellation changed history")
    let next=try session.reply("请介绍一下你自己",maxTokens:8)
    precondition(next.cacheHitTokens>session.warmPrefixTokens,"benchmark lost saved chat cache")
    let data:[String:Any]=["samples":try JSONSerialization.jsonObject(with:JSONEncoder().encode(samples)),"history_restored":true,"cancel_restored":true,"chat_cache_restored":true]
    try JSONSerialization.data(withJSONObject:data,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[3]))
    print("Prefill fixtures: exact 256/512/1024; history, cache and cancellation restored")
} else if CommandLine.arguments.count==4 && CommandLine.arguments[1]=="--prompt-export" {
    let tokenizer=try ChatTokenizer(directory:URL(fileURLWithPath:CommandLine.arguments[2]))
    let profiles=try ChatPromptProfile.allCases.map {profile -> [String:Any] in
        let tokens=try tokenizer.encode(ChatTokenizer.message("system",profile.text)).count
        precondition(tokens<1920,"Prompt exceeds App context budget")
        return ["id":profile.rawValue,"title":profile.title,"system":profile.text,"system_tokens":tokens]
    }
    try JSONSerialization.data(withJSONObject:profiles,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:CommandLine.arguments[3]))
    print(profiles.map{($0["id"]!,$0["system_tokens"]!)})
} else if CommandLine.arguments.count==4 && CommandLine.arguments[1]=="--release-probe" {
    let directory=URL(fileURLWithPath:CommandLine.arguments[2]),mode=CommandLine.arguments[3]
    var model:Int4Model?,session:ChatSession?
    try autoreleasepool {
        if mode=="load" {model=try Int4Model(directory:directory)}
        else {
            session=try ChatSession(directory:directory)
            if mode=="warm" {try session!.warmDecode()}
            if mode=="chat" {_ = try session!.reply("你好",maxTokens:8)}
        }
    }
    let active=Memory.activeMemory
    autoreleasepool {model=nil;session=nil;releaseChatRuntimeCaches()}
    print("RELEASE_PROBE",mode,active,Memory.activeMemory);fflush(stdout)
} else if CommandLine.arguments.count==3 && CommandLine.arguments[1]=="--chat-ownership" {
    let root=URL(fileURLWithPath:CommandLine.arguments[2])
    var receipts=[[String:Any]]()
    for name in ["2b","4b","2b"] {
        let before=Memory.activeMemory
        var session:ChatSession?
        try autoreleasepool {
            session=try ChatSession(directory:root.appendingPathComponent(name+"_int4_compact"))
            try session!.warmDecode()
        }
        let warm=Memory.activeMemory
        var updates=0,cancelled=false
        try autoreleasepool {
          do {
            _ = try session!.reply("请介绍一下你自己",cancelled:{updates>=2}) {_ in updates+=1}
          } catch is CancellationError {cancelled=true}
        }
        precondition(cancelled && session!.turns.isEmpty,"cancelled reply was committed")
        try autoreleasepool {_ = try session!.reply("你好",maxTokens:8)}
        autoreleasepool {
            session!.clear();precondition(session!.turns.isEmpty)
            session=nil;releaseChatRuntimeCaches()
        }
        let released=Memory.activeMemory
        receipts.append(["model":name,"before_bytes":before,"warm_bytes":warm,"released_bytes":released,"cancel_between_steps":cancelled,"updates_before_cancel":updates])
        precondition(Memory.activeMemory<64*1024*1024,"model buffers retained after unloading")
    }
    print(String(data:try JSONSerialization.data(withJSONObject:receipts,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
} else if CommandLine.arguments.count==4 && CommandLine.arguments[1]=="--tokenizer-check" {
    let t=try ChatTokenizer(directory:URL(fileURLWithPath:CommandLine.arguments[2]))
    let cases=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[3]))) as! [[String:Any]]
    var failed=0
    for (i,c) in cases.enumerated() {
        let actual=try t.encode(c["text"] as! String),expected=c["ids"] as! [Int]
        if actual != expected {failed+=1;print("Mismatch \(i): \(actual.prefix(20)) != \(expected.prefix(20))")}
    }
    print("Tokenizer \(cases.count-failed)/\(cases.count)");if failed>0 {exit(2)}
} else if CommandLine.arguments.count==5 && CommandLine.arguments[1]=="--chat-check" {
    let fixture=try JSONSerialization.jsonObject(with:Data(contentsOf:URL(fileURLWithPath:CommandLine.arguments[3]))) as! [String:Any]
    let session=try ChatSession(directory:URL(fileURLWithPath:CommandLine.arguments[2]),system:fixture["system"] as! String,contextLimit:fixture["contextLimit"] as? Int ?? 2048,promptProfile:fixture["profile"] as? String ?? "baseline")
    try session.warmDecode()
    var turns=[ChatTurn]()
    for input in fixture["inputs"] as! [String] {turns.append(try session.reply(input,maxTokens:fixture["maxTokens"] as? Int ?? 48))}
    try JSONEncoder().encode(turns).write(to:URL(fileURLWithPath:CommandLine.arguments[4]))
    print("Native chat complete: \(turns.count) turns; cache hits \(turns.map{$0.cacheHitTokens})")
} else if CommandLine.arguments.dropFirst().first=="--prepare-regression" {
    let result=try prepareRegression()
    print(String(data:try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]),encoding:.utf8)!)
    if result["all_passed"] as? Bool != true {exit(2)}
} else if CommandLine.arguments.dropFirst().first=="--projection-regression" && CommandLine.arguments.count==3 {
    let result=try projectionRegression(URL(fileURLWithPath:CommandLine.arguments[2]))
    let data=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
    print(String(data:data,encoding:.utf8)!)
    if result["all_passed"] as? Bool != true {exit(2)}
} else if CommandLine.arguments.dropFirst().first=="--static-regression" {
    let result=staticTailRegression()
    let data=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
    print(String(data:data,encoding:.utf8)!)
    if result["all_passed"] as? Bool != true {exit(2)}
} else if CommandLine.arguments.dropFirst().first=="--rope-regression" {
    let data=try JSONSerialization.data(withJSONObject:ropeRegression(),options:[.prettyPrinted,.sortedKeys])
    print(String(data:data,encoding:.utf8)!)
} else if CommandLine.arguments.count==4 {
    Memory.cacheLimit=256*1024*1024
    do {
        guard let policy=ProjectionPolicy(rawValue:ProcessInfo.processInfo.environment["ST_SWIFT_PROJECTIONS"] ?? "baseline")
        else {throw EngineError.invalidConfiguration("unknown projection policy")}
        guard let deferred=Int(ProcessInfo.processInfo.environment["ST_SWIFT_DEFERRED_GDN"] ?? "1") else {throw EngineError.invalidConfiguration("invalid deferred GDN interval")}
        guard let history=HistoryPolicy(rawValue:ProcessInfo.processInfo.environment["ST_SWIFT_GDN_HISTORY"] ?? "dynamic") else {throw EngineError.invalidConfiguration("invalid GDN history policy")}
        guard let prepare=GDNPreparePolicy(rawValue:ProcessInfo.processInfo.environment["ST_SWIFT_GDN_PREPARE"] ?? "baseline") else {throw EngineError.invalidConfiguration("invalid GDN prepare policy")}
        let model=try Int4Model(directory:URL(fileURLWithPath:CommandLine.arguments[1]),projectionPolicy:policy,deferredInterval:deferred,historyPolicy:history,gdnPreparePolicy:prepare)
        let trace=ProcessInfo.processInfo.environment["ST_SWIFT_TRACE"]
        model.traceEnabled=trace != nil
        let execution=DecodeExecution(rawValue:ProcessInfo.processInfo.environment["ST_SWIFT_EXECUTION"] ?? "dynamic")
        guard let execution else {throw EngineError.invalidConfiguration("unknown execution mode")}
        let benchmark=ProcessInfo.processInfo.environment["ST_SWIFT_BENCHMARK"]
        let packet=URL(fileURLWithPath:CommandLine.arguments[2])
        guard let chunk=Int(ProcessInfo.processInfo.environment["ST_SWIFT_DECODE_CHUNK"] ?? "1"),[1,2,4,8].contains(chunk)
        else {throw EngineError.invalidConfiguration("decode chunk must be 1/2/4/8")}
        let result:[String:Any]
        if benchmark=="projections" {result=try model.benchmarkProjections(packet)}
        else if benchmark=="chunks" {result=try model.benchmarkChunks(packet)}
        else if benchmark=="chunkState" {result=try model.chunkStateRegression(packet)}
        else if benchmark=="deferredState" {result=try model.deferredStateRegression(packet)}
        else if benchmark=="deferredMemory" {result=try model.validatePacket(packet,execution:.compiled,memoryProbe:true)}
        else if benchmark=="deferred" {result=try model.benchmarkDeferred(packet)}
        else if benchmark=="historySuite" {result=try model.validateHistorySuite(packet)}
        else if benchmark=="historyState" {result=try model.historyStateSuite(packet)}
        else if benchmark=="historyProbe" {result=try model.probeHistoryRequest(packet)}
        else if benchmark=="history" {result=try model.benchmarkHistory(packet)}
        else if benchmark=="prepareSuite" {result=try model.validatePrepareSuite(packet)}
        else if benchmark=="prepareState" {result=try model.prepareStateRegression(packet)}
        else if benchmark=="prepareBench" {result=try model.benchmarkPrepare(packet)}
        else if benchmark=="prepareRequests" {result=try model.benchmarkPrepareRequests(packet)}
        else if benchmark=="1" {result=try model.benchmarkPacket(packet)}
        else {result=try model.validatePacket(packet,execution:execution,decodeChunkSize:chunk)}
        let data=try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys])
        try data.write(to:URL(fileURLWithPath:CommandLine.arguments[3]))
        if let trace {try model.saveTrace(URL(fileURLWithPath:trace))}
        print(String(data:data,encoding:.utf8)!)
        if let parity=result["gold_token_equal"] as? [String:[String:Bool]],
           parity.values.contains(where:{$0.values.contains(false)}) {exit(2)}
        if let parity=result["token_equal_to_dynamic"] as? [String:Bool],parity.values.contains(false) {exit(2)}
        if let parity=result["token_equal_to_baseline"] as? [String:Bool],parity.values.contains(false) {exit(2)}
        if let parity=result["token_equal_to_chunk1"] as? [String:Bool],parity.values.contains(false) {exit(2)}
        if let passed=result["all_passed"] as? Bool,!passed {exit(2)}
        if result["prefix_unchanged"] as? Bool == false {exit(2)}
    } catch {fputs("\(error)\n",stderr);exit(1)}
} else {print(runtimeSmoke())}
