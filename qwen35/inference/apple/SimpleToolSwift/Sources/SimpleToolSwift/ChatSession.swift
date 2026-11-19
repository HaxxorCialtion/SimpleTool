import Foundation
import MLX
import Darwin

@inline(__always) private func chatClock()->Double {Double(clock_gettime_nsec_np(CLOCK_UPTIME_RAW))*1e-9}

/// Explicit caller policy; speed/parity benchmarks retain raw greedy decoding.
public enum ChatDecodingPolicy:String {case greedy,conversation}

public struct ChatDiagnostics:Codable,Sendable {
    public let promptText:String,promptTokenIDs:[Int],systemPrompt:String
    public let phases:[String:Double],prefillChunkSeconds:[Double],markerStepSeconds:[Double],decodeStepSeconds:[Double]
    public let activeBatchSizes:[Int],headFirstTokenSeconds:[String:Double],headStopSeconds:[String:Double]
    public let firstTokenLatencySeconds:Double,maxTokens:Int
    public var decodingPolicy:String? = nil
    public var temperature:Float? = nil
}

public struct ChatTurn:Codable,Identifiable,Sendable {
    public var id:UUID
    public let createdAt:Date?
    public let user:String,channels:[String:String],stops:[String:String]
    public let prefillSeconds:Double,decodeSeconds:Double,cacheHitTokens:Int,newPrefillTokens:Int
    public let tokenIDs:[String:[Int]]?
    public let diagnostics:ChatDiagnostics?,modelID:String?,promptProfile:String?
    public init(user:String,channels:[String:String],stops:[String:String],prefillSeconds:Double,decodeSeconds:Double,cacheHitTokens:Int,newPrefillTokens:Int,tokenIDs:[String:[Int]]?=nil,diagnostics:ChatDiagnostics?=nil,modelID:String?=nil,promptProfile:String?=nil) {
        id=UUID();createdAt=Date();self.user=user;self.channels=channels;self.stops=stops
        self.prefillSeconds=prefillSeconds;self.decodeSeconds=decodeSeconds
        self.cacheHitTokens=cacheHitTokens;self.newPrefillTokens=newPrefillTokens
        self.tokenIDs=tokenIDs
        self.diagnostics=diagnostics;self.modelID=modelID;self.promptProfile=promptProfile
    }
}

/// Call every method on one serial inference lane. Token cancellation is checked
/// between GPU evaluations. Shared prefix is immutable, branch states independent.
public final class ChatSession {
    public static let heads=SimpleToolProtocol.heads
    public static let defaultSystem=SimpleToolProtocol.defaultSystem
    private let model:Int4Model,tokenizer:ChatTokenizer
    private let systemPrompt:String,profile:String,modelName:String
    public private(set) var loadMetrics=[String:Double]()
    private var systemIDs:[Int],systemCache:[LayerState]
    private var savedIDs:[Int],savedCache:[LayerState]
    private var taskSystemIDs=[Int](),taskSystemCache=[LayerState]()
    private var diagnosticIgnoreHeadEOS=false // scoped only by benchmarkFixedHeads
    private var conditionalHistory:Bool
    public private(set) var turns=[ChatTurn]()
    public let contextLimit:Int
    public var decodingPolicy:ChatDecodingPolicy = .greedy
    /// Sampling temperature for conversation policy, across all heads. Zero is argmax.
    /// Raw greedy and fixed-step diagnostics always bypass stochastic selection.
    public var temperature:Float = 0.7 {
        didSet {precondition(temperature.isFinite && (0...2).contains(temperature))}
    }
    public var precision:InferencePrecision {model.precision}
    public var numericalPolicy:String {model.numerics.rawValue}
    public var decodeProjectionPolicy:String {model.decodeProjectionMode}
    private var decodeStrideOverride:Int?
    private var decodeProfiling=false
    /// Diagnostic tuning only. Configure on the exclusive inference lane.
    public func configureDecodeTuning(projections:String="reuse",stride:Int?=nil,fusedPrepare:Bool=false,fusedQ8:Bool=false,profile:Bool=false) {
        precondition(["legacy","reuse","reuse-qmv","reuse-consistent"].contains(projections))
        precondition(stride == nil || [0,4,8,16].contains(stride!))
        model.decodeProjectionMode=projections;decodeStrideOverride=stride
        model.decodeFusedPrepare=fusedPrepare;decodeProfiling=profile;model.operatorProfiling=false
        model.decodeFusedQ8=fusedQ8
        model.projectionTiles=[:]
        model.operatorSeconds=[:];model.operatorCalls=[:]
    }
    public var operatorProfile:[String:Double] {model.operatorSeconds}
    public var operatorProfileCalls:[String:Int] {model.operatorCalls}
    public func benchmarkProjectionTiles(batch:Int=4)->[[String:Any]] {model.projectionTileProbe(batch:batch)}
    public func configureProjectionTiles(_ entries:[[String:Int]]) {
        model.projectionTiles=[:]
        for e in entries {
            guard let kind=e["kind"],let b=e["batch"],let n=e["n"],let k=e["k"],let r=e["rows"],let t=e["threads"],
                  (0...2).contains(kind),(1...4).contains(b),n>0,k>0,[1,2,4,8].contains(r),[64,128,256].contains(t) else {preconditionFailure("invalid projection tile")}
            model.projectionTiles["\(kind)/\(b)/\(n)/\(k)"]=(r,t)
        }
    }
    /// Measured iPhone policy. Keep memory-safe stride4 for 4B, packed weights,
    /// FP32 GDN state and the original tiles. Small isolated tile gains were
    /// not established as a repeatable E2E gain under mobile clock variation.
    public func useIPhoneDecodeDefaults() {
        // Matched build42 phone sweeps: the 0.8B shape benefits from shared
        // B1..B4 arithmetic to preserve the long-chat parity fixture. 2B/4B
        // retain compact B1 QMV; select by architecture, not directory name.
        let smallShape=model.c.hidden==1024 && model.c.layers==24
        configureDecodeTuning(projections:precision == .fp32 ? "reuse":smallShape ? "reuse-consistent":"reuse-qmv")
    }
    /// Opt-in block evaluation for measured memory/latency experiments. Lane-only.
    public var layerEvaluationStride=0 {
        didSet {
            precondition([0,4,8,16].contains(layerEvaluationStride))
            model.prefillEvaluationStride=layerEvaluationStride
        }
    }
    public init(directory:URL,system:String=ChatSession.defaultSystem,contextLimit:Int=8192,promptProfile:String="baseline",initialLayerEvaluationStride:Int=0,allocatorCacheBytes:Int=128*1024*1024,precision:InferencePrecision = .fp32,numerics:NumericalPolicy = .standard,cancelled:()->Bool={false}) throws {
        self.contextLimit=contextLimit
        systemPrompt=system;profile=promptProfile;modelName=directory.lastPathComponent
        guard [0,4,8,16].contains(initialLayerEvaluationStride),allocatorCacheBytes>=0 else {throw EngineError.invalidConfiguration("Invalid initial memory policy")}
        Memory.cacheLimit=allocatorCacheBytes
        if cancelled() {throw CancellationError()}
        let start=chatClock()
        model=try Int4Model(directory:directory,projectionPolicy:.both,precision:precision,numerics:numerics)
        model.prefillEvaluationStride=initialLayerEvaluationStride
        layerEvaluationStride=initialLayerEvaluationStride
        if cancelled() {throw CancellationError()}
        let weightsLoaded=chatClock()
        tokenizer=try ChatTokenizer(directory:directory)
        let protocolURL=directory.appendingPathComponent("simpletool_config.json")
        conditionalHistory=FileManager.default.fileExists(atPath:protocolURL.path)
        if conditionalHistory {
            let data=try JSONSerialization.jsonObject(with:Data(contentsOf:protocolURL)) as? [String:Any]
            guard data?["heads"] as? [String] == Self.heads,
                  let declared=data?["head_token_ids"] as? [String:Int] else {throw EngineError.invalidConfiguration("Unexpected SimpleTool heads")}
            for head in Self.heads {
                guard try tokenizer.encode("<\(head)>")==[declared[head] ?? -1] else {throw EngineError.invalidConfiguration("Branch tokenizer mismatch: \(head)")}
            }
        }
        if cancelled() {throw CancellationError()}
        let tokenizerLoaded=chatClock()
        systemIDs=try tokenizer.encode(ChatTokenizer.message("system",system))
        guard systemIDs.count<contextLimit-128 else {throw EngineError.invalidPacket("System prefix too long")}
        let encoded=chatClock()
        systemCache=model.newCache()
        for offset in stride(from:0,to:systemIDs.count,by:64) {
            if cancelled() {throw CancellationError()}
            systemCache=model.prefill(Array(systemIDs[offset..<min(offset+64,systemIDs.count)]),cache:systemCache,chunkSize:64)
        }
        if cancelled() {throw CancellationError()}
        savedIDs=systemIDs;savedCache=systemCache
        loadMetrics=["weights_load":weightsLoaded-start,"tokenizer_load":tokenizerLoaded-weightsLoaded,"system_tokenize":encoded-tokenizerLoaded,"system_prefill":chatClock()-encoded]
    }
    public var weightBytes:Int {model.weightBytes}
    public var warmPrefixTokens:Int {systemIDs.count}
    public func restore(_ history:[ChatTurn]) {turns=history;savedIDs=systemIDs;savedCache=systemCache}
    public func clear() {restore([]);Memory.clearCache()}
    public func warmDecode() throws {
        let ids=try tokenizer.encode(ChatTokenizer.generationSuffix)
        let seed=model.prefill(ids,cache:systemCache.map{$0.copy()},chunkSize:64)
        _ = try decode(seed,maxTokens:1,headNames:Self.heads,cancelled:{false},onUpdate:{_ in})
        Memory.clearCache()
    }
    /// Includes user-message framing and the empty-think generation suffix.
    /// Every generated fixture is re-encoded to verify its exact token count.
    public func benchmarkInput(newPrefillTokens target:Int) throws ->String {
        let suffix=try tokenizer.encode(ChatTokenizer.generationSuffix).count
        guard target>=64,systemIDs.count+target+24<=contextLimit else {
            throw EngineError.invalidPacket("该提示词下无法测试 \(target) token，请切换较短的提示词。")
        }
        let intro="这是长输入性能测试。请阅读材料，最后只用一句话概括，不要逐段复述。材料：\n"
        let paragraph="今天我整理了书桌，把几本书放回书架，然后走到窗边看了一会儿天空。下午准备散步，回来后做饭，并记录一天里值得记住的小事。\n"
        let ending="\n请概括材料。"
        func count(_ text:String) throws ->Int {try tokenizer.encode(ChatTokenizer.message("user",text)).count+suffix}
        var body=intro
        while try count(body+paragraph+ending)<=target {body+=paragraph}
        // A partial natural paragraph avoids padding most of the context with
        // repetitive synthetic tokens. Small ASCII padding fills the remainder.
        for character in paragraph {
            if try count(body+String(character)+ending)>target-8 {break}
            body.append(character)
        }
        for padding in [" x"," z"," a"] {
            var candidate=body+ending
            for _ in 0...target {
                let n=try count(candidate)
                if n==target {return candidate}
                if n>target {break}
                candidate+=padding
            }
        }
        throw EngineError.invalidPacket("无法构造精确长度的 benchmark 输入。")
    }
    /// Replays only the immutable system cache. Restore normal conversation and
    /// cache on success, cancellation or failure without reloading any weights.
    public func benchmarkReply(_ input:String,newPrefillTokens:Int,maxTokens:Int=24,headNames:[String]=ChatSession.heads,cancelled:()->Bool={false}) throws ->ChatTurn {
        let oldTurns=turns,oldIDs=savedIDs,oldCache=savedCache,oldProtocol=conditionalHistory
        conditionalHistory=false
        defer {conditionalHistory=oldProtocol;turns=oldTurns;savedIDs=oldIDs;savedCache=oldCache;Memory.clearCache()}
        turns=[];savedIDs=systemIDs;savedCache=systemCache
        let turn=try reply(input,maxTokens:maxTokens,headNames:headNames,cancelled:cancelled)
        guard turn.newPrefillTokens==newPrefillTokens,turn.cacheHitTokens==systemIDs.count else {
            throw EngineError.invalidPacket("Benchmark token/cache count mismatch")
        }
        return turn
    }
    /// Diagnostic throughput only. Executes real B1/B4 forwards for the full
    /// step budget even after EOS; never use this generated text in business UI.
    /// benchmarkReply restores all normal conversation/cache state on exit.
    public func benchmarkFixedHeads(_ input:String,newPrefillTokens:Int,decodeSteps:Int=32,headNames:[String]=ChatSession.heads,cancelled:()->Bool={false}) throws ->ChatTurn {
        precondition(!diagnosticIgnoreHeadEOS)
        diagnosticIgnoreHeadEOS=true
        defer {diagnosticIgnoreHeadEOS=false}
        return try benchmarkReply(input,newPrefillTokens:newPrefillTokens,maxTokens:decodeSteps,headNames:headNames,cancelled:cancelled)
    }
    public func reply(_ input:String,maxTokens:Int=128,headNames:[String]=ChatSession.heads,memoryRecords:[[String:Any]]=[],preparedPayload:String?=nil,cancelled:()->Bool={false},onUpdate:([String:String])->Void={_ in}) throws ->ChatTurn {
        let requestStart=chatClock()
        guard !input.trimmingCharacters(in:.whitespacesAndNewlines).isEmpty,(1...4096).contains(maxTokens),!headNames.isEmpty else {throw EngineError.invalidPacket("Empty input or invalid output limit")}
        let suffix=try tokenizer.encode(ChatTokenizer.generationSuffix)
        let now=Date()
        var selectedHistory=preparedPayload == nil ? turns : []
        var payload=try preparedPayload ?? (conditionalHistory ? SimpleToolProtocol.payload(input:input,history:selectedHistory,memories:memoryRecords,now:now) : input)
        if conditionalHistory {
            while try !selectedHistory.isEmpty && (tokenizer.encode(ChatTokenizer.message("user",payload))).count+systemIDs.count+suffix.count+maxTokens>contextLimit {
                selectedHistory.removeFirst()
                payload=try SimpleToolProtocol.payload(input:input,history:selectedHistory,memories:memoryRecords,now:now)
            }
        }
        var tailText=ChatTokenizer.message("user",payload)
        var tail=try tokenizer.encode(tailText)
        guard systemIDs.count+tail.count+suffix.count+maxTokens<=contextLimit else {throw EngineError.invalidPacket("这条消息太长，请缩短后重试。")}
        // Encode a contiguous suffix of complete turns once, starting from the
        // newest. A long saved transcript must not trigger repeated full replay.
        for t in (conditionalHistory ? [] : turns).reversed() {
            let assistant=Self.heads.map{"<\($0)>\(t.channels[$0] ?? "")</\($0)>"}.joined()
            let text=ChatTokenizer.message("user",t.user)+ChatTokenizer.message("assistant",assistant)
            let ids=try tokenizer.encode(text)
            if systemIDs.count+ids.count+tail.count+suffix.count>contextLimit {break}
            tail=ids+tail
            tailText=text+tailText
        }
        let bodyIDs=systemIDs+tail
        if cancelled() {throw CancellationError()}
        let reuse=bodyIDs.starts(with:savedIDs),hit=reuse ? savedIDs.count : systemIDs.count
        let start=chatClock()
        let source=(reuse ? savedCache : systemCache).map{$0.copy()}
        let cloned=chatClock()
        let remainder=Array(bodyIDs.dropFirst(hit))
        var chunks=[Double]()
        for offset in stride(from:0,to:remainder.count,by:64) {
            if cancelled() {throw CancellationError()}
            let chunkStart=chatClock()
            _ = model.prefill(Array(remainder[offset..<min(offset+64,remainder.count)]),cache:source,chunkSize:64)
            chunks.append(chatClock()-chunkStart)
        }
        let common=source
        savedCache=common;savedIDs=bodyIDs
        let suffixStart=chatClock()
        let forkSource=model.prefill(suffix,cache:common.map{$0.copy()},chunkSize:64)
        let begin=chatClock(),pp=begin-start
        let generated=try decode(forkSource,maxTokens:maxTokens,headNames:headNames,cancelled:cancelled,onUpdate:onUpdate,honorEOS:!diagnosticIgnoreHeadEOS)
        let decodeSeconds=chatClock()-begin
        let diagnostic=ChatDiagnostics(promptText:ChatTokenizer.message("system",systemPrompt)+tailText+ChatTokenizer.generationSuffix,
            promptTokenIDs:bodyIDs+suffix,systemPrompt:systemPrompt,
            phases:["tokenize_and_history":start-requestStart,"cache_clone":cloned-start,
                    "incremental_prefill":chunks.reduce(0,+),"generation_suffix_prefill":begin-suffixStart,
                    "markers":generated.markerSeconds.reduce(0,+),"decode_total":decodeSeconds,
                    "request_total":chatClock()-requestStart],prefillChunkSeconds:chunks,
            markerStepSeconds:generated.markerSeconds,decodeStepSeconds:generated.stepSeconds,
            activeBatchSizes:generated.batches,headFirstTokenSeconds:generated.first,
            headStopSeconds:generated.finished,firstTokenLatencySeconds:begin-requestStart+(generated.first.values.min() ?? 0),maxTokens:maxTokens,decodingPolicy:diagnosticIgnoreHeadEOS ? "greedy":decodingPolicy.rawValue,temperature:diagnosticIgnoreHeadEOS || decodingPolicy == .greedy ? 0:temperature)
        let turn=ChatTurn(user:input,channels:generated.text,stops:generated.stops,
            prefillSeconds:pp,decodeSeconds:decodeSeconds,
            cacheHitTokens:hit,newPrefillTokens:bodyIDs.count+suffix.count-hit,tokenIDs:generated.tokens,
            diagnostics:diagnostic,modelID:modelName,promptProfile:profile)
        turns.append(turn)
        return turn
    }
    /// Ordinary autoregression for memory tasks: no branch marker and no chat history mutation.
    public func memoryReply(system:String,input:String,maxTokens:Int=2048,cancelled:()->Bool={false}) throws ->ChatTurn {
        try singleHeadReply(system:system,input:input,maxTokens:maxTokens,honorEOS:true,cancelled:cancelled)
    }
    /// Diagnostic timing only: runs the requested number of greedy AR steps even
    /// after EOS. Never use its text as a user-facing memory result.
    public func benchmarkAutoregression(system:String,input:String,decodeSteps:Int,cancelled:()->Bool={false}) throws ->ChatTurn {
        try singleHeadReply(system:system,input:input,maxTokens:decodeSteps,honorEOS:false,cancelled:cancelled)
    }
    private func singleHeadReply(system:String,input:String,maxTokens:Int,honorEOS:Bool,cancelled:()->Bool) throws ->ChatTurn {
        let begin=chatClock()
        let prefix=try tokenizer.encode(ChatTokenizer.message("system",system))
        let tail=try tokenizer.encode(ChatTokenizer.message("user",input)+ChatTokenizer.generationSuffix)
        guard !tail.isEmpty,prefix.count+tail.count+maxTokens<=contextLimit,(1...4096).contains(maxTokens) else {throw EngineError.invalidPacket("Memory task exceeds context/output budget")}
        let hit=prefix==taskSystemIDs ? prefix.count : 0
        if hit==0 {
            taskSystemCache=[];taskSystemIDs=[];Memory.clearCache()
            var cache=model.newCache()
            for offset in stride(from:0,to:prefix.count,by:64) {
                if cancelled() {throw CancellationError()}
                cache=model.prefill(Array(prefix[offset..<min(offset+64,prefix.count)]),cache:cache,chunkSize:64)
            }
            taskSystemIDs=prefix;taskSystemCache=cache
        }
        let cache=taskSystemCache.map{$0.copy()},body=Array(tail.dropLast())
        for offset in stride(from:0,to:body.count,by:64) {
            if cancelled() {throw CancellationError()}
            _ = model.prefill(Array(body[offset..<min(offset+64,body.count)]),cache:cache,chunkSize:64)
        }
        let decodeStart=chatClock()
        let result=try decode(cache,maxTokens:maxTokens,headNames:["text"],cancelled:cancelled,onUpdate:{_ in},initialTokens:[[tail.last!]],honorEOS:honorEOS)
        return ChatTurn(user:input,channels:result.text,stops:result.stops,prefillSeconds:decodeStart-begin,decodeSeconds:chatClock()-decodeStart,cacheHitTokens:hit,newPrefillTokens:prefix.count+tail.count-hit,tokenIDs:result.tokens,modelID:modelName,promptProfile:honorEOS ? "memory" : "diagnostic-fixed-ar-steps")
    }
    public func releaseMemoryCache() {taskSystemIDs=[];taskSystemCache=[];Memory.clearCache()}
    private func decode(_ source:[LayerState],maxTokens:Int,headNames:[String],cancelled:()->Bool,onUpdate:([String:String])->Void,initialTokens:[[Int]]?=nil,honorEOS:Bool=true) throws ->(text:[String:String],stops:[String:String],tokens:[String:[Int]],markerSeconds:[Double],stepSeconds:[Double],batches:[Int],first:[String:Double],finished:[String:Double]) {
        let decodeStart=chatClock()
        let layerEvaluationStride=decodeStrideOverride ?? self.layerEvaluationStride
        model.operatorProfiling=decodeProfiling
        model.operatorSeconds=[:];model.operatorCalls=[:]
        defer {model.operatorProfiling=false}
        let heads=headNames,markers=try initialTokens ?? heads.map{try tokenizer.encode("<\($0)>")}
        let closes=try heads.map{try tokenizer.encode("</\($0)>")},eos=try tokenizer.tokenID("<|im_end|>")
        let extraEOS=try tokenizer.tokenID("<|endoftext|>")
        guard Set(markers.map{$0.count}).count==1,!markers[0].isEmpty else {throw EngineError.invalidPacket("Unequal markers")}
        let caches=source.map{$0.fork(batch:heads.count,capacity:maxTokens+markers[0].count)}
        // Every caller provides fresh transient LayerState objects (system/task
        // caches are copied before the generation suffix). Ownership of this
        // seed transfers to the fork: do not retain a second B1 GDN state for
        // the whole B4 decode. Immutable prefix KV remains owned by caches.
        for state in source {
            state.conv=nil;state.matrix=nil;state.k=nil;state.v=nil
            state.k8=nil;state.v8=nil
        }
        // Preserve static shared-prefix tails and the fused int4 kernels, but
        // avoid whole-step compiled tapes: the pinned MLX build retains weight
        // allocations after unloading these tapes. Model switching must release
        // all model storage before the next checkpoint is loaded.
        let policy=honorEOS && initialTokens == nil ? decodingPolicy : .greedy
        let samplingTemperature=policy == .greedy ? Float(0):temperature
        var raw=Array(repeating:[Int](),count:heads.count),active=Array(0..<heads.count),stops=[String:String]()
        let compiled:CompiledDecode? = model.precision == .fp32 && policy == .greedy ? CompiledDecode(model,caches,compileStep:false,evaluationStride:layerEvaluationStride):nil
        func forward(_ tokens:MLXArray,project:Bool=true)->MLXArray {
            if let compiled {return compiled.forward(tokens,project:project)}
            let hidden=model.hidden(tokens,cache:caches,evaluateEvery:layerEvaluationStride)
            guard project else {return MLXArray.zeros([tokens.dim(0)],dtype:.int32)}
            let logits=model.linear(hidden[0...,-1,0...],"embed_tokens")
            return ConversationLogits.select(logits,policy:policy,heads:heads,active:active,history:raw,protected:Set([eos,extraEOS]+closes.flatMap{$0}),temperature:samplingTemperature)
        }
        var prediction:MLXArray?
        var markerSeconds=[Double](),stepSeconds=[Double](),batches=[Int](),first=[String:Double](),finished=[String:Double]()
        for pos in markers[0].indices {
            if cancelled() {throw CancellationError()}
            let markerStart=chatClock()
            prediction=forward(MLXArray(markers.map{$0[pos]},[heads.count,1]),project:pos==markers[0].count-1)
            eval([prediction!]+caches.flatMap{$0.arrays})
            markerSeconds.append(chatClock()-markerStart)
        }
        func visible()->[String:String] {
            Dictionary(uniqueKeysWithValues:heads.enumerated().map {i,h in
                let tokens=(raw[i].last==eos || raw[i].last==extraEOS) ? Array(raw[i].dropLast()) : raw[i]
                let text=(try? model.decoder.decode(tokens)) ?? ""
                return (h,text.components(separatedBy:"</\(h)>").first!.trimmingCharacters(in:.whitespacesAndNewlines))
            })
        }
        for step in 0..<maxTokens {
            if cancelled() {throw CancellationError()}
            let chosen=prediction!.asArray(Int32.self).map{Int($0)}
            batches.append(active.count)
            if step==0 {for index in active {first[heads[index]]=chatClock()-decodeStart}}
            var keep=[Int]()
            for (row,index) in active.enumerated() {
                let token=chosen[row];raw[index].append(token)
                let tail=try model.decoder.decode(Array(raw[index].suffix(16)))
                if honorEOS && (token==eos || token==extraEOS) {stops[heads[index]]="eos"}
                else if honorEOS && !conditionalHistory && initialTokens == nil && (raw[index].suffix(closes[index].count).elementsEqual(closes[index]) || tail.contains("</\(heads[index])>")) {stops[heads[index]]="channel_close"}
                else {keep.append(row)}
                if stops[heads[index]] != nil {finished[heads[index]]=chatClock()-decodeStart}
            }
            let current=visible()
            // Speech streams; structured channels become visible only at an actual stop.
            onUpdate(current.filter{$0.key=="speech" || stops[$0.key] != nil})
            if keep.isEmpty || step+1==maxTokens {break}
            if keep.count != active.count {
                let indices=MLXArray(keep)
                if layerEvaluationStride>0 {
                    // Shrink in blocks: a bulk take otherwise keeps every B4
                    // state alive while all B3/B2 replacements are allocated.
                    for first in stride(from:0,to:caches.count,by:layerEvaluationStride) {
                        let block=Array(caches[first..<min(first+layerEvaluationStride,caches.count)])
                        block.forEach{$0.filter(indices)}
                        model.evaluate(block)
                        Stream.gpu.synchronize()
                    }
                } else {caches.forEach{$0.filter(indices)};model.evaluate(caches)}
            }
            active=keep.map{active[$0]}
            let stepStart=chatClock()
            prediction=forward(MLXArray(keep.map{chosen[$0]},[keep.count,1]))
            eval([prediction!]+caches.flatMap{$0.arrays})
            stepSeconds.append(chatClock()-stepStart)
        }
        for head in heads where stops[head]==nil {stops[head]="max_tokens";finished[head]=chatClock()-decodeStart}
        return (visible(),stops,Dictionary(uniqueKeysWithValues:zip(heads,raw)),markerSeconds,stepSeconds,batches,first,finished)
    }
}
