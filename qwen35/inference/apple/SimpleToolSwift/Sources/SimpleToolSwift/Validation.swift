import Foundation
import CryptoKit
import MLX

extension Int4Model {
    public func validatePacket(_ url:URL,execution:DecodeExecution = .dynamic,decodeChunkSize:Int=1,memoryProbe:Bool=false,prefixCache:[LayerState]?=nil,batchOnly:Bool=false,checkPrefix:Bool=false) throws ->[String:Any] {
        guard [1,2,4,8].contains(decodeChunkSize),decodeChunkSize==1 || execution == .compiled else {
            throw EngineError.invalidConfiguration("decode chunks 2/4/8 require compiled execution")
        }
        guard deferredInterval==1 || execution == .compiled else {throw EngineError.invalidConfiguration("deferred GDN requires compiled execution")}
        guard let packet=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=packet["prompt_ids"] as? [Int],let heads=packet["heads"] as? [String],
              let markers=packet["marker_ids"] as? [String:[Int]],let gold=packet["gold"] as? [String:[Int]],
              let count=packet["max_tokens"] as? Int,let ignoreStops=packet["ignore_stops"] as? Bool,
              count>0,!ids.isEmpty,(Set(heads)==Set(["speech","expression","action","command"]) || Set(heads)==Set(SimpleToolProtocol.heads)),
              heads.count==4,heads.allSatisfy{markers[$0]?.isEmpty==false && gold[$0] != nil},
              Set(heads.compactMap{markers[$0]?.count}).count==1 else {
            throw EngineError.invalidPacket("requires four equal-length complete markers and fixed-length gold")
        }
        if let prefixCache {
            guard prefixCache.count==c.layers,prefixCache.allSatisfy({state in
                state.offset==ids.count && state.prefixK==nil &&
                    (state.conv?.dim(0)==1 || state.k?.dim(0)==1) &&
                    (state.matrix?.dim(0)==1 || state.v?.dim(0)==1)
            }) else {throw EngineError.invalidPacket("supplied prefix must be a single unbranched matching-length prefill")}
        }
        let start=Date(),source=prefixCache ?? prefill(ids),prefillSeconds=prefixCache==nil ? Date().timeIntervalSince(start) : 0
        func sourceDigest()->[String] {source.flatMap{$0.arrays}.map{SHA256.hash(data:$0.asData(access:.copy).data).map{String(format:"%02x",$0)}.joined()}}
        let hashChecked=memoryProbe || checkPrefix
        let before=hashChecked ? sourceDigest() : []
        if memoryProbe {Memory.clearCache();Memory.peakMemory=0}
        let initialActive=Memory.activeMemory
        let eos=packet["eos_id"] as? Int,closes=packet["close_ids"] as? [String:[Int]] ?? [:]
        if !ignoreStops && (eos==nil || heads.contains{closes[$0]==nil}) {
            throw EngineError.invalidPacket("missing natural stop contract")
        }
        var outputs=[String:[String:[Int]]](),timings=[String:Double](),traces=[String:[Int]]()
        var stopReasons=[String:[String:String]]()
        var decoded=[String:[String:String]](),decoderEqual=[String:[String:Bool]]()
        var discardedAhead=[String:Int](),evaluationBlocks=[String:Int]()
        for mode in (memoryProbe || batchOnly) ? ["batch4"] : ["serial","batch4"] {
            let begin=Date();var result=[String:[Int]](),trace=[Int](),reasons=[String:String](),discarded=0,blocks=0
            let groups=mode=="serial" ? heads.map{[$0]} : [heads]
            for group in groups {
                let capacity=count+markers[group[0]]!.count
                let cache=source.map{mode=="serial" && execution == .dynamic ? $0.copy() : $0.fork(batch:group.count,capacity:execution == .dynamic ? nil : capacity)}
                let compiled=execution == .compiled ? CompiledDecode(self,cache) : nil
                func forward(_ token:MLXArray,project:Bool=true)->MLXArray {
                    if let compiled {return compiled.forward(token,project:project)}
                    if project {return logits(token,cache:cache).argMax(axis:-1).asType(.int32)}
                    eval(hidden(token,cache:cache));return MLXArray.zeros([token.dim(0)],dtype:.int32)
                }
                var next:MLXArray?
                for i in 0..<markers[group[0]]!.count {
                    let token=MLXArray(group.map{markers[$0]![i]},[group.count,1])
                    next=forward(token,project:i==markers[group[0]]!.count-1)
                    eval([next!]+cache.flatMap{$0.arrays})
                }
                var raw=group.map{_ in [Int]()}
                var active=Array(group.indices)
                var generated=0
                while generated<count {
                    let blockCount=min(decodeChunkSize,count-generated)
                    let block:MLXArray
                    if blockCount>1 {block=compiled!.advancePending(next!,count:blockCount)}
                    else {block=next!.reshaped(1,active.count)}
                    eval([block]+cache.flatMap{$0.arrays});blocks+=1
                    trace.append(contentsOf:Array(repeating:active.count,count:blockCount))
                    let chosen=block.asArray(Int32.self).map{Int($0)}
                    var keep=[Int]()
                    for (row,index) in active.enumerated() {
                        var ended=false
                        for column in 0..<blockCount {
                            let token=chosen[column*active.count+row]
                            raw[index].append(token)
                            let close=closes[group[index]] ?? []
                            let tailText=try decoder.decode(Array(raw[index].suffix(16)))
                            ended = !ignoreStops && (token==eos ||
                                (!close.isEmpty && Array(raw[index].suffix(close.count))==close) ||
                                tailText.contains("</\(group[index])>"))
                            if ended {
                                reasons[group[index]]=token==eos ? "eos" : "channel_close"
                                discarded+=blockCount-column-1;break
                            }
                        }
                        if !ended {keep.append(row)}
                    }
                    generated+=blockCount
                    if keep.isEmpty || generated==count {break}
                    if keep.count != active.count {cache.forEach{$0.filter(MLXArray(keep))};evaluate(cache)}
                    let priorBatch=active.count
                    active=keep.map{active[$0]}
                    next=forward(MLXArray(keep.map{chosen[(blockCount-1)*priorBatch+$0]},[keep.count,1]))
                    // The next block evaluates this transition together with
                    // its intermediate steps, without a second host barrier.
                }
                for (h,tokens) in zip(group,raw) {result[h]=tokens;if reasons[h]==nil {reasons[h]="max_tokens"}}
            }
            outputs[mode]=result;timings[mode]=Date().timeIntervalSince(begin);traces[mode]=trace
            stopReasons[mode]=reasons
            discardedAhead[mode]=discarded;evaluationBlocks[mode]=blocks
            decoded[mode]=try result.mapValues{try decoder.decode($0)}
            if let goldText=packet["gold_text"] as? [String:String] {
                decoderEqual[mode]=Dictionary(uniqueKeysWithValues:heads.map{($0,decoded[mode]![$0]==goldText[$0])})
            }
        }
        let equal=outputs.mapValues{result in Dictionary(uniqueKeysWithValues:heads.map{($0,result[$0]==gold[$0])})}
        let decodePeak=Memory.peakMemory,finalActive=Memory.activeMemory,finalCache=Memory.cacheMemory
        let immutable = !hashChecked || before==sourceDigest()
        return ["outputs":outputs,"gold_token_equal":equal,"decode_s":timings,"prefill_s":prefillSeconds,
                "weight_bytes":weightBytes,"mlx_peak_bytes":decodePeak,"ignore_stops":ignoreStops,
                "memory_probe":memoryProbe,"initial_active_bytes":initialActive,"final_active_bytes":finalActive,
                "final_cache_bytes":finalCache,"prefix_hash_checked":hashChecked,
                "prefix_unchanged":hashChecked ? immutable as Any : NSNull(),
                "memory_scope":memoryProbe ? "one model and B4 request after prefill/allocator clear; MLX active allocations, not physical RSS or phone RAM" : "validation serial and batch paths coexist; not deployment RAM",
                "active_batch_by_step":traces,"decoded":decoded,"decoded_gold_equal":decoderEqual,
                "stop_reasons":stopReasons,
                "prefix_kv_bytes_once":source.reduce(0){$0+($1.k?.nbytes ?? 0)+($1.v?.nbytes ?? 0)},
                "compute":"float32","weight_bits":4,"prefill_count":prefixCache==nil ? 1 : 0,
                "prefix_cache_supplied":prefixCache != nil,
                "execution":execution.rawValue,
                "projection_policy":projectionPolicy.rawValue,
                "decode_chunk_size":decodeChunkSize,"discarded_after_stop":discardedAhead,
                "deferred_gdn_interval":deferredInterval,
                "gdn_history_policy":historyPolicy.rawValue,
                "gdn_prepare_policy":gdnPreparePolicy.rawValue,
                "decode_evaluation_blocks":evaluationBlocks,
                "limitation":"prompt and marker encoding external; no image or iPhone runtime acceptance"]
    }
}
