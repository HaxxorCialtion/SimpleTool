import Foundation
import MLX

/// Per-branch generated-token history only. Caller prompts, expression and command
/// never receive text repetition penalties. Temperature applies to all active heads.
enum ConversationLogits {
    static func select(_ logits:MLXArray,policy:ChatDecodingPolicy,heads:[String],active:[Int],history:[[Int]],protected:Set<Int>,temperature:Float=0)->MLXArray {
        guard policy == .conversation else {return logits.argMax(axis:-1).asType(.int32)}
        let rows=active.enumerated().map {row,index -> MLXArray in
            var scores=logits[row].asType(.float32)
            guard heads[index] == "speech" || heads[index] == "inner" else {return sample(scores,temperature:temperature)}
            let recent=Array(history[index].suffix(64))
            let ids=Array(Set(recent).subtracting(protected)).sorted()
            if !ids.isEmpty {
                let indices=MLXArray(ids),values=scores[indices]
                scores[indices]=which(values .< 0,values*Float(1.1),values/Float(1.1))
            }
            // Allow a phrase twice, then block a third identical 3-token ngram.
            // This remains a decoding constraint, never post-hoc text deletion.
            let blocked=blockedTokens(recent).subtracting(protected)
            if !blocked.isEmpty {scores[MLXArray(blocked.sorted())]=MLXArray(Float(-Float.infinity))}
            return sample(scores,temperature:temperature)
        }
        return stacked(rows)
    }
    static func sample(_ scores:MLXArray,temperature:Float,key:MLXArray?=nil)->MLXArray {
        precondition(temperature.isFinite && (0...2).contains(temperature))
        if temperature == 0 {return scores.argMax(axis:-1).asType(.int32)}
        return MLXRandom.categorical(scores.asType(.float32)/temperature,key:key).asType(.int32)
    }
    static func blockedTokens(_ tokens:[Int])->Set<Int> {
        guard tokens.count>=5 else {return []}
        let suffix=Array(tokens.suffix(2))
        var counts=[Int:Int]()
        for start in 0..<(tokens.count-2) where Array(tokens[start..<start+2])==suffix {
            counts[tokens[start+2],default:0]+=1
        }
        return Set(counts.filter{$0.value>=2}.map{$0.key})
    }
}

/// Small numerical/branch-routing regression, independent of model quality.
public func validateConversationLogits() throws -> [String:Bool] {
    let logits=MLXArray([Float(0),10,9,Float(0),-2,-2.1,-5,-5,Float(0),Float(0),Float(0),Float(0)],[3,4])
    let heads=["speech","expression","inner","command"]
    let history=[[1],[],[0],[]]
    let b3=ConversationLogits.select(logits,policy:.conversation,heads:heads,active:[0,2,3],history:history,protected:[]).asArray(Int32.self)
    // positive penalty changes 10->9.09; keep it above 9. Negative -2 becomes -2.2.
    guard b3 == [1,1,0] else {throw EngineError.invalidPacket("conversation B3 branch routing/sign regression: \(b3)")}
    let test=MLXArray([Float(0),10,9.5,Float(0)],[1,4])
    let b1=ConversationLogits.select(test,policy:.conversation,heads:heads,active:[0],history:history,protected:[]).asArray(Int32.self)
    let protected=ConversationLogits.select(test,policy:.conversation,heads:heads,active:[0],history:history,protected:[1]).asArray(Int32.self)
    let command=ConversationLogits.select(test,policy:.conversation,heads:heads,active:[3],history:[[1],[],[],[1]],protected:[]).asArray(Int32.self)
    guard b1 == [2],protected == [1],command == [1],
          ConversationLogits.blockedTokens([1,2,3,1,2,3,1,2]) == [3],
          ConversationLogits.blockedTokens([1,2,3,1,2]).isEmpty else {throw EngineError.invalidPacket("conversation penalty/protection/ngram regression")}
    let draws=ConversationLogits.sample(broadcast(MLXArray([Float(0),1]),to:[16384,2]),temperature:0.7,key:MLXRandom.key(73)).asArray(Int32.self)
    let fraction=Double(draws.reduce(0,+))/Double(draws.count)
    let raw=ConversationLogits.select(test,policy:.greedy,heads:heads,active:[0],history:history,protected:[],temperature:2).asArray(Int32.self)
    guard abs(fraction-1/(1+exp(-1/0.7)))<0.02,raw == [1] else {throw EngineError.invalidPacket("temperature categorical distribution or raw-greedy isolation failed: \(fraction)")}
    return ["temperature_0_7_distribution":true,"greedy_ignores_temperature":true,"b4_to_b3_original_head_mapping":true,"b1_speech_penalty":true,"negative_logits_sign":true,"eos_protected":true,"structured_command_unchanged":true,"third_ngram_only":true]
}
