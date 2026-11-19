import Foundation
import CryptoKit
import MLX

extension Int4Model {
    // Real checkpoint state equality, including reconstruction after deferred
    // replay. Baseline and candidate share every immutable weight array.
    public func deferredStateRegression(_ url:URL,tokens:Int=19) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],
              let markers=p["marker_ids"] as? [String:[Int]],heads.count==4,!ids.isEmpty,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),Set(heads.compactMap{markers[$0]?.count}).count==1
        else {throw EngineError.invalidPacket("deferred state packet")}
        let source=prefill(ids)
        func digest()->[String] {
            source.flatMap{$0.arrays}.map{SHA256.hash(data:$0.asData(access:.copy).data).map{String(format:"%02x",$0)}.joined()}
        }
        let before=digest(),baseline=Int4Model(sharing:self,policy:projectionPolicy,deferredInterval:1)
        var cases=[[String:Any]]()
        for batch in 1...4 {
            let selected=Array(heads.prefix(batch)),length=markers[selected[0]]!.count,capacity=length+tokens-1
            func run(_ model:Int4Model,_ chunk:Int)->([Int],[MLXArray],Int) {
                let cache=source.map{$0.fork(batch:batch,capacity:capacity)},compiled=CompiledDecode(model,cache)
                var prediction:MLXArray!
                for i in 0..<length {
                    prediction=compiled.forward(MLXArray(selected.map{markers[$0]![i]},[batch,1]),project:i==length-1)
                    eval([prediction!]+cache.flatMap{$0.arrays})
                }
                var raw=[Int](),generated=0
                while generated<tokens {
                    let n=min(chunk,tokens-generated),block=compiled.advancePending(prediction,count:n)
                    eval([block]+cache.flatMap{$0.arrays})
                    let values=block.asArray(Int32.self).map{Int($0)}
                    // Canonical token-major layout for both chunk sizes.
                    raw.append(contentsOf:values);generated+=n
                    if generated<tokens {prediction=compiled.forward(MLXArray(Array(values.suffix(batch)),[batch,1]))}
                }
                let canonical=model.canonicalState(cache);eval(canonical)
                return (raw,canonical,compiled.used)
            }
            let reference=run(baseline,1)
            for interval in [2,4,8,16] {
                let candidate=Int4Model(sharing:self,policy:projectionPolicy,deferredInterval:interval)
                for chunk in [1,8] {
                    let result=run(candidate,chunk)
                    let differences=zip(reference.1,result.1).enumerated().compactMap{index,pair -> Int? in
                        pair.0.shape==pair.1.shape && pair.0.dtype==pair.1.dtype &&
                        pair.0.asData(access:.copy).data==pair.1.asData(access:.copy).data ? nil : index
                    }
                    let passed=result.0==reference.0 && differences.isEmpty && result.2==capacity
                    cases.append(["batch":batch,"interval":interval,"chunk":chunk,"passed":passed,
                        "token_equal":result.0==reference.0,"state_bitwise_equal":differences.isEmpty,
                        "different_state_indices":differences,"used":result.2,"capacity":capacity])
                }
            }
        }
        let immutable=before==digest()
        return ["cases":cases,"source_digest_before":before,"source_unchanged":immutable,
                "all_passed":immutable && cases.allSatisfy{$0["passed"] as? Bool == true},
                "weight_bits":4,"compute":"float32","projection_policy":projectionPolicy.rawValue,
                "scope":"real model B1..4; intervals2/4/8/16; chunks1/8; 19 tokens; exact suffix capacity; materialized matrix/conv/tail byte equality",
                "limitation":"Mac execution; natural stops separate; no phone or multi-turn acceptance"]
    }
}
