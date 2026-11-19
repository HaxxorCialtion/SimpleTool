import Foundation
import CryptoKit
import MLX

extension Int4Model {
    public func chunkStateRegression(_ url:URL,tokens:Int=17) throws ->[String:Any] {
        guard let packet=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=packet["prompt_ids"] as? [Int],let heads=packet["heads"] as? [String],
              let markers=packet["marker_ids"] as? [String:[Int]],heads.count==4,!ids.isEmpty,tokens>8,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),Set(heads.compactMap{markers[$0]?.count}).count==1
        else {throw EngineError.invalidPacket("chunk state regression packet")}
        let source=prefill(ids)
        func digest()->[String] {
            source.flatMap{$0.arrays}.map{SHA256.hash(data:$0.asData(access:.copy).data).map{String(format:"%02x",$0)}.joined()}
        }
        let initialDigest=digest()
        var cases=[[String:Any]]()
        for batch in 1...4 {
            let selected=Array(heads.prefix(batch)),length=markers[selected[0]]!.count
            // Exact capacity: the final pending token has not been consumed.
            let capacity=length+tokens-1
            var referenceTokens=[Int](),referenceState=[MLXArray]()
            for chunk in [1,2,4,8] {
                let cache=source.map{$0.fork(batch:batch,capacity:capacity)}
                let compiled=CompiledDecode(self,cache)
                var prediction:MLXArray!
                for i in 0..<length {
                    prediction=compiled.forward(MLXArray(selected.map{markers[$0]![i]},[batch,1]),project:i==length-1)
                    eval([prediction!]+cache.flatMap{$0.arrays})
                }
                var generated=0,raw=[Int]()
                while generated<tokens {
                    let n=min(chunk,tokens-generated)
                    let block=compiled.advancePending(prediction,count:n)
                    eval([block]+cache.flatMap{$0.arrays})
                    let values=block.asArray(Int32.self).map{Int($0)};raw.append(contentsOf:values)
                    generated+=n
                    if generated<tokens {prediction=compiled.forward(MLXArray(Array(values.suffix(batch)),[batch,1]))}
                }
                let state=cache.flatMap{$0.prefixK==nil ? [$0.conv!,$0.matrix!] : [$0.k!,$0.v!]}
                if chunk==1 {referenceTokens=raw;referenceState=state}
                let differences=zip(state,referenceState).enumerated().compactMap{index,pair -> Int? in
                    pair.0.shape==pair.1.shape && pair.0.dtype==pair.1.dtype &&
                    pair.0.asData(access:.copy).data==pair.1.asData(access:.copy).data ? nil : index
                }
                let equal=raw==referenceTokens && differences.isEmpty && compiled.used==capacity
                cases.append(["batch":batch,"chunk":chunk,"token_equal":raw==referenceTokens,
                    "state_bitwise_equal":differences.isEmpty,"different_state_indices":differences,
                    "used":compiled.used,"capacity":capacity,"passed":equal])
            }
        }
        let immutable=initialDigest==digest()
        return ["cases":cases,"source_digest_before":initialDigest,"source_unchanged":immutable,
            "all_passed":immutable && cases.allSatisfy{$0["passed"] as? Bool == true},
            "weight_bits":4,"compute":"float32","projection_policy":projectionPolicy.rawValue,
            "scope":"actual model B1..4, chunks1/2/4/8, 17 tokens with partial last block; exact KV capacity, full conv/matrix/tail bytes",
            "limitation":"fixed token/state equivalence; natural stops tested separately; not phone or multi-turn acceptance"]
    }
}
