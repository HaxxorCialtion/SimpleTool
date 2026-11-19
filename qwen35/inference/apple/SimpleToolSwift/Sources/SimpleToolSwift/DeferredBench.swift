import Foundation
import MLX

extension Int4Model {
    public func benchmarkDeferred(_ url:URL,tokens:Int=96,rounds:Int=5) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],
              let markers=p["marker_ids"] as? [String:[Int]],heads.count==4,!ids.isEmpty,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),Set(heads.compactMap{markers[$0]?.count}).count==1,
              tokens>0,rounds>=3 else {throw EngineError.invalidPacket("deferred benchmark packet")}
        let source=prefill(ids),intervals=[1,2,4,8,16]
        let models=Dictionary(uniqueKeysWithValues:intervals.map{($0,Int4Model(sharing:self,policy:projectionPolicy,deferredInterval:$0))})
        let shared=models.values.allSatisfy{m in weights.keys.allSatisfy{weights[$0]! === m.weights[$0]!}}
        var runners=[String:FixedRunner](),samples=[String:[Double]](),first=[String:Double](),reference=[String:[[Int]]](),parity=[String:Bool]()
        let keys=intervals.flatMap{["\($0).single","\($0).batch4"]}
        for interval in intervals {for batch in [1,4] {
            let key="\(interval).\(batch==1 ? "single" : "batch4")"
            runners[key]=FixedRunner(models[interval]!,source,Array(heads.prefix(batch)).map{markers[$0]!},tokens,.compiled)
            samples[key]=[]
        }}
        for key in keys {
            let (seconds,raw)=runners[key]!.run();first[key]=seconds
            let batch=key.components(separatedBy:".").last!
            if key.hasPrefix("1.") {reference[batch]=raw}
            parity[key]=raw==reference[batch]
        }
        for round in 0..<rounds {
            let shift=round%keys.count
            for key in Array(keys[shift...])+Array(keys[..<shift]) {
                let (seconds,raw)=runners[key]!.run();samples[key]!.append(seconds)
                parity[key]=parity[key]! && raw==reference[key.components(separatedBy:".").last!]
            }
        }
        let medians=samples.mapValues{$0.sorted()[rounds/2]}
        let efficiency=Dictionary(uniqueKeysWithValues:intervals.map{(String($0),medians["\($0).single"]!/medians["\($0).batch4"]!)})
        return ["median_s":medians,"samples_s":samples,"first_run_s":first,"token_equal_to_baseline":parity,
                "efficiency_single_over_four":efficiency,"fixed_tokens_each_head":tokens,
                "projection_policy":projectionPolicy.rawValue,"weight_bits":4,"compute":"float32",
                "weight_bytes":weightBytes,"prefill_count":1,"prefix_tokens":ids.count,"mlx_peak_bytes":Memory.peakMemory,
                "packed_tensor_objects_shared":shared,
                "scope":"one int4 weight set and one prefix; 10 interval/batch runners; rotating five warm rounds; compiled single-token decode",
                "limitation":"stops ignored; peak includes all runners; first-run not fair cold-start; competing load uncontrolled; no phone/power acceptance"]
    }
}
