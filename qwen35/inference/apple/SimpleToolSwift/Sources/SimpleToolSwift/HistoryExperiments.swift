import Foundation
import MLX

extension Int4Model {
    public func validateHistorySuite(_ url:URL) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let packets=p["packets"] as? [String],packets.count==5,
              let directory=p["output_directory"] as? String,let size=p["size"] as? String
        else {throw EngineError.invalidPacket("history suite manifest")}
        var rows=[[String:Any]]()
        for (policy,intervals) in [(HistoryPolicy.dynamic,[1,16]),(.fixed,[2,4,8,16]),(.fused,[2,4,8,16])] {
            for interval in intervals {
                let variant=Int4Model(sharing:self,policy:projectionPolicy,deferredInterval:interval,historyPolicy:policy)
                for chunk in [1,8] {for (i,packet) in packets.enumerated() {
                    let result=try variant.validatePacket(URL(fileURLWithPath:packet),execution:.compiled,decodeChunkSize:chunk)
                    let equal=result["gold_token_equal"] as! [String:[String:Bool]]
                    let text=result["decoded_gold_equal"] as! [String:[String:Bool]]
                    let passed=equal.values.allSatisfy{$0.values.allSatisfy{$0}} && text.values.allSatisfy{$0.values.allSatisfy{$0}}
                    let name="swift_fixed_history_\(size)_\(policy.rawValue)_i\(interval)_c\(chunk)_case\(i)_20261008"
                    let output=URL(fileURLWithPath:directory).appendingPathComponent(name+".json")
                    try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:output)
                    rows.append(["policy":policy.rawValue,"interval":interval,"chunk":chunk,"case":i,"passed":passed,"result":output.path])
                    print(name,passed)
                    if !passed {return ["cases":rows,"all_passed":false]}
                }}
            }
        }
        return ["cases":rows,"all_passed":rows.allSatisfy{$0["passed"] as? Bool == true},
                "weight_bits":4,"compute":"float32","scope":"shared int4 weights; per-request prefill1; real serial/B4 natural stops and ByteLevel text; separate state hash suite"]
    }
    public func historyStateSuite(_ url:URL) throws ->[String:Any] {
        var rows=[[String:Any]]()
        for policy in [HistoryPolicy.fixed,.fused] {
            let variant=Int4Model(sharing:self,policy:projectionPolicy,deferredInterval:16,historyPolicy:policy)
            let result=try variant.deferredStateRegression(url)
            rows.append(["policy":policy.rawValue,"result":result,"passed":result["all_passed"] as? Bool == true])
        }
        return ["cases":rows,"all_passed":rows.allSatisfy{$0["passed"] as? Bool == true},
                "weight_bits":4,"compute":"float32","scope":"two fixed history implementations; each32 canonical full-state cases; per-family source hash checked"]
    }
    public func probeHistoryRequest(_ url:URL) throws ->[String:Any] {
        var rows=[[String:Any]]()
        for request in 0..<2 {
            let start=Date(),result=try validatePacket(url,execution:.compiled,memoryProbe:true)
            let equal=result["gold_token_equal"] as! [String:[String:Bool]]
            let passed=equal.values.allSatisfy{$0.values.allSatisfy{$0}} && result["prefix_unchanged"] as? Bool == true
            rows.append(["request":request,"wall_s":Date().timeIntervalSince(start),"result":result,"passed":passed])
        }
        return ["requests":rows,"all_passed":rows.allSatisfy{$0["passed"] as? Bool == true},
                "policy":historyPolicy.rawValue,"interval":deferredInterval,"weight_bits":4,"compute":"float32",
                "scope":"fresh process, two natural B4 requests; each prefill1; decode_s excludes prefill; not cold GPU/disk/phone; MLX allocation peaks, not RSS"]
    }
    public func benchmarkHistory(_ url:URL,tokens:Int=96,rounds:Int=5) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],
              let markers=p["marker_ids"] as? [String:[Int]],heads.count==4,!ids.isEmpty,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),Set(heads.compactMap{markers[$0]?.count}).count==1,
              tokens>0,rounds>=3
        else {throw EngineError.invalidPacket("history benchmark packet")}
        let policies:[(String,Int,HistoryPolicy)]=[("baseline1",1,.dynamic),("dynamic4",4,.dynamic),("fixed4",4,.fixed),("fused4",4,.fused),
            ("dynamic16",16,.dynamic),("fixed16",16,.fixed),("fused16",16,.fused)]
        let source=prefill(ids)
        let models=Dictionary(uniqueKeysWithValues:policies.map{($0.0,Int4Model(sharing:self,policy:projectionPolicy,deferredInterval:$0.1,historyPolicy:$0.2))})
        let shared=models.values.allSatisfy{m in weights.keys.allSatisfy{weights[$0]! === m.weights[$0]!}}
        var runners=[String:FixedRunner](),samples=[String:[Double]](),first=[String:Double](),reference=[String:[[Int]]](),parity=[String:Bool]()
        let keys=policies.flatMap{["\($0.0).single","\($0.0).batch4"]}
        for policy in policies {for batch in [1,4] {
            let key="\(policy.0).\(batch==1 ? "single" : "batch4")"
            runners[key]=FixedRunner(models[policy.0]!,source,Array(heads.prefix(batch)).map{markers[$0]!},tokens,.compiled)
            samples[key]=[]
        }}
        for key in keys {
            let (seconds,raw)=runners[key]!.run();first[key]=seconds
            let batch=key.components(separatedBy:".").last!
            if key.hasPrefix("baseline1.") {reference[batch]=raw}
            parity[key]=raw==reference[batch]
        }
        for round in 0..<rounds {
            let shift=(3*round)%keys.count
            for key in Array(keys[shift...])+Array(keys[..<shift]) {
                let (seconds,raw)=runners[key]!.run();samples[key]!.append(seconds)
                parity[key]=parity[key]! && raw==reference[key.components(separatedBy:".").last!]
            }
        }
        let medians=samples.mapValues{$0.sorted()[rounds/2]}
        return ["median_s":medians,"samples_s":samples,"first_run_s":first,"token_equal_to_baseline":parity,
                "fixed_tokens_each_head":tokens,"packed_tensor_objects_shared":shared,"weight_bits":4,"compute":"float32",
                "prefill_count":1,"weight_bytes":weightBytes,"mlx_peak_bytes":Memory.peakMemory,
                "scope":"one int4 weight set and one prefix; 14 runners; five rotating warm rounds, compiled single-token",
                "limitation":"stops ignored; first-run paths share prewarm; peak includes all runners; external load uncontrolled; no phone/power acceptance"]
    }
}
