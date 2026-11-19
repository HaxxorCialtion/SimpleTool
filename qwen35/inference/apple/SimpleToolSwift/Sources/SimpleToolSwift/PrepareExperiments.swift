import Foundation
import CryptoKit
import MLX

extension Int4Model {
    private func prepareVariants()->[(String,Int4Model)] {
        [("baseline",Int4Model(sharing:self,policy:.baseline,deferredInterval:1,gdnPreparePolicy:.baseline)),
         ("prepare",Int4Model(sharing:self,policy:.baseline,deferredInterval:1,gdnPreparePolicy:.fused)),
         ("projections",Int4Model(sharing:self,policy:.both,deferredInterval:1,gdnPreparePolicy:.baseline)),
         ("combined",Int4Model(sharing:self,policy:.both,deferredInterval:1,gdnPreparePolicy:.fused)),
         ("history16",Int4Model(sharing:self,policy:.baseline,deferredInterval:16,historyPolicy:.fused,gdnPreparePolicy:.fused)),
         ("combined16",Int4Model(sharing:self,policy:.both,deferredInterval:16,historyPolicy:.fused,gdnPreparePolicy:.fused))]
    }
    private func prepareManifest(_ url:URL) throws ->(packets:[URL],directory:URL,size:String) {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let packets=p["packets"] as? [String],packets.count==5,
              let directory=p["output_directory"] as? String,let size=p["size"] as? String
        else {throw EngineError.invalidPacket("prepare suite manifest")}
        return (packets.map{URL(fileURLWithPath:$0)},URL(fileURLWithPath:directory),size)
    }
    private func prepareDigest(_ source:[LayerState])->[String] {
        source.flatMap{$0.arrays}.map{SHA256.hash(data:$0.asData(access:.copy).data).map{String(format:"%02x",$0)}.joined()}
    }
    public func validatePrepareSuite(_ url:URL) throws ->[String:Any] {
        let manifest=try prepareManifest(url),variants=prepareVariants()
        let shared=variants.allSatisfy{_,model in weights.keys.allSatisfy{weights[$0]! === model.weights[$0]!}}
        var rows=[[String:Any]]()
        for (i,packet) in manifest.packets.enumerated() {
            let p=try JSONSerialization.jsonObject(with:Data(contentsOf:packet)) as! [String:Any]
            let source=prefill(p["prompt_ids"] as! [Int]),before=prepareDigest(source)
            for (name,model) in variants {for chunk in [1,8] {
                let result=try model.validatePacket(packet,execution:.compiled,decodeChunkSize:chunk,prefixCache:source,checkPrefix:true)
                let equal=result["gold_token_equal"] as! [String:[String:Bool]]
                let text=result["decoded_gold_equal"] as! [String:[String:Bool]]
                let passed=equal.count==2 && equal.values.allSatisfy{$0.count==4 && $0.values.allSatisfy{$0}}
                    && text.count==2 && text.values.allSatisfy{$0.count==4 && $0.values.allSatisfy{$0}}
                    && result["prefix_unchanged"] as? Bool == true
                let stem="swift_prepare_\(manifest.size)_\(name)_c\(chunk)_case\(i)_20261008"
                let output=manifest.directory.appendingPathComponent(stem+".json")
                try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:output)
                rows.append(["variant":name,"chunk":chunk,"case":i,"passed":passed,"result":output.path])
                print(stem,passed);fflush(stdout)
                if !passed {return ["cases":rows,"all_passed":false]}
            }}
            precondition(before==prepareDigest(source),"source mutation")
        }
        return ["cases":rows,"all_passed":shared && rows.allSatisfy{$0["passed"] as? Bool == true},
                "packed_tensor_objects_shared":shared,"prefill_count":5,"weight_bits":4,"compute":"float32",
                "scope":"five existing real inputs; one prefill per input shared by six variants/chunks; serial and B4 raw tokens/text; source SHA checked"]
    }
    public func prepareStateRegression(_ url:URL,tokens:Int=19) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],let markers=p["marker_ids"] as? [String:[Int]],
              heads.count==4,!ids.isEmpty,Set(heads.compactMap{markers[$0]?.count}).count==1,
              heads.allSatisfy({markers[$0]?.isEmpty==false}) else {throw EngineError.invalidPacket("prepare state packet")}
        let source=prefill(ids),before=prepareDigest(source),variants=prepareVariants()
        var rows=[[String:Any]]()
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
                    raw.append(contentsOf:values);generated+=n
                    if generated<tokens {prediction=compiled.forward(MLXArray(Array(values.suffix(batch)),[batch,1]))}
                }
                let canonical=model.canonicalState(cache);eval(canonical)
                return (raw,canonical,compiled.used)
            }
            let reference=run(variants[0].1,1)
            let projectionReference=run(variants.first{$0.0=="projections"}!.1,1)
            for (name,model) in variants {for chunk in [1,8] {
                let candidate=run(model,chunk)
                let stateReference=model.projectionPolicy == .both ? projectionReference : reference
                let sameCount=stateReference.1.count==candidate.1.count
                let differences=zip(stateReference.1,candidate.1).enumerated().compactMap{index,pair -> Int? in
                    pair.0.shape==pair.1.shape && pair.0.dtype==pair.1.dtype &&
                    pair.0.asData(access:.copy).data==pair.1.asData(access:.copy).data ? nil : index
                }
                let sameState=sameCount && differences.isEmpty
                let baselineDifferences=zip(reference.1,candidate.1).enumerated().compactMap{index,pair -> Int? in
                    pair.0.shape==pair.1.shape && pair.0.dtype==pair.1.dtype &&
                    pair.0.asData(access:.copy).data==pair.1.asData(access:.copy).data ? nil : index
                }
                let sameTokens=reference.0==candidate.0,passed=sameTokens && sameState && candidate.2==capacity
                rows.append(["batch":batch,"variant":name,"chunk":chunk,"passed":passed,"token_equal":sameTokens,
                             "state_bitwise_equal":sameState,"state_array_count_equal":sameCount,"state_array_count":candidate.1.count,
                             "state_reference_projection_policy":model.projectionPolicy.rawValue,
                             "state_bitwise_equal_to_baseline":reference.1.count==candidate.1.count && baselineDifferences.isEmpty,
                             "different_baseline_state_indices":baselineDifferences,
                             "different_state_indices":differences,"used":candidate.2,"capacity":capacity])
                print("prepare state",batch,name,chunk,passed);fflush(stdout)
            }}
        }
        let immutable=before==prepareDigest(source)
        return ["cases":rows,"all_passed":immutable && rows.allSatisfy{$0["passed"] as? Bool == true},"prefix_unchanged":immutable,
                "prefill_count":1,"weight_bits":4,"compute":"float32",
                "scope":"B1..4, six variants, chunks1/8; actual complete markers and 19 tokens; token reference baseline; all canonical state arrays byte comparison against unfused prepare/dense history with matching projection policy; baseline state differences recorded separately; one source SHA"]
    }
    public func benchmarkPrepare(_ url:URL,tokens:Int=96,rounds:Int=5) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],let markers=p["marker_ids"] as? [String:[Int]],
              heads.count==4,!ids.isEmpty,heads.allSatisfy({markers[$0]?.isEmpty==false}),
              Set(heads.compactMap{markers[$0]?.count}).count==1,rounds>=3 else {throw EngineError.invalidPacket("prepare benchmark packet")}
        let source=prefill(ids),variants=prepareVariants()
        let shared=variants.allSatisfy{_,model in weights.keys.allSatisfy{weights[$0]! === model.weights[$0]!}}
        let keys=variants.flatMap{["\($0.0).single","\($0.0).batch4"]}
        var runners=[String:FixedRunner](),samples=[String:[Double]](),first=[String:Double](),reference=[String:[[Int]]](),parity=[String:Bool]()
        for (name,model) in variants {for B in [1,4] {
            let key="\(name).\(B==1 ? "single" : "batch4")"
            runners[key]=FixedRunner(model,source,Array(heads.prefix(B)).map{markers[$0]!},tokens,.compiled)
            samples[key]=[]
        }}
        for key in keys {
            let (seconds,raw)=runners[key]!.run();first[key]=seconds
            let B=key.components(separatedBy:".").last!
            if key.hasPrefix("baseline.") {reference[B]=raw}
            parity[key]=raw==reference[B]
        }
        for round in 0..<rounds {
            let shift=(5*round)%keys.count
            for key in Array(keys[shift...])+Array(keys[..<shift]) {
                let (seconds,raw)=runners[key]!.run();samples[key]!.append(seconds)
                parity[key]=parity[key]! && raw==reference[key.components(separatedBy:".").last!]
            }
        }
        return ["median_s":samples.mapValues{$0.sorted()[rounds/2]},"samples_s":samples,"first_run_s":first,
                "token_equal_to_baseline":parity,"packed_tensor_objects_shared":shared,"prefill_count":1,
                "weight_bits":4,"compute":"float32","fixed_tokens_each_head":tokens,
                "scope":"12 compiled single/B4 runners, six variants, one prefix and weight set; 5 rotating warm rounds, stops ignored",
                "limits":"sequential prewarm; external load uncontrolled; all states coexist, no deployment RAM, GPU-only, phone or power claim"]
    }
    public func benchmarkPrepareRequests(_ url:URL,rounds:Int=5) throws ->[String:Any] {
        let manifest=try prepareManifest(url),variants=prepareVariants(),keys=variants.map{$0.0}
        var rows=[[String:Any]]()
        for (i,packet) in manifest.packets.enumerated() {
            let p=try JSONSerialization.jsonObject(with:Data(contentsOf:packet)) as! [String:Any]
            let source=prefill(p["prompt_ids"] as! [Int]),before=prepareDigest(source)
            var samples=Dictionary(uniqueKeysWithValues:keys.map{($0,[Double]())}),first=[String:Double]()
            func run(_ key:String) throws ->Double {
                let model=variants.first{$0.0==key}!.1
                let result=try model.validatePacket(packet,execution:.compiled,prefixCache:source,batchOnly:true,checkPrefix:true)
                let equal=result["gold_token_equal"] as! [String:[String:Bool]]
                guard equal["batch4"]?.count==4,equal["batch4"]!.values.allSatisfy({$0}),result["prefix_unchanged"] as? Bool == true
                else {throw EngineError.invalidPacket("prepare request gold/source mismatch")}
                return (result["decode_s"] as! [String:Double])["batch4"]!
            }
            for key in keys {first[key]=try run(key)}
            for round in 0..<rounds {
                let shift=(round+i)%keys.count
                for key in Array(keys[shift...])+Array(keys[..<shift]) {samples[key]!.append(try run(key))}
            }
            precondition(before==prepareDigest(source),"source mutation")
            rows.append(["case":i,"median_s":samples.mapValues{$0.sorted()[rounds/2]},"samples_s":samples,"first_run_s":first,"gold_and_source_equal":true])
            print("prepare request",i,rows.last!["median_s"]!);fflush(stdout)
        }
        return ["cases":rows,"all_passed":true,"prefill_count":5,"weight_bits":4,"compute":"float32",
                "packed_tensor_objects_shared":variants.allSatisfy{_,model in weights.keys.allSatisfy{weights[$0]! === model.weights[$0]!}},
                "scope":"five inputs, one prefill per input shared by six variants; first+5 rotating warm natural B4 requests; elapsed decode window includes fork/markers/stop/filter",
                "limits":"prefix hash/JSON/prefill outside decode_s; external load uncontrolled; no phone, RAM or cold startup claim"]
    }
}
