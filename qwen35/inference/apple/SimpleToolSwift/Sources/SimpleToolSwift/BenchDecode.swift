import Foundation
import MLX

final class FixedRunner {
    let model:Int4Model,execution:DecodeExecution,source:[LayerState]
    let markers:[[Int]],count:Int,initial:[LayerState],cache:[LayerState]
    let compiled:CompiledDecode?
    let decodeChunkSize:Int
    init(_ model:Int4Model,_ source:[LayerState],_ markers:[[Int]],_ count:Int,_ execution:DecodeExecution,decodeChunkSize:Int=1) {
        precondition([1,2,4,8].contains(decodeChunkSize) && (decodeChunkSize==1 || execution == .compiled))
        self.decodeChunkSize=decodeChunkSize
        self.model=model;self.source=source;self.markers=markers;self.count=count;self.execution=execution
        let capacity=count+markers[0].count
        func fork()->[LayerState] {source.map{execution == .dynamic && markers.count==1 ? $0.copy() :
            $0.fork(batch:markers.count,capacity:execution == .dynamic ? nil : capacity)}}
        initial=fork();cache=initial.map{$0.copy()}
        compiled=execution == .compiled ? CompiledDecode(model,cache) : nil
    }
    func reset() {
        if let compiled {compiled.reset(initial)}
        else {for (c,s) in zip(cache,initial) {
            c.conv=s.conv;c.matrix=s.matrix;c.k=s.k;c.v=s.v;c.tailUsed=s.tailUsed;c.offset=s.offset
        }}
        model.evaluate(cache)
    }
    func step(_ input:MLXArray,project:Bool=true)->MLXArray {
        let prediction:MLXArray
        if let compiled {prediction=compiled.forward(input,project:project)}
        else if project {prediction=model.logits(input,cache:cache).argMax(axis:-1).asType(.int32)}
        else {eval(model.hidden(input,cache:cache));prediction=MLXArray.zeros([input.dim(0)],dtype:.int32)}
        eval([prediction]+cache.flatMap{$0.arrays})
        return prediction
    }
    func run()->(Double,[[Int]]) {
        reset();let start=Date();var prediction:MLXArray!
        for i in markers[0].indices {
            prediction=step(MLXArray(markers.map{$0[i]},[markers.count,1]),project:i==markers[0].count-1)
        }
        var raw=markers.map{_ in [Int]()}
        var generated=0
        while generated<count {
            let n=min(decodeChunkSize,count-generated)
            let block=n>1 ? compiled!.advancePending(prediction,count:n) : prediction.reshaped(1,markers.count)
            eval([block]+cache.flatMap{$0.arrays})
            let chosen=block.asArray(Int32.self).map{Int($0)}
            for column in 0..<n {for row in markers.indices {raw[row].append(chosen[column*markers.count+row])}}
            generated+=n
            if generated<count {
                let input=MLXArray(Array(chosen.suffix(markers.count)),[markers.count,1])
                if let compiled {prediction=compiled.forward(input)}
                else {prediction=step(input)}
            }
        }
        return (Date().timeIntervalSince(start),raw)
    }
}

extension Int4Model {
    public func benchmarkChunks(_ url:URL,tokens:Int=96,rounds:Int=5) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],
              let markers=p["marker_ids"] as? [String:[Int]],heads.count==4,!ids.isEmpty,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),Set(heads.compactMap{markers[$0]?.count}).count==1,
              tokens>0,rounds>=3 else {throw EngineError.invalidPacket("chunk benchmark packet")}
        let source=prefill(ids),sizes=[1,2,4,8]
        var runners=[String:FixedRunner](),samples=[String:[Double]](),first=[String:Double](),reference=[String:[[Int]]](),parity=[String:Bool]()
        let keys=sizes.flatMap{["\($0).single","\($0).batch4"]}
        for n in sizes {for batch in [1,4] {
            let key="\(n).\(batch==1 ? "single" : "batch4")"
            runners[key]=FixedRunner(self,source,Array(heads.prefix(batch)).map{markers[$0]!},tokens,.compiled,decodeChunkSize:n)
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
        let efficiency=Dictionary(uniqueKeysWithValues:sizes.map{(String($0),medians["\($0).single"]!/medians["\($0).batch4"]!)})
        return ["median_s":medians,"samples_s":samples,"first_run_s":first,"token_equal_to_chunk1":parity,
                "efficiency_single_over_four":efficiency,"fixed_tokens_each_head":tokens,
                "projection_policy":projectionPolicy.rawValue,"weight_bits":4,"compute":"float32",
                "weight_bytes":weightBytes,"prefill_count":1,"prefix_tokens":ids.count,"mlx_peak_bytes":Memory.peakMemory,
                "scope":"compiled GPU autoregressive chunks1/2/4/8; eight runners, one source; rotating warm rounds",
                "limitation":"stops ignored; first-run costs not cold-device comparisons; all runner states coexist; no phone/power acceptance"]
    }
    public func benchmarkProjections(_ url:URL,tokens:Int=96,rounds:Int=5) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],
              let markers=p["marker_ids"] as? [String:[Int]],heads.count==4,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),!ids.isEmpty,tokens>0,rounds>=3,
              Set(heads.compactMap{markers[$0]?.count}).count==1 else {throw EngineError.invalidPacket("projection benchmark packet")}
        let source=prefill(ids)
        let policies=ProjectionPolicy.allCases
        let models=Dictionary(uniqueKeysWithValues:policies.map{($0.rawValue,Int4Model(sharing:self,policy:$0))})
        let shared=models.values.allSatisfy{variant in weights.keys.allSatisfy{weights[$0]! === variant.weights[$0]!}}
        var runners=[String:FixedRunner](),samples=[String:[Double]](),first=[String:Double](),reference=[String:[[Int]]](),parity=[String:Bool]()
        let keys=policies.flatMap{["\($0.rawValue).single","\($0.rawValue).batch4"]}
        for policy in policies {for batch in [1,4] {
            let key="\(policy.rawValue).\(batch==1 ? "single" : "batch4")"
            runners[key]=FixedRunner(models[policy.rawValue]!,source,Array(heads.prefix(batch)).map{markers[$0]!},tokens,.compiled)
            samples[key]=[]
        }}
        for key in keys {
            let (seconds,raw)=runners[key]!.run();first[key]=seconds
            let batch=key.components(separatedBy:".").last!
            if key.hasPrefix("baseline.") {reference[batch]=raw}
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
        let efficiency=Dictionary(uniqueKeysWithValues:policies.map{($0.rawValue,medians[$0.rawValue+".single"]!/medians[$0.rawValue+".batch4"]!)})
        return ["median_s":medians,"samples_s":samples,"first_run_s":first,
                "token_equal_to_baseline":parity,"packed_tensor_objects_shared":shared,
                "efficiency_single_over_four":efficiency,"fixed_tokens_each_head":tokens,
                "weight_bits":4,"compute":"float32","weight_bytes":weightBytes,
                "prefill_count":1,"prefix_tokens":ids.count,"mlx_peak_bytes":Memory.peakMemory,
                "scope":"compiled fixed capacity; eight runners, common weights and prefix; five rotating warm rounds",
                "limitation":"stops ignored, external workload uncontrolled; peak includes all variants; no phone/energy acceptance"]
    }
    public func benchmarkPacket(_ url:URL,tokens:Int=96,rounds:Int=5) throws ->[String:Any] {
        guard let p=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any],
              let ids=p["prompt_ids"] as? [Int],let heads=p["heads"] as? [String],
              let markers=p["marker_ids"] as? [String:[Int]],heads.count==4,
              heads.allSatisfy({markers[$0]?.isEmpty==false}),!ids.isEmpty,tokens>0,rounds>=3,
              Set(heads.compactMap{markers[$0]?.count}).count==1 else {throw EngineError.invalidPacket("benchmark packet")}
        let source=prefill(ids)
        var runners=[String:FixedRunner](),samples=[String:[Double]](),cold=[String:Double](),reference=[String:[[Int]]](),parity=[String:Bool]()
        let executions:[DecodeExecution]=[.dynamic,.staticCache,.compiled]
        let keys=executions.flatMap{e in ["\(e.rawValue).single","\(e.rawValue).batch4"]}
        for e in executions {for batch in [1,4] {
            let key="\(e.rawValue).\(batch==1 ? "single" : "batch4")"
            runners[key]=FixedRunner(self,source,Array(heads.prefix(batch)).map{markers[$0]!},tokens,e)
            samples[key]=[]
        }}
        // Warm complete paths once; record first-run compile cost separately.
        for key in keys {
            let (seconds,raw)=runners[key]!.run();cold[key]=seconds
            let batch=key.components(separatedBy:".").last!
            if key.hasPrefix("dynamic.") {reference[batch]=raw}
            parity[key]=raw==reference[batch]
        }
        for round in 0..<rounds {
            let shift=round%keys.count
            let order=Array(keys[shift...])+Array(keys[..<shift])
            for key in order {
                let (seconds,raw)=runners[key]!.run();samples[key]!.append(seconds)
                parity[key]=parity[key]! && raw==reference[key.components(separatedBy:".").last!]
            }
        }
        let medians=samples.mapValues{v in v.sorted()[v.count/2]}
        var efficiency=[String:Double]()
        for e in executions {efficiency[e.rawValue]=medians[e.rawValue+".single"]!/medians[e.rawValue+".batch4"]!}
        return ["samples_s":samples,"median_s":medians,"first_run_s":cold,"fixed_tokens_each_head":tokens,
                "token_equal_to_dynamic":parity,"efficiency_single_over_four":efficiency,
                "weight_bits":4,"compute":"float32","weight_bytes":weightBytes,
                "prefix_tokens":ids.count,"prefill_count":1,"mlx_peak_bytes":Memory.peakMemory,
                "scope":"warm fixed-count marker+decode; stops ignored, caches reset; rotating shared-process comparisons",
                "limitation":"source and six runners coexist; peak is not single-model deployment peak; no phone or power measurement"]
    }
}
