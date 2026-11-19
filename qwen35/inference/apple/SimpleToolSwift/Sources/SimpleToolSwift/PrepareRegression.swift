import Foundation
import MLX

// Check against the pinned Swift core, rather than a different MLX release.
public func prepareRegression() throws -> [String:Any] {
    let bodies=try MetalBodies()
    let activate=compile(shapeless:true) {(x:MLXArray) in x*sigmoid(x)}
    var rows=[[String:Any]]()
    for (hk,hv) in [(16,32),(16,16),(2,4),(1,1),(3,6)] {
        let d=(2*hk+hv)*128
        for b in 1...4 { for magnitude:Float in [0,0.001,1,20] {for strided in [false,true] {
            func input(_ count:Int,_ salt:Int)->MLXArray {
                var values=[Float]();values.reserveCapacity(count)
                for i in 0..<count {
                    let integer=(i*37+salt*53)%2003-1001
                    values.append(magnitude*Float(integer)/Float(1001))
                }
                return MLXArray(values)
            }
            let history:MLXArray,qkv:MLXArray
            if strided {
                history=input(b*3*d*2,1).reshaped(b,3,d*2)[0...,0...,.stride(by:2)]
                qkv=input(b*d*2,2).reshaped(b,1,d*2)[0...,0...,.stride(by:2)]
            } else {
                history=input(b*3*d,1).reshaped(b,3,d)
                qkv=input(b*d,2).reshaped(b,1,d)
            }
            let weight=input(d*4,3).reshaped(d,4,1)
            let full=concatenated([history,qkv],axis:1)
            let cv=activate(conv1d(full,weight,groups:d))
            let chunks=cv.split(indices:[hk*128,2*hk*128],axis:-1)
            let q=Float(1.0/128.0)*MLXFast.rmsNorm(chunks[0].reshaped(b,1,hk,128),weight:.mlxNone,eps:1e-6/128)
            let k=Float(pow(128.0,-0.5))*MLXFast.rmsNorm(chunks[1].reshaped(b,1,hk,128),weight:.mlxNone,eps:1e-6/128)
            let gold=[q,k,chunks[2].reshaped(b,1,hv,128),contiguous(full[0...,(-3)...,0...])]
            let actual=bodies.prepare([history,qkv,weight],template:[("B",b),("D",d),("Hk",hk),("Hv",hv)],
                                     grid:((((b*(2*hk+hv))+3)/4)*128,1,1),threadGroup:(128,1,1),
                                     outputShapes:gold.map{$0.shape},outputDTypes:[.float32,.float32,.float32,.float32])
            eval(gold+actual)
            let exact=zip(gold,actual).map{$0.asData(access:.copy).data==$1.asData(access:.copy).data}
            let differences=zip(gold,actual).map{abs($0-$1).max().item(Float.self)}
            rows.append(["key_heads":hk,"value_heads":hv,"batch":b,"magnitude":magnitude,"strided":strided,
                         "bitwise_equal_q_k_v_history":exact,"max_abs_q_k_v_history":differences,"passed":exact.allSatisfy{$0}])
        }}}
    }
    return ["cases":rows,"all_passed":rows.allSatisfy{$0["passed"] as? Bool == true},
            "scope":"160 pinned Swift FP32 prepare operator cases; zeros/scales/strided input; four output byte comparisons"]
}
