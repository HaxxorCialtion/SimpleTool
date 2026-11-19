import Foundation
import MLX
public func ropeRegression()->[String:Any] {
    let x=(MLXArray(Array(0..<2048)).asType(.float32)/2048).reshaped(1,8,1,256)
    var cases=[[String:Any]]()
    for b in 1...4 {
        for offset in [0,20,37] {
            let input=contiguous(broadcast(x,to:[b,8,1,256]))
            let reference=broadcast(MLXFast.RoPE(x,dimensions:64,traditional:false,base:10000000,scale:1,offset:offset),to:input.shape)
            let corrected=MLXFast.RoPE(input,dimensions:64,traditional:false,base:10000000,scale:1,
                offset:MLXArray(Array(repeating:Int32(offset),count:b)))
            let legacy=MLXFast.RoPE(input,dimensions:64,traditional:false,base:10000000,scale:1,offset:offset)
            let error=abs(corrected-reference).max().item(Float.self)
            let oldError=abs(legacy-reference).max().item(Float.self)
            cases.append(["batch":b,"offset":offset,"corrected_max_abs":error,"legacy_max_abs":oldError,"passed":error<2e-6])
        }
    }
    return ["cases":cases,"all_passed":cases.allSatisfy{$0["passed"] as? Bool == true},
            "scope":"pinned MLX Swift 0.31.1; contiguous B1..4,T1,D256,partial RoPE64"]
}
