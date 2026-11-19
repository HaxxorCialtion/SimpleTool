import Foundation
import MLX

public func staticTailRegression()->[String:Any] {
    let compiled=compile(shapeless:false) { (args:[MLXArray])->[MLXArray] in [updateTail(args[0],args[1],args[2])] }
    var cases=[[String:Any]]()
    for batch in 1...4 {for capacity in [3,17] {for mode in ["raw","compiled"] {
        let shape=[batch,2,capacity,256]
        let seed=MLXArray.full(shape,values:MLXArray(Float(-999)))
        var state=seed;let expected=MLXArray.full(shape,values:MLXArray(Float(-999)))
        var passed=true
        for offset in 0..<capacity {
            let value=(MLXArray(0..<(batch*2*256)).asType(.float32)+Float(offset*2048)).reshaped(batch,2,1,256)
            let index=MLXArray(Int32(offset))
            state=mode=="raw" ? updateTail(state,value,index) : compiled([state,value,index])[0]
            expected[0...,0...,offset..<(offset+1),0...]=value
            eval([state,expected])
            passed=passed && arrayEqual(state,expected).item(Bool.self)
        }
        let immutable=arrayEqual(seed,MLXArray.full(shape,values:MLXArray(Float(-999)))).item(Bool.self)
        cases.append(["batch":batch,"capacity":capacity,"mode":mode,"passed":passed && immutable,"seed_unchanged":immutable])
    }}}
    return ["cases":cases,"all_passed":cases.allSatisfy{$0["passed"] as? Bool == true},
            "scope":"dynamic MLX-C slice update; every slot including last; immutable input; B1..4,Hkv2,D256"]
}
