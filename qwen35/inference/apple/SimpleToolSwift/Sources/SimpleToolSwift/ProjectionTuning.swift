import Foundation
import MLX

extension Int4Model {
    /// Actual packed weights with deterministic authored activations. These
    /// isolated wall times include GPU synchronization; select with E2E next.
    func projectionTileProbe(batch:Int)->[[String:Any]] {
        precondition((1...4).contains(batch))
        let ops:[(String,String,Int)] = [
            ("gate_up","layers.0.mlp.gate_proj",2),
            ("down","layers.0.mlp.down_proj",0),
            ("vocab","embed_tokens",batch==4 ? 0:1),
            ("gdn_qkv","layers.0.linear_attn.in_proj_qkv",1),
            ("gdn_out","layers.0.linear_attn.out_proj",1)]
        var rows=[[String:Any]]()
        for (label,path,kind) in ops {
            let weight=w(path+".weight"),scale=w(path+".scales"),bias=w(path+".biases")
            let n=weight.dim(0),k=weight.dim(1)*8
            guard k%512==0 else {continue}
            let x=(sin(MLXArray(0..<(batch*k)).asType(.float32)*0.173)*0.13).asType(precision.dtype).reshaped(batch,1,k)
            var args=[x,weight,scale,bias]
            let fn:MLXFast.MLXFastKernel
            if kind==2 {
                let up="layers.0.mlp.up_proj"
                args += [w(up+".weight"),w(up+".scales"),w(up+".biases")]
                fn=bodies.swiglu16
            } else {fn=kind==0 ? bodies.row16:bodies.stream16}
            var reference:MLXArray?
            for tile in [1,2,4,8] {
                for threads in [64,128,256] {
                    if kind==1 && tile==8 {continue}
                    let groupRows=tile*threads/32
                    var times=[Double](),error:Float=0
                    for trial in -1..<3 {
                        let begin=ProcessInfo.processInfo.systemUptime
                        let result=fn(args,template:[("M",batch),("N",n),("K",k),("ROWS",tile)],
                                      grid:(((n+groupRows-1)/groupRows)*threads,1,1),threadGroup:(threads,1,1),
                                      outputShapes:[[batch,1,n]],outputDTypes:[precision.dtype])[0]
                        eval(result);Stream.gpu.synchronize()
                        let elapsed=ProcessInfo.processInfo.systemUptime-begin
                        if trial>=0 {times.append(elapsed)}
                        if reference==nil {reference=result}
                        if trial==0 {error=abs(result-reference!).max().item(Float.self)}
                    }
                    rows.append(["op":label,"kind":kind,"batch":batch,"n":n,"k":k,"rows":tile,"threads":threads,"seconds":times,"max_abs_difference":error])
                }
            }
        }
        return rows
    }
}
