import Foundation
import MLX

/// Packed affine INT8 cache, group64; functional updates preserve borrowed prefixes.
struct Q8KV {
    var q:MLXArray,scale:MLXArray,bias:MLXArray
    var length:Int
    var arrays:[MLXArray] {[q,scale,bias]}
    static func empty(batch:Int,heads:Int,capacity:Int,dim:Int,dtype:DType)->Q8KV {
        let n=max(64,((capacity+63)/64)*64)
        return Q8KV(q:.zeros([batch,heads,n,dim/4],dtype:.uint32),
                    scale:.zeros([batch,heads,n,dim/64],dtype:dtype),
                    bias:.zeros([batch,heads,n,dim/64],dtype:dtype),length:0)
    }
    mutating func append(_ value:MLXArray) {
        let end=length+value.dim(2)
        if end>q.dim(2) {
            let count=((end-q.dim(2)+63)/64)*64
            func grow(_ a:MLXArray)->MLXArray {concatenated([a,.zeros([a.dim(0),a.dim(1),count,a.dim(3)],dtype:a.dtype)],axis:2)}
            q=grow(q);scale=grow(scale);bias=grow(bias)
        }
        let packed=quantized(value,groupSize:64,bits:8)
        let offset=MLXArray(Int32(length))
        q=updateTail(q,packed.wq,offset);scale=updateTail(scale,packed.scales,offset);bias=updateTail(bias,packed.biases!,offset)
        length=end
    }
    mutating func filter(_ indices:MLXArray) {
        q=q.take(indices,axis:0);scale=scale.take(indices,axis:0);bias=bias.take(indices,axis:0)
    }
    func product(_ x:MLXArray,transpose:Bool)->MLXArray {
        quantizedMM(x,q.expandedDimensions(axis:2),scales:scale.expandedDimensions(axis:2),
                    biases:bias.expandedDimensions(axis:2),transpose:transpose,groupSize:64,bits:8)
    }
}

extension Int4Model {
    func q8Attention(_ q:MLXArray,_ k:MLXArray,_ v:MLXArray,_ cache:LayerState)->MLXArray {
        let b=q.dim(0),h=q.dim(1),s=q.dim(2),d=q.dim(3),hk=k.dim(1),r=h/hk
        if cache.k8 == nil {
            cache.k8=Q8KV.empty(batch:b,heads:hk,capacity:k.dim(2),dim:d,dtype:k.dtype)
            cache.v8=Q8KV.empty(batch:b,heads:hk,capacity:v.dim(2),dim:d,dtype:v.dtype)
        }
        let prior=cache.k8!.length
        cache.k8!.append(k);cache.v8!.append(v)
        let query=q.reshaped(b,hk,r,s,d)*Float(pow(Double(d),-0.5))
        let tail=cache.k8!,values=cache.v8!
        if decodeFusedQ8,s==1,let pk=cache.prefixK8,let pv=cache.prefixV8 {
            let template:[(String,any KernelTemplateArg)]=[("B",b),("H",h),("HK",hk),("D",d),("PN",pk.length),("TN",tail.length),("PC",pk.q.dim(2)),("TC",tail.q.dim(2))]
            let scores=bodies.q8Scores([query,pk.q,pk.scale,pk.bias,tail.q,tail.scale,tail.bias],template:template,
                grid:((((pk.length+tail.length)+3)/4)*128,b*h,1),threadGroup:(128,1,1),
                outputShapes:[[b,h,1,pk.length+tail.length]],outputDTypes:[q.dtype])[0]
            let probs=softmax(scores,axis:-1,precise:true)
            return bodies.q8Values([probs,pv.q,pv.scale,pv.bias,values.q,values.scale,values.bias],template:template,
                grid:(((d+31)/32)*128,b*h,1),threadGroup:(128,1,1),outputShapes:[[b,h,1,d]],outputDTypes:[q.dtype])[0]
        }
        let tailScores=tail.product(query,transpose:true)
        let positions=MLXArray(0..<tail.q.dim(2))
        let valid=(positions .< tail.length) .&& (positions .<= (MLXArray(0..<s)+prior).expandedDimensions(axis:-1))
        let maskedTail=which(valid,tailScores,-Float.infinity)
        let result:MLXArray
        if let pk=cache.prefixK8,let pv=cache.prefixV8 {
            precondition(s==1 && pk.q.dim(0)==1)
            let prefix=which(MLXArray(0..<pk.q.dim(2)) .< pk.length,pk.product(query,transpose:true),-Float.infinity)
            let p=softmax(concatenated([prefix,maskedTail],axis:-1),axis:-1,precise:true)
            let split=pk.q.dim(2)
            result=pv.product(p[.ellipsis,..<split],transpose:false)+values.product(p[.ellipsis,split...],transpose:false)
        } else {
            result=values.product(softmax(maskedTail,axis:-1,precise:true),transpose:false)
        }
        return result.reshaped(b,h,s,d)
    }
}
