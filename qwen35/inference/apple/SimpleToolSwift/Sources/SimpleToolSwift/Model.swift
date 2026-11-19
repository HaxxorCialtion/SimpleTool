// Dense Qwen3.5 over unchanged affine-int4/group64 tensors. Precision is explicit;
// A16 uses FP16 projections and activations with FP32 recurrent GDN state/gates.
// Arithmetic contract follows the project's validated Python model; packed
// GDN arithmetic is Apple MIT (see bundled provenance and third_party license).
import Foundation
import MLX

public enum ProjectionPolicy:String,CaseIterable {case baseline,stream,fused,both
    var usesStream:Bool {self == .stream || self == .both}
    var usesFused:Bool {self == .fused || self == .both}
}
/// Opt-in diagnostic mixed precision. Packed INT4 and projection I/O stay unchanged.
public enum NumericalPolicy:String {case standard,residual,gdn,stable
    var fp32Residual:Bool {self == .residual || self == .stable}
    var fp32GDN:Bool {self == .gdn || self == .stable}
}
public enum HistoryPolicy:String {case dynamic,fixed,fused}
public enum GDNPreparePolicy:String {case baseline,fused}
public enum InferencePrecision:String {case fp32="w4a32",a16="w4a16",a16kv8="w4a16-kv8"
    var dtype:DType {self == .fp32 ? .float32:.float16}
}

public enum EngineError:Error { case invalidConfiguration(String),missingTensor(String),invalidPacket(String) }

public struct EngineConfig {
    let layers:Int,hidden:Int,heads:Int,kvHeads:Int,headDim:Int,interval:Int
    let keyHeads:Int,valueHeads:Int,keyDim:Int,valueDim:Int,convKernel:Int
    let epsilon:Float,ropeTheta:Float,ropeDims:Int
    init(_ url:URL) throws {
        guard let root=try JSONSerialization.jsonObject(with:Data(contentsOf:url)) as? [String:Any]
        else {throw EngineError.invalidConfiguration("config must be an object")}
        let c=(root["text_config"] as? [String:Any]) ?? root
        func int(_ key:String,_ fallback:Int=0)->Int { (c[key] as? NSNumber)?.intValue ?? fallback }
        func float(_ key:String,_ fallback:Float)->Float { (c[key] as? NSNumber)?.floatValue ?? fallback }
        let q=(root["quantization"] as? [String:Any]) ?? [:]
        guard (q["bits"] as? Int)==4,(q["group_size"] as? Int)==64,(q["mode"] as? String ?? "affine")=="affine",
              (c["tie_word_embeddings"] as? Bool)==true,int("num_experts")==0
        else { throw EngineError.invalidConfiguration("dense tied int4/group64 checkpoint required") }
        layers=int("num_hidden_layers");hidden=int("hidden_size");heads=int("num_attention_heads")
        kvHeads=int("num_key_value_heads");headDim=int("head_dim",hidden/max(1,heads))
        interval=int("full_attention_interval",4);keyHeads=int("linear_num_key_heads")
        valueHeads=int("linear_num_value_heads");keyDim=int("linear_key_head_dim")
        valueDim=int("linear_value_head_dim");convKernel=int("linear_conv_kernel_dim",4)
        epsilon=float("rms_norm_eps",1e-6)
        let r=c["rope_parameters"] as? [String:Any] ?? [:]
        ropeTheta=(r["rope_theta"] as? NSNumber)?.floatValue ?? 100000
        ropeDims=Int(Float(headDim)*((r["partial_rotary_factor"] as? NSNumber)?.floatValue ?? 0.25))
        guard layers>0,hidden>0,heads>0,kvHeads>0,interval>0,keyDim==128,valueDim%8==0,
              valueHeads>0,keyHeads>0,valueHeads%keyHeads==0,convKernel==4
        else { throw EngineError.invalidConfiguration("unsupported dense GDN shape") }
    }
}

public final class LayerState {
    var k8:Q8KV?,v8:Q8KV?,prefixK8:Q8KV?,prefixV8:Q8KV?
    var conv:MLXArray?,matrix:MLXArray?,k:MLXArray?,v:MLXArray?
    var deferredKeys:MLXArray?,deferredGates:MLXArray?,deferredDeltas:MLXArray?
    var historyPhase:MLXArray?
    var writeHistory=false
    var prefixK:MLXArray?,prefixV:MLXArray?
    var tailUsed:MLXArray?
    var offset=0
    func copy()->LayerState {
        let n=LayerState();n.conv=conv;n.matrix=matrix;n.k=k;n.v=v
        n.prefixK=prefixK;n.prefixV=prefixV;n.tailUsed=tailUsed;n.offset=offset
        n.deferredKeys=deferredKeys;n.deferredGates=deferredGates;n.deferredDeltas=deferredDeltas
        n.historyPhase=historyPhase;n.writeHistory=writeHistory
        n.k8=k8;n.v8=v8;n.prefixK8=prefixK8;n.prefixV8=prefixV8;return n
    }
    func fork(batch:Int,capacity:Int?=nil)->LayerState {
        let n=LayerState();n.offset=offset
        if let conv { n.conv=broadcast(conv,to:[batch]+Array(conv.shape.dropFirst())) }
        n.matrix=matrix // immutable batch1 seed; first recurrence writes batchB
        n.prefixK=k;n.prefixV=v
        if let k8,let v8 {
            n.prefixK8=k8;n.prefixV8=v8
            n.k8=Q8KV.empty(batch:batch,heads:k8.q.dim(1),capacity:capacity ?? 64,dim:k8.q.dim(3)*4,dtype:k8.scale.dtype)
            n.v8=Q8KV.empty(batch:batch,heads:v8.q.dim(1),capacity:capacity ?? 64,dim:v8.q.dim(3)*4,dtype:v8.scale.dtype)
        }
        if let capacity,let k,let v {
            precondition(capacity>0)
            n.k=MLXArray.zeros([batch,k.dim(1),capacity,k.dim(3)],dtype:k.dtype)
            n.v=MLXArray.zeros([batch,v.dim(1),capacity,v.dim(3)],dtype:v.dtype)
            n.tailUsed=MLXArray(Int32(0))
        }
        return n
    }
    func filter(_ indices:MLXArray) {
        conv=conv?.take(indices,axis:0)
        if deferredKeys == nil || matrix?.dim(0) != 1 {matrix=matrix?.take(indices,axis:0)}
        deferredKeys=deferredKeys?.take(indices,axis:0);deferredGates=deferredGates?.take(indices,axis:0)
        deferredDeltas=deferredDeltas?.take(indices,axis:0)
        k=k?.take(indices,axis:0);v=v?.take(indices,axis:0)
        k8?.filter(indices);v8?.filter(indices)
    }
    public var arrays:[MLXArray] { [conv,matrix,deferredKeys,deferredGates,deferredDeltas,historyPhase,k,v,prefixK,prefixV,tailUsed].compactMap{$0} + [k8,v8,prefixK8,prefixV8].compactMap{$0}.flatMap{$0.arrays} }
}

final class MetalBodies {
    let q8Scores:MLXFast.MLXFastKernel,q8Values:MLXFast.MLXFastKernel
    let row16:MLXFast.MLXFastKernel,qmv16:MLXFast.MLXFastKernel,stream16:MLXFast.MLXFastKernel,swiglu16:MLXFast.MLXFastKernel
    let row:MLXFast.MLXFastKernel,qmv:MLXFast.MLXFastKernel
    let gdn:MLXFast.MLXFastKernel,shared:MLXFast.MLXFastKernel
    let stream:MLXFast.MLXFastKernel,swiglu:MLXFast.MLXFastKernel
    let deferred:MLXFast.MLXFastKernel,materialize:MLXFast.MLXFastKernel
    let fixedHistory:MLXFast.MLXFastKernel,fusedHistory:MLXFast.MLXFastKernel,fixedMaterialize:MLXFast.MLXFastKernel
    let prepare:MLXFast.MLXFastKernel
    init() throws {
        func body(_ name:String) throws ->String {
            let u=Bundle.module.url(forResource:name+".metal",withExtension:"txt",subdirectory:"Kernels")!
            return try String(contentsOf:u,encoding:.utf8)
        }
        row=MLXFast.metalKernel(name:"st_swift_int4_row",inputNames:["x","w","scales","biases"],outputNames:["y"],source:try body("row_reuse"))
        qmv=MLXFast.metalKernel(name:"st_swift_int4_qmv",inputNames:["x","w","scales","biases"],outputNames:["y"],source:try body("compact_qmv"))
        gdn=MLXFast.metalKernel(name:"st_swift_gdn_packed",inputNames:["q","k","v","g","beta","state_in","T"],outputNames:["y","state_out"],source:try body("packed_gdn"))
        shared=MLXFast.metalKernel(name:"st_swift_gdn_shared",inputNames:["q","k","v","g","beta","prefix"],outputNames:["y","state"],source:try body("shared_gdn"))
        stream=MLXFast.metalKernel(name:"st_swift_stream_qmv",inputNames:["x","w","scales","biases"],outputNames:["y"],source:try body("stream_qmv"))
        swiglu=MLXFast.metalKernel(name:"st_swift_fused_swiglu",inputNames:["x","gw","gs","gb","uw","us","ub"],outputNames:["y"],source:try body("fused_swiglu"))
        deferred=MLXFast.metalKernel(name:"st_swift_deferred_gdn",inputNames:["q","k","v","g","beta","base","history_k","history_g","history_delta"],outputNames:["y","delta_out","state_out"],source:try body("deferred_gdn"))
        materialize=MLXFast.metalKernel(name:"st_swift_deferred_materialize",inputNames:["base","history_k","history_g","history_delta"],outputNames:["state_out"],source:try body("deferred_materialize"))
        let inputs=["q","k","v","g","beta","base","history_k","history_g","history_delta","pending"]
        fixedHistory=MLXFast.metalKernel(name:"st_swift_fixed_history",inputNames:inputs,outputNames:["y","delta_out","state_out"],source:try body("fixed_history"))
        fusedHistory=MLXFast.metalKernel(name:"st_swift_fused_history",inputNames:inputs,outputNames:["y","delta_out","state_out","history_k_out","history_g_out","history_delta_out"],source:try body("fused_history"))
        fixedMaterialize=MLXFast.metalKernel(name:"st_swift_fixed_materialize",inputNames:["base","history_k","history_g","history_delta","pending"],outputNames:["state_out"],source:try body("fixed_materialize"))
        prepare=MLXFast.metalKernel(name:"st_swift_gdn_prepare",inputNames:["history","qkv","weight"],outputNames:["q","k","v","next_history"],source:try body("gdn_prepare"))
        q8Scores=MLXFast.metalKernel(name:"st_q8_shared_scores",inputNames:["query","pk","ps","pb","tk","ts","tb"],outputNames:["scores"],source:try body("q8_scores"))
        q8Values=MLXFast.metalKernel(name:"st_q8_shared_values",inputNames:["prob","pv","ps","pb","tv","ts","tb"],outputNames:["output"],source:try body("q8_values"))
        // FP16 storage/I-O, explicit FP32 accumulators. In particular, promote
        // each half read before addition in the qmv/stream group sum.
        func halfBody(_ name:String)throws->String {
            let source=try body(name)
            let pattern=try NSRegularExpression(pattern:"x\\[[^\\]]+\\]")
            return pattern.stringByReplacingMatches(in:source,range:NSRange(source.startIndex...,in:source),withTemplate:"float($0)")
        }
        row16=MLXFast.metalKernel(name:"st_swift_int4_row_a16",inputNames:["x","w","scales","biases"],outputNames:["y"],source:try halfBody("row_reuse"))
        qmv16=MLXFast.metalKernel(name:"st_swift_int4_qmv_a16",inputNames:["x","w","scales","biases"],outputNames:["y"],source:try halfBody("compact_qmv"))
        stream16=MLXFast.metalKernel(name:"st_swift_stream_qmv_a16",inputNames:["x","w","scales","biases"],outputNames:["y"],source:try halfBody("stream_qmv"))
        swiglu16=MLXFast.metalKernel(name:"st_swift_fused_swiglu_a16",inputNames:["x","gw","gs","gb","uw","us","ub"],outputNames:["y"],source:try halfBody("fused_swiglu"))
    }
}

public final class Int4Model {
    // Diagnostic switches are lane-only and never change packed weight storage.
    var decodeProjectionMode = "reuse"
    // Opt-in eight-head ABI. Existing four-head callers retain their exact gate.
    var decodeReuseBatchLimit = 4
    var decodeFusedPrepare = false
    var decodeFusedQ8 = false
    var projectionTiles=[String:(Int,Int)]()
    func projectionTile(_ kind:Int,_ m:Int,_ n:Int,_ k:Int,_ rows:Int,_ threads:Int)->(Int,Int) {
        if projectionTiles.isEmpty {return (rows,threads)}
        return projectionTiles["\(kind)/\(m)/\(n)/\(k)"] ?? (rows,threads)
    }
    var operatorProfiling = false
    var operatorSeconds = [String:Double]()
    var operatorCalls = [String:Int]()
    func measured(_ name:String,_ operation:()->MLXArray)->MLXArray {
        guard operatorProfiling else {return operation()}
        let begin=ProcessInfo.processInfo.systemUptime
        let result=operation()
        eval(result);Stream.gpu.synchronize()
        operatorSeconds[name,default:0]+=ProcessInfo.processInfo.systemUptime-begin
        operatorCalls[name,default:0]+=1
        return result
    }
    public let precision:InferencePrecision
    public let numerics:NumericalPolicy
    let c:EngineConfig,bodies:MetalBodies
    var weights:[String:MLXArray]
    public let projectionPolicy:ProjectionPolicy
    public let deferredInterval:Int
    public let historyPolicy:HistoryPolicy
    public let gdnPreparePolicy:GDNPreparePolicy
    public var prefillEvaluationStride=0
    public var traceEnabled=false
    var trace=[String:MLXArray](),traceCall=0
    public func saveTrace(_ url:URL) throws {try save(arrays:trace,url:url)}
    let decoder:ByteDecoder
    let compiledSilu=compile(shapeless:true) { (x:MLXArray) in x*sigmoid(x) }
    let compiledSwiGLU=compile(shapeless:true) { (gate:MLXArray,x:MLXArray) in (gate*sigmoid(gate))*x }
    let compiledG=compile(shapeless:true) { (aLog:MLXArray,a:MLXArray,bias:MLXArray) in
        exp(-exp(aLog)*logAddExp(a+bias,MLXArray(Float(0))))
    }
    public init(directory:URL,projectionPolicy:ProjectionPolicy = .baseline,deferredInterval:Int=1,historyPolicy:HistoryPolicy = .dynamic,gdnPreparePolicy:GDNPreparePolicy = .baseline,precision:InferencePrecision = .fp32,numerics:NumericalPolicy = .standard) throws {
        self.precision=precision;self.numerics=numerics
        guard [1,2,4,8,16].contains(deferredInterval) else {throw EngineError.invalidConfiguration("deferred GDN interval must be 1/2/4/8/16")}
        self.deferredInterval=deferredInterval
        guard historyPolicy == .dynamic || deferredInterval>1 else {throw EngineError.invalidConfiguration("fixed history requires deferred interval >1")}
        self.historyPolicy=historyPolicy
        self.gdnPreparePolicy=gdnPreparePolicy
        self.projectionPolicy=projectionPolicy
        c=try EngineConfig(directory.appendingPathComponent("config.json"));bodies=try MetalBodies()
        decoder=try ByteDecoder(directory.appendingPathComponent("tokenizer.json"))
        var result=[String:MLXArray]()
        let files=try FileManager.default.contentsOfDirectory(at:directory,includingPropertiesForKeys:nil)
            .filter{$0.pathExtension=="safetensors"}.sorted{$0.lastPathComponent<$1.lastPathComponent}
        guard !files.isEmpty else { throw EngineError.invalidConfiguration("missing safetensors") }
        for file in files {
            for (key,value) in try loadArrays(url:file) {
                guard key.hasPrefix("language_model.model.") else { continue }
                let name=String(key.dropFirst("language_model.model.".count))
                result[name]=name.hasSuffix(".scales") || name.hasSuffix(".biases") || value.dtype == .uint32
                    ? value : value.asType(precision.dtype)
                if precision != .fp32 && (name.hasSuffix(".A_log") || name.hasSuffix(".dt_bias") || (numerics.fp32GDN && name.hasSuffix(".conv1d.weight"))) {result[name]=value.asType(.float32)}
            }
        }
        weights=result
        guard let embedding=weights["embed_tokens.weight"],embedding.dtype == .uint32,
              embedding.dim(1)*8==c.hidden else { throw EngineError.invalidConfiguration("invalid packed embedding") }
        var dense=["embed_tokens"],required=["norm.weight"]
        for i in 0..<c.layers {
            let p="layers.\(i)"
            required += [p+".input_layernorm.weight",p+".post_attention_layernorm.weight"]
            dense += ["gate_proj","up_proj","down_proj"].map{p+".mlp."+$0}
            if (i+1)%c.interval==0 {
                dense += ["q_proj","k_proj","v_proj","o_proj"].map{p+".self_attn."+$0}
                required += [p+".self_attn.q_norm.weight",p+".self_attn.k_norm.weight"]
            } else {
                dense += ["in_proj_qkv","in_proj_z","in_proj_b","in_proj_a","out_proj"].map{p+".linear_attn."+$0}
                required += ["A_log","dt_bias","conv1d.weight","norm.weight"].map{p+".linear_attn."+$0}
            }
        }
        for name in required where weights[name]==nil {throw EngineError.missingTensor(name)}
        for name in dense {
            guard let w=weights[name+".weight"],let scale=weights[name+".scales"],let bias=weights[name+".biases"],
                  w.ndim==2,w.dtype == .uint32,scale.shape==[w.dim(0),w.dim(1)/8],bias.shape==scale.shape
            else {throw EngineError.invalidConfiguration("Incomplete int4 tensor: \(name)")}
        }
        eval(Array(weights.values))
    }
    // Benchmark variants share immutable packed weights, metadata and kernels.
    // No reloading, concatenation, requantization or dense weight materialization.
    init(sharing base:Int4Model,policy:ProjectionPolicy,deferredInterval:Int?=nil,historyPolicy:HistoryPolicy?=nil,gdnPreparePolicy:GDNPreparePolicy?=nil) {
        precision=base.precision;numerics=base.numerics
        c=base.c;bodies=base.bodies;weights=base.weights;decoder=base.decoder;projectionPolicy=policy
        self.deferredInterval=deferredInterval ?? base.deferredInterval
        self.historyPolicy=self.deferredInterval>1 ? (historyPolicy ?? base.historyPolicy) : .dynamic
        self.gdnPreparePolicy=gdnPreparePolicy ?? base.gdnPreparePolicy
    }
    func w(_ name:String)->MLXArray { precondition(weights[name] != nil,"missing tensor \(name)");return weights[name]! }
    public func newCache()->[LayerState] { (0..<c.layers).map{_ in LayerState()} }
    public func evaluate(_ caches:[LayerState]) { eval(caches.flatMap{$0.arrays}) }
    public var weightBytes:Int { weights.values.reduce(0){$0+$1.nbytes} }
    func linear(_ x:MLXArray,_ path:String)->MLXArray {
        if !operatorProfiling {return linearImpl(x,path)}
        return measured("linear." + String(path.split(separator:".").last!)) {linearImpl(x,path)}
    }
    private func linearImpl(_ x:MLXArray,_ path:String)->MLXArray {
        let weight=w(path+".weight"),s=w(path+".scales"),bias=w(path+".biases")
        let k=x.dim(-1),n=weight.dim(0),m=x.size/k
        // Diagnostic: use the same arithmetic and fused MLP for B1..B4,
        // including after independent heads stop. Not a production default.
        let consistent=decodeProjectionMode == "reuse-consistent"
        if precision != .fp32 && decodeProjectionMode == "legacy" {
            return quantizedMM(x,weight,scales:s.asType(.float16),biases:bias.asType(.float16),groupSize:64,bits:4)
        }
        let row=precision == .fp32 ? bodies.row:bodies.row16
        let qmv=precision == .fp32 ? bodies.qmv:bodies.qmv16
        let stream=precision == .fp32 ? bodies.stream:bodies.stream16
        let outputShape=Array(x.shape.dropLast())+[n]
        if !consistent && m==1 && k%512==0 && n%8==0 && (s.dtype == .float16 || (precision != .fp32 && decodeProjectionMode == "reuse-qmv")) {
            return qmv([x,weight,s,bias],template:[("K",k),("N",n)],
                grid:(((n+7)/8)*64,1,1),threadGroup:(64,1,1),outputShapes:[outputShape],outputDTypes:[precision.dtype])[0]
        }
        if (2...decodeReuseBatchLimit).contains(m) || (consistent && m==1) {
            if projectionPolicy.usesStream && k%512==0 && n>=256 && !((consistent || m==4) && (n>=65536 || k>4096)) {
                let (rows,threads)=projectionTile(1,m,n,k,4,128)
                let groupRows=rows*threads/32
                return stream([x,weight,s,bias],template:[("M",m),("N",n),("K",k),("ROWS",rows)],
                    grid:(((n+groupRows-1)/groupRows)*threads,1,1),threadGroup:(threads,1,1),
                    outputShapes:[outputShape],outputDTypes:[precision.dtype])[0]
            }
            let (rows,threads)=projectionTile(0,m,n,k,n<256 ? 1 : n>=65536 ? 4 : 8,n<256 ? 128 : k<=2560 || n>=65536 ? 256 : 128)
            let groupRows=rows*threads/32
            return row([x,weight,s,bias],template:[("M",m),("N",n),("K",k),("ROWS",rows)],
                grid:(((n+groupRows-1)/groupRows)*threads,1,1),threadGroup:(threads,1,1),
                outputShapes:[outputShape],outputDTypes:[precision.dtype])[0]
        }
        return quantizedMM(x,weight,scales:s.asType(precision.dtype),biases:bias.asType(precision.dtype),groupSize:64,bits:4)
    }
    func norm(_ x:MLXArray,_ path:String)->MLXArray { MLXFast.rmsNorm(x,weight:w(path+".weight"),eps:c.epsilon) }
    func silu(_ x:MLXArray)->MLXArray { compiledSilu(x) }
    func mlp(_ x:MLXArray,_ path:String)->MLXArray {
        let k=x.dim(-1),m=x.size/k
        if projectionPolicy.usesFused && ((2...decodeReuseBatchLimit).contains(m) || (decodeProjectionMode == "reuse-consistent" && m==1)) && !(precision != .fp32 && decodeProjectionMode == "legacy") {
            let gp=path+".gate_proj",up=path+".up_proj",n=w(gp+".weight").dim(0)
            let is4b=n==9216 && k==2560
            let (rows,threads)=projectionTile(2,m,n,k,is4b && m != 3 ? 1 : 4,is4b && m != 3 ? 256 : 128)
            let groupRows=rows*threads/32
            let values=[x,w(gp+".weight"),w(gp+".scales"),w(gp+".biases"),w(up+".weight"),w(up+".scales"),w(up+".biases")]
            let swiglu=precision == .fp32 ? bodies.swiglu:bodies.swiglu16
            let out=measured("fused.gate_up_swiglu") {swiglu(values,template:[("M",m),("N",n),("K",k),("ROWS",rows)],
                grid:(((n+groupRows-1)/groupRows)*threads,1,1),threadGroup:(threads,1,1),
                outputShapes:[Array(x.shape.dropLast())+[n]],outputDTypes:[precision.dtype])[0]}
            return linear(out,path+".down_proj")
        }
        return linear(compiledSwiGLU(linear(x,path+".gate_proj"),linear(x,path+".up_proj")),path+".down_proj")
    }
    func recurrence(_ q:MLXArray,_ k:MLXArray,_ v:MLXArray,_ g:MLXArray,_ beta:MLXArray,_ state:MLXArray)->(MLXArray,MLXArray) {
        let b=q.dim(0),s=q.dim(1),hk=q.dim(2),hv=v.dim(2),dv=v.dim(3)
        if state.dim(0)==1 && b>1 && s==1 {
            let r=bodies.shared([q,k,v,g,beta,state],template:[("Hk",hk),("Hv",hv),("Dv",dv)],
                grid:(32,dv/8,b*hv),threadGroup:(32,2,1),outputShapes:[v.shape,[b,hv,dv,128]],outputDTypes:[.float32,.float32])
            return (r[0],r[1])
        }
        precondition(state.dim(0)==b)
        let r=bodies.gdn([q,k,v,g,beta,state,MLXArray(Int32(s))],
            template:[("InT",DType.float32),("StT",DType.float32),("Dk",128),("Dv",dv),("Hk",hk),("Hv",hv)],
            grid:(32,dv/8,b*hv),threadGroup:(32,2,1),outputShapes:[v.shape,state.shape],outputDTypes:[.float32,.float32])
        return (r[0],r[1])
    }
    func gdn(_ x:MLXArray,_ p:String,_ cache:LayerState)->MLXArray {
        let b=x.dim(0),s=x.dim(1),kd=c.keyHeads*c.keyDim,vd=c.valueHeads*c.valueDim,cd=2*kd+vd
        let qkv=linear(x,p+".in_proj_qkv"),z=linear(x,p+".in_proj_z").reshaped(b,s,c.valueHeads,c.valueDim)
        let ba=linear(x,p+".in_proj_b"),a=linear(x,p+".in_proj_a")
        let history=cache.conv ?? MLXArray.zeros([b,3,cd],dtype:numerics.fp32GDN ? .float32:precision.dtype)
        let q:MLXArray,k:MLXArray,v:MLXArray
        if (gdnPreparePolicy == .fused || decodeFusedPrepare) && s==1 && (1...4).contains(b) && c.valueDim==128 {
            let groups=b*(2*c.keyHeads+c.valueHeads)
            let values=bodies.prepare([history,qkv,w(p+".conv1d.weight")],
                template:[("B",b),("D",cd),("Hk",c.keyHeads),("Hv",c.valueHeads)],
                grid:(((groups+3)/4)*128,1,1),threadGroup:(128,1,1),
                outputShapes:[[b,1,c.keyHeads,128],[b,1,c.keyHeads,128],[b,1,c.valueHeads,128],history.shape],
                outputDTypes:[.float32,.float32,.float32,.float32])
            q=values[0];k=values[1];v=values[2];cache.conv=values[3]
        } else {
            let full=concatenated([history,numerics.fp32GDN ? qkv.asType(.float32):qkv],axis:1)
            cache.conv=contiguous(full[0...,(-3)...,0...])
            let cv=silu(conv1d(full,w(p+".conv1d.weight"),groups:cd))
            let chunks=cv.split(indices:[kd,2*kd],axis:-1)
            let rawQ=chunks[0].reshaped(b,s,c.keyHeads,128),rawK=chunks[1].reshaped(b,s,c.keyHeads,128)
            let inv=Float(pow(128.0,-0.5)),eps:Float=1e-6/128
            q=Float(1.0/128.0)*MLXFast.rmsNorm(rawQ,weight:.mlxNone,eps:eps)
            k=inv*MLXFast.rmsNorm(rawK,weight:.mlxNone,eps:eps)
            v=chunks[2].reshaped(b,s,c.valueHeads,c.valueDim)
        }
        let g=compiledG(w(p+".A_log"),a.asType(.float32),w(p+".dt_bias"))
        let state=cache.matrix ?? MLXArray.zeros([b,c.valueHeads,c.valueDim,128],dtype:.float32)
        let out:MLXArray
        if cache.deferredKeys != nil {
            precondition(s==1 && deferredInterval>1)
            out=deferredRecurrence(q,k,v,g,sigmoid(ba),cache)
        } else {
            let result=recurrence(q.asType(.float32),k.asType(.float32),v.asType(.float32),g,sigmoid(ba.asType(.float32)),state)
            out=result.0;cache.matrix=result.1
        }
        cache.offset+=s
        return linear(compiledSwiGLU(z,norm(out.asType(precision.dtype),p+".norm")).reshaped(b,s,-1),p+".out_proj")
    }
    func attention(_ x:MLXArray,_ p:String,_ cache:LayerState)->MLXArray {
        let b=x.dim(0),s=x.dim(1),d=c.headDim,h=c.heads,hk=c.kvHeads,r=h/hk
        let qs=linear(x,p+".q_proj").reshaped(b,s,h,2*d).split(parts:2,axis:-1)
        let gate=qs[1].reshaped(b,s,-1)
        var q=norm(qs[0],p+".q_norm").transposed(0,2,1,3)
        var k=norm(linear(x,p+".k_proj").reshaped(b,s,hk,d),p+".k_norm").transposed(0,2,1,3)
        let v=linear(x,p+".v_proj").reshaped(b,s,hk,d).transposed(0,2,1,3)
        // Pinned core 0.31.1 selects rope_single for contiguous T1 and one
        // offset, dispatching only N heads instead of B*N. Give B offsets
        // when B>1 to select the general batched kernel; all positions agree.
        let position=cache.tailUsed.map{$0+Int32(cache.prefixK!.dim(2))} ?? MLXArray(Int32(cache.offset))
        let ropeOffset=b>1 ? broadcast(position,to:[b]) : position
        q=MLXFast.RoPE(q,dimensions:c.ropeDims,traditional:false,base:c.ropeTheta,scale:1,offset:ropeOffset)
        k=MLXFast.RoPE(k,dimensions:c.ropeDims,traditional:false,base:c.ropeTheta,scale:1,offset:ropeOffset)
        if precision == .a16kv8 {
            let output=q8Attention(q,k,v,cache)
            cache.offset+=s
            return linear(output.transposed(0,2,1,3).reshaped(b,s,-1)*sigmoid(gate),p+".o_proj")
        }
        if let used=cache.tailUsed {
            cache.k=updateTail(cache.k!,k,used);cache.v=updateTail(cache.v!,v,used)
            cache.tailUsed=used+1
        } else {
            cache.k=cache.k.map{concatenated([$0,k],axis:2)} ?? k
            cache.v=cache.v.map{concatenated([$0,v],axis:2)} ?? v
        }
        cache.offset+=s
        let scale=Float(pow(Double(d),-0.5));let output:MLXArray
        if let pk=cache.prefixK,let pv=cache.prefixV {
            precondition(s==1 && pk.dim(0)==1)
            let n=pk.dim(2)
            let grouped=q.reshaped(b,hk,r,d).transposed(1,0,2,3).reshaped(hk,b*r,d)
            let ps=(grouped*scale).matmul(pk[0].swappedAxes(-1,-2)).reshaped(hk,b,r,n).transposed(1,0,2,3)
            var ts=(q.reshaped(b,hk,r,d)*scale).matmul(cache.k!.swappedAxes(-1,-2))
            if let used=cache.tailUsed {ts=which(MLXArray(0..<cache.k!.dim(2)) .< used,ts,-Float.infinity)}
            let probs=softmax(concatenated([ps,ts],axis:-1),axis:-1,precise:true)
            let pp=probs[.ellipsis,..<n].transposed(1,0,2,3).reshaped(hk,b*r,n)
            let po=pp.matmul(pv[0]).reshaped(hk,b,r,d).transposed(1,0,2,3)
            let to=probs[.ellipsis,n...].matmul(cache.v!)
            output=(po+to).reshaped(b,h,1,d)
            if traceEnabled {
                let base="call\(traceCall-1).\(p)"
                trace[base+".q"]=q;trace[base+".pk"]=pk;trace[base+".pv"]=pv
                trace[base+".tk"]=cache.k!;trace[base+".tv"]=cache.v!;trace[base+".out"]=output
            }
        } else {
            output=MLXFast.scaledDotProductAttention(queries:q,keys:cache.k!,values:cache.v!,scale:scale,
                mask:s>1 ? .causal : .none)
        }
        return linear(output.transposed(0,2,1,3).reshaped(b,s,-1)*sigmoid(gate),p+".o_proj")
    }
    public func hidden(_ ids:MLXArray,cache:[LayerState],evaluateEvery:Int=0)->MLXArray {
        precondition(cache.count==c.layers)
        var x=dequantized(w("embed_tokens.weight").take(ids,axis:0),
            scales:w("embed_tokens.scales").take(ids,axis:0).asType(precision.dtype),
            biases:w("embed_tokens.biases").take(ids,axis:0).asType(precision.dtype),groupSize:64,bits:4,dtype:precision.dtype)
        let call=traceCall;traceCall+=1
        if traceEnabled {trace["call\(call).embedding"]=x}
        for i in 0..<c.layers {
            let p="layers.\(i)",n=norm(x.asType(precision.dtype),p+".input_layernorm")
            let update=measured((i+1)%c.interval != 0 ? "stage.gdn":"stage.attention") {
                (i+1)%c.interval != 0 ? gdn(n,p+".linear_attn",cache[i]) : attention(n,p+".self_attn",cache[i])
            }
            let h=numerics.fp32Residual ? x.asType(.float32)+update.asType(.float32):x+update
            let mlp=norm(h.asType(precision.dtype),p+".post_attention_layernorm")
            let branch=measured("stage.mlp") {self.mlp(mlp,p+".mlp")}
            x=numerics.fp32Residual ? h+branch.asType(.float32):h+branch
            if traceEnabled {trace["call\(call).layer\(i)"]=x}
            if evaluateEvery>0 && ((i+1)%evaluateEvery==0 || i+1==c.layers) {
                // Evaluate a block before building the next one, so overwritten
                // GDN states and temporary activations can be released promptly.
                // No arithmetic, weight precision, batch or shared KV changes.
                let first=(i/evaluateEvery)*evaluateEvery
                eval([x]+cache[first...i].flatMap{$0.arrays})
                // eval submits GPU work; wait at this explicit memory boundary
                // before enqueuing the next block and retaining its resources.
                Stream.gpu.synchronize()
            }
        }
        return norm(x.asType(precision.dtype),"norm")
    }
    public func logits(_ ids:MLXArray,cache:[LayerState])->MLXArray { linear(hidden(ids,cache:cache)[0...,-1,0...],"embed_tokens") }
    public func prefill(_ ids:[Int],cache:[LayerState]?=nil,chunkSize:Int=256)->[LayerState] {
        precondition(chunkSize>0)
        let result=cache ?? newCache()
        for start in stride(from:0,to:ids.count,by:chunkSize) {
            let part=Array(ids[start..<min(ids.count,start+chunkSize)])
            // Prefill only publishes cache state. Do not force unused final
            // layer output/MLP/norm onto the GPU; lazy dependency evaluation
            // still executes every operation required by K/V and GDN state.
            _ = hidden(MLXArray(part,[1,part.count]),cache:result,evaluateEvery:prefillEvaluationStride)
            evaluate(result)
        }
        return result
    }
}
