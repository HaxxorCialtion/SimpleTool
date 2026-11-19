import Foundation
import MLX

public func projectionRegression(_ root:URL) throws ->[String:Any] {
    let bodies=try MetalBodies();var cases=[[String:Any]]()
    let folders=try FileManager.default.contentsOfDirectory(at:root,includingPropertiesForKeys:nil)
        .filter{FileManager.default.fileExists(atPath:$0.appendingPathComponent("format.json").path)}
        .sorted{$0.lastPathComponent<$1.lastPathComponent}
    for folder in folders {
        let format=try JSONSerialization.jsonObject(with:Data(contentsOf:folder.appendingPathComponent("format.json"))) as! [String:Any]
        let n=format["N"] as! Int,k=format["K"] as! Int
        func read(_ name:String,_ shape:[Int],_ dtype:DType) throws ->MLXArray {
            let data=try Data(contentsOf:folder.appendingPathComponent(name+".bin"))
            guard data.count==shape.reduce(1,*)*dtype.size else {throw EngineError.invalidPacket("fixture byte length \(name)")}
            return MLXArray(data,shape,dtype:dtype)
        }
        let gw=try read("gw",[n,k/8],.uint32),uw=try read("uw",[n,k/8],.uint32)
        for compact in [true,false] {
            let dtype:DType=compact ? .float16 : .float32,suffix=compact ? "half" : "float"
            let gs=try read("gs_"+suffix,[n,k/64],dtype),gb=try read("gb_"+suffix,[n,k/64],dtype)
            let us=try read("us_"+suffix,[n,k/64],dtype),ub=try read("ub_"+suffix,[n,k/64],dtype)
            for batch in 2...4 {
                let x=try read("x\(batch)",[batch,k],.float32),gold=try read("expected\(batch)",[batch,n],.float32)
                var qmvReference:MLXArray?
                if k%512==0 && n%8==0 {
                    qmvReference=concatenated((0..<batch).map{bodies.qmv([x[$0].reshaped(1,k),gw,gs,gb],
                        template:[("K",k),("N",n)],grid:(((n+7)/8)*64,1,1),threadGroup:(64,1,1),
                        outputShapes:[[1,n]],outputDTypes:[.float32])[0]},axis:0)
                    eval(qmvReference!)
                }
                for rows in [1,2,4,8] {for threads in [64,128,256] {
                    let groupRows=rows*threads/32,grid=(((n+groupRows-1)/groupRows)*threads,1,1)
                    let templates:[(String,any KernelTemplateArg)]=[("M",batch),("N",n),("K",k),("ROWS",rows)]
                    let fused=bodies.swiglu([x,gw,gs,gb,uw,us,ub],template:templates,grid:grid,
                        threadGroup:(threads,1,1),outputShapes:[[batch,n]],outputDTypes:[.float32])[0]
                    let difference=abs(fused-gold).max().item(Float.self)
                    cases.append(["kernel":"fused","fixture":folder.lastPathComponent,"batch":batch,
                        "rows":rows,"threads":threads,"compact":compact,"max_abs":difference,
                        "passed":arrayEqual(fused,gold).item(Bool.self),"reference":"saved Python separate projections + SwiGLU"])
                    if let reference=qmvReference {
                        let streamed=bodies.stream([x,gw,gs,gb],template:templates,grid:grid,
                            threadGroup:(threads,1,1),outputShapes:[[batch,n]],outputDTypes:[.float32])[0]
                        let difference=abs(streamed-reference).max().item(Float.self)
                        cases.append(["kernel":"stream","fixture":folder.lastPathComponent,"batch":batch,
                            "rows":rows,"threads":threads,"compact":compact,"max_abs":difference,
                            "passed":arrayEqual(streamed,reference).item(Bool.self),"reference":"independent single-vector compact QMV for each row"])
                    }
                }}
            }
        }
    }
    guard !cases.isEmpty else {throw EngineError.invalidPacket("missing projection fixtures")}
    return ["cases":cases,"all_passed":cases.allSatisfy{$0["passed"] as? Bool == true},
            "scope":"Swift MLX kernels; structural + actual4B first-layer weights, wide/compact metadata; B2/3/4; R1/2/4/8; T64/128/256"]
}
