import Foundation
import MLX
public func runtimeSmoke() -> [Int32] { (MLXArray([Int32(1),2,3])+1).asArray(Int32.self) }
