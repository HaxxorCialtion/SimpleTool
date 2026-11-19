// swift-tools-version: 6.2
import PackageDescription
let package=Package(name:"SimpleToolSwift",platforms:[.macOS(.v14),.iOS(.v17)],
    products:[.library(name:"SimpleToolSwift",targets:["SimpleToolSwift"]),
              .executable(name:"SimpleToolSwiftCLI",targets:["SimpleToolSwiftCLI"]),.executable(name:"InfraSmoke",targets:["InfraSmoke"]),
              .executable(name:"MobileActionsBench",targets:["MobileActionsBench"])],
    dependencies:[.package(path:"../Vendor/mlx-swift")],
    targets:[.target(name:"SimpleToolSwift",dependencies:[.product(name:"MLX",package:"mlx-swift")],
                     resources:[.copy("Kernels")]),
             .executableTarget(name:"SimpleToolSwiftCLI",dependencies:["SimpleToolSwift"]),
             .executableTarget(name:"InfraSmoke",dependencies:["SimpleToolSwift",.product(name:"MLX",package:"mlx-swift")]),
             .executableTarget(name:"MobileActionsBench",dependencies:["SimpleToolSwift"])],
    swiftLanguageModes:[.v5])
