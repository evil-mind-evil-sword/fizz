// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "FizzDebugger",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "FizzDebugger", targets: ["FizzDebugger"])
    ],
    targets: [
        .systemLibrary(
            name: "CFizz",
            pkgConfig: nil,
            providers: []
        ),
        .executableTarget(
            name: "FizzDebugger",
            dependencies: ["CFizz"],
            path: "FizzDebugger",
            exclude: ["FizzDebugger-Bridging-Header.h", "Preview Content"],
            swiftSettings: [
                .unsafeFlags(["-I../include"]),
            ],
            linkerSettings: [
                .unsafeFlags(["-L../zig-out/lib", "-lfizz", "-Wl,-rpath,@executable_path/../zig-out/lib"])
            ]
        )
    ]
)
