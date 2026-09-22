// swift-tools-version: 6.1
import PackageDescription

let package = Package(
    name: "apple-embeddings-server",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(
            name: "apple-embeddings-server",
            targets: ["AppleEmbeddingsServer"]
        )
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
        .package(url: "https://github.com/hummingbird-project/hummingbird.git", from: "2.0.0")
    ],
    targets: [
        .executableTarget(
            name: "AppleEmbeddingsServer",
            dependencies: [
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
                .product(name: "Hummingbird", package: "hummingbird")
            ]
        ),
        .testTarget(
            name: "AppleEmbeddingsServerTests",
            dependencies: [
                "AppleEmbeddingsServer",
                .product(name: "HummingbirdTesting", package: "hummingbird")
            ]
        )
    ]
)
