import Testing
import Foundation
import HTTPTypes
import Hummingbird
import HummingbirdTesting
@testable import AppleEmbeddingsServer

@Suite("Apple Embeddings Server Tests")
struct AppleEmbeddingsServerTests {

    @Test("Model Aliases and Properties")
    func testModelAliases() {
        #expect(ModelAlias.allCases.count == 6)
        #expect(ModelAlias.default == .latin)
        #expect(ModelAlias.latin.rawValue == "apple-nl-contextual-latin")
        #expect(ModelAlias.cjk.rawValue == "apple-nl-contextual-cjk")
    }

    @Test("Embedding Request JSON Decoding - Single String")
    func testSingleStringRequestDecoding() throws {
        let json = """
        {
            "input": "hello world",
            "model": "apple-nl-contextual-latin",
            "encoding_format": "float"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let req = try decoder.decode(EmbeddingRequest.self, from: json)
        #expect(req.model == "apple-nl-contextual-latin")
        #expect(req.encodingFormat == "float")
        #expect(req.input.values == ["hello world"])
    }

    @Test("Embedding Request JSON Decoding - String Array")
    func testArrayStringRequestDecoding() throws {
        let json = """
        {
            "input": ["first sentence", "second sentence"],
            "model": "apple-nl-contextual-latin"
        }
        """.data(using: .utf8)!

        let decoder = JSONDecoder()
        let req = try decoder.decode(EmbeddingRequest.self, from: json)
        #expect(req.input.values == ["first sentence", "second sentence"])
    }

    @Test("Mean Pool and L2 Normalization Logic")
    func testMeanPoolAndL2Normalize() throws {
        let dummyVectors: [[Double]] = [
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0]
        ]
        let dimension = 3

        // Mean vector before normalization = [1.5, 2.0, 0.0]
        // Magnitude = sqrt(1.5^2 + 2.0^2) = sqrt(2.25 + 4.0) = sqrt(6.25) = 2.5
        // Normalized vector = [1.5/2.5, 2.0/2.5, 0.0] = [0.6, 0.8, 0.0]
        let result = try EmbeddingEngine.meanPoolAndL2Normalize(tokenVectors: dummyVectors, expectedDimension: dimension)

        #expect(result.count == 3)
        #expect(abs(result[0] - 0.6) < 1e-5)
        #expect(abs(result[1] - 0.8) < 1e-5)
        #expect(abs(result[2] - 0.0) < 1e-5)

        // Verify norm is 1.0
        let norm = sqrt(result.reduce(0.0) { $0 + $1 * $1 })
        #expect(abs(norm - 1.0) < 1e-5)
    }

    @Test("Mean Pool throws on Empty Token Vectors")
    func testMeanPoolEmptyInput() {
        #expect(throws: EmbeddingEngineError.self) {
            try EmbeddingEngine.meanPoolAndL2Normalize(tokenVectors: [], expectedDimension: 512)
        }
    }

    @Test("HTTP Endpoint /health")
    func testHealthEndpoint() async throws {
        // Test with mock engine logic or server setup
        struct DummyEngine {
            static func createServer() async throws -> EmbeddingServer {
                // If running on macOS with NLContextualEmbedding:
                let engine = try await EmbeddingEngine(alias: .latin)
                return EmbeddingServer(engine: engine)
            }
        }

        #if os(macOS)
        let server = try await DummyEngine.createServer()
        let router = server.buildRouter()
        let app = Application(router: router)

        try await app.test(.live) { client in
            try await client.execute(uri: "/health", method: .get) { response in
                #expect(response.status == .ok)
                let bodyString = String(buffer: response.body)
                #expect(bodyString.contains("ok"))
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /v1/models")
    func testModelsEndpoint() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        try await app.test(.live) { client in
            try await client.execute(uri: "/v1/models", method: .get) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("apple-nl-contextual-latin"))
                #expect(body.contains("apple-nl-contextual-cjk"))
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /metadata")
    func testMetadataEndpoint() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        try await app.test(.live) { client in
            try await client.execute(uri: "/metadata", method: .get) { response in
                #expect(response.status == .ok)
                let body = String(buffer: response.body)
                #expect(body.contains("apple-nl-contextual-latin"))
                #expect(body.contains("dimension"))
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /v1/embeddings - Single String Request")
    func testEmbeddingsSingleString() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        let reqPayload = """
        {
            "input": "apple embeddings server testing",
            "model": "apple-nl-contextual-latin",
            "encoding_format": "float"
        }
        """

        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/embeddings",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: reqPayload)
            ) { response in
                #expect(response.status == .ok)
                let decoder = JSONDecoder()
                let resp = try decoder.decode(EmbeddingResponse.self, from: Data(buffer: response.body))
                #expect(resp.data.count == 1)
                #expect(resp.data[0].index == 0)
                #expect(resp.data[0].embedding.count == (await engine.dimension))
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /v1/embeddings - Order Preservation")
    func testEmbeddingsOrderPreservation() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        let reqPayload = """
        {
            "input": ["first document", "second document", "third document"],
            "model": "apple-nl-contextual-latin"
        }
        """

        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/embeddings",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: reqPayload)
            ) { response in
                #expect(response.status == .ok)
                let decoder = JSONDecoder()
                let resp = try decoder.decode(EmbeddingResponse.self, from: Data(buffer: response.body))
                #expect(resp.data.count == 3)
                #expect(resp.data[0].index == 0)
                #expect(resp.data[1].index == 1)
                #expect(resp.data[2].index == 2)
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /v1/embeddings - Unsupported Model")
    func testUnsupportedModel() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        let reqPayload = """
        {
            "input": "hello",
            "model": "non-existent-model"
        }
        """

        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/embeddings",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: reqPayload)
            ) { response in
                #expect(response.status == .badRequest)
                let decoder = JSONDecoder()
                let errResp = try decoder.decode(OpenAIErrorResponse.self, from: Data(buffer: response.body))
                #expect(errResp.error.type == "invalid_request_error")
                #expect(errResp.error.param == "model")
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /v1/embeddings - Unsupported Encoding Format")
    func testUnsupportedEncodingFormat() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        let reqPayload = """
        {
            "input": "hello",
            "encoding_format": "base64"
        }
        """

        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/embeddings",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: reqPayload)
            ) { response in
                #expect(response.status == .badRequest)
                let decoder = JSONDecoder()
                let errResp = try decoder.decode(OpenAIErrorResponse.self, from: Data(buffer: response.body))
                #expect(errResp.error.param == "encoding_format")
            }
        }
        #endif
    }

    @Test("HTTP Endpoint /v1/embeddings - Malformed JSON")
    func testMalformedJSON() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)
        let server = EmbeddingServer(engine: engine)
        let app = Application(router: server.buildRouter())

        try await app.test(.live) { client in
            try await client.execute(
                uri: "/v1/embeddings",
                method: .post,
                headers: [.contentType: "application/json"],
                body: ByteBuffer(string: "{ invalid json ")
            ) { response in
                #expect(response.status == .badRequest)
                let decoder = JSONDecoder()
                let errResp = try decoder.decode(OpenAIErrorResponse.self, from: Data(buffer: response.body))
                #expect(errResp.error.type == "invalid_request_error")
            }
        }
        #endif
    }

    @Test("Concurrent Requests Handling")
    func testConcurrentRequests() async throws {
        #if os(macOS)
        let engine = try await EmbeddingEngine(alias: .latin)

        await withTaskGroup(of: Void.self) { group in
            for i in 0..<10 {
                group.addTask {
                    let res = try? await engine.generateEmbedding(for: "Concurrent text request \(i)")
                    #expect(res != nil)
                }
            }
        }
        #endif
    }
}
