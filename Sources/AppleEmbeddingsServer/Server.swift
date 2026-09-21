import Foundation
import Hummingbird
import HTTPTypes

public struct ServerContext: Sendable {
    public let defaultEngine: EmbeddingEngine
    public let defaultAlias: ModelAlias
}

public struct EmbeddingServer: Sendable {
    public let host: String
    public let port: Int
    public let engine: EmbeddingEngine

    public init(host: String = "127.0.0.1", port: Int = 8080, engine: EmbeddingEngine) {
        self.host = host
        self.port = port
        self.engine = engine
    }

    public func buildRouter() -> Router<BasicRequestContext> {
        let router = Router()

        // GET /health
        router.get("health") { _, _ -> Response in
            let body = "{\"status\":\"ok\"}"
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(string: body))
            )
        }

        // GET /metadata
        router.get("metadata") { _, _ -> Response in
            let metadata = await self.engine.getMetadata()
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            guard let data = try? encoder.encode(metadata) else {
                return Response(status: .internalServerError)
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: data))
            )
        }

        // GET /v1/models
        router.get("v1/models") { _, _ -> Response in
            let activeMeta = await self.engine.getMetadata()
            let modelsData: [ModelObject] = ModelAlias.allCases.map { alias in
                if alias == self.engine.alias {
                    return ModelObject(
                        id: alias.rawValue,
                        modelIdentifier: activeMeta.modelIdentifier,
                        revision: activeMeta.revision,
                        dimension: activeMeta.dimension,
                        maximumSequenceLength: activeMeta.maximumSequenceLength
                    )
                } else {
                    return ModelObject(id: alias.rawValue)
                }
            }

            let response = ModelListResponse(data: modelsData)
            let encoder = JSONEncoder()
            guard let data = try? encoder.encode(response) else {
                return Response(status: .internalServerError)
            }
            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: data))
            )
        }

        // POST /v1/embeddings
        router.post("v1/embeddings") { request, context -> Response in
            let bodyBuffer: ByteBuffer
            do {
                bodyBuffer = try await request.body.collect(upTo: 10 * 1024 * 1024)
            } catch {
                return Self.makeErrorResponse(
                    message: "Failed to read request body",
                    status: .badRequest,
                    param: "body"
                )
            }

            let requestData = Data(buffer: bodyBuffer)
            let decoder = JSONDecoder()
            let req: EmbeddingRequest
            do {
                req = try decoder.decode(EmbeddingRequest.self, from: requestData)
            } catch {
                return Self.makeErrorResponse(
                    message: "Malformed JSON payload: \(error.localizedDescription)",
                    status: .badRequest,
                    param: "input"
                )
            }

            // Check encoding format
            if let fmt = req.encodingFormat, fmt != "float" {
                return Self.makeErrorResponse(
                    message: "Unsupported encoding_format: '\(fmt)'. Only 'float' is supported.",
                    status: .badRequest,
                    param: "encoding_format"
                )
            }

            // Check requested model matching loaded server engine alias
            if let reqModel = req.model {
                if reqModel != self.engine.alias.rawValue {
                    return Self.makeErrorResponse(
                        message: "Requested model '\(reqModel)' does not match the model currently hosted by this server ('\(self.engine.alias.rawValue)').",
                        status: .badRequest,
                        param: "model"
                    )
                }
            }

            let inputValues = req.input.values

            // Validate non-empty array / non-empty strings
            if inputValues.isEmpty {
                return Self.makeErrorResponse(
                    message: "Input string or array cannot be empty",
                    status: .badRequest,
                    param: "input"
                )
            }

            var embeddingDataList: [EmbeddingData] = []
            var totalTokensEstimate = 0

            for (index, text) in inputValues.enumerated() {
                let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
                if trimmed.isEmpty {
                    return Self.makeErrorResponse(
                        message: "Input text at index \(index) is empty",
                        status: .badRequest,
                        param: "input"
                    )
                }

                do {
                    let vector = try await self.engine.generateEmbedding(for: text)
                    embeddingDataList.append(EmbeddingData(index: index, embedding: vector))
                    totalTokensEstimate += text.count
                } catch let err as EmbeddingEngineError {
                    switch err {
                    case .sequenceLengthExceeded(let current, let max):
                        return Self.makeErrorResponse(
                            message: "Input at index \(index) exceeds maximum sequence length (\(current) > \(max))",
                            status: .badRequest,
                            param: "input"
                        )
                    case .emptyInput:
                        return Self.makeErrorResponse(
                            message: "Input text at index \(index) is empty",
                            status: .badRequest,
                            param: "input"
                        )
                    default:
                        return Self.makeErrorResponse(
                            message: err.description,
                            status: .internalServerError,
                            param: nil
                        )
                    }
                } catch {
                    return Self.makeErrorResponse(
                        message: "Internal engine error: \(error.localizedDescription)",
                        status: .internalServerError,
                        param: nil
                    )
                }
            }

            let responseModel = req.model ?? self.engine.alias.rawValue
            let usage = Usage(promptTokens: totalTokensEstimate, totalTokens: totalTokensEstimate)
            let responseObj = EmbeddingResponse(data: embeddingDataList, model: responseModel, usage: usage)

            let encoder = JSONEncoder()
            guard let responseData = try? encoder.encode(responseObj) else {
                return Self.makeErrorResponse(
                    message: "Failed to serialize response JSON",
                    status: .internalServerError,
                    param: nil
                )
            }

            return Response(
                status: .ok,
                headers: [.contentType: "application/json"],
                body: .init(byteBuffer: .init(data: responseData))
            )
        }

        return router
    }

    public func run() async throws {
        let router = buildRouter()
        let app = Application(
            router: router,
            configuration: .init(address: .hostname(host, port: port))
        )
        try await app.run()
    }

    private static func makeErrorResponse(
        message: String,
        status: HTTPResponse.Status,
        type: String = "invalid_request_error",
        param: String? = nil,
        code: String? = nil
    ) -> Response {
        let errorObj = OpenAIErrorResponse(
            error: OpenAIError(message: message, type: type, param: param, code: code)
        )
        let encoder = JSONEncoder()
        let data = (try? encoder.encode(errorObj)) ?? Data()
        return Response(
            status: status,
            headers: [.contentType: "application/json"],
            body: .init(byteBuffer: .init(data: data))
        )
    }
}
