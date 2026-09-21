import Foundation
import NaturalLanguage

/// Supported Apple Contextual Embedding Model Aliases
public enum ModelAlias: String, Codable, CaseIterable, Sendable {
    case latin = "apple-nl-contextual-latin"
    case cyrillic = "apple-nl-contextual-cyrillic"
    case cjk = "apple-nl-contextual-cjk"
    case arabic = "apple-nl-contextual-arabic"
    case indic = "apple-nl-contextual-indic"
    case thai = "apple-nl-contextual-thai"

    public static let `default`: ModelAlias = .latin

    /// Maps model alias to Apple NLLanguage
    public var language: NLLanguage {
        switch self {
        case .latin:
            return .english
        case .cyrillic:
            return .russian
        case .cjk:
            return .simplifiedChinese
        case .arabic:
            return .arabic
        case .indic:
            return .hindi
        case .thai:
            return .thai
        }
    }

    /// Maps model alias to Apple NLScript
    public var script: NLScript {
        switch self {
        case .latin:
            return .latin
        case .cyrillic:
            return .cyrillic
        case .cjk:
            return .simplifiedHan
        case .arabic:
            return .arabic
        case .indic:
            return .devanagari
        case .thai:
            return .thai
        }
    }
}

// MARK: - OpenAI Request Models

public struct EmbeddingRequest: Decodable, Sendable {
    public let input: Input
    public let model: String?
    public let encodingFormat: String?

    public enum Input: Decodable, Sendable {
        case single(String)
        case array([String])

        public var values: [String] {
            switch self {
            case .single(let str):
                return [str]
            case .array(let arr):
                return arr
            }
        }

        public init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            if let str = try? container.decode(String.self) {
                self = .single(str)
            } else if let arr = try? container.decode([String].self) {
                self = .array(arr)
            } else {
                throw DecodingError.typeMismatch(
                    Input.self,
                    DecodingError.Context(
                        codingPath: decoder.codingPath,
                        debugDescription: "Expected String or Array of Strings for 'input'"
                    )
                )
            }
        }
    }

    enum CodingKeys: String, CodingKey {
        case input
        case model
        case encodingFormat = "encoding_format"
    }
}

// MARK: - OpenAI Response Models

public struct EmbeddingResponse: Encodable, Sendable {
    public let object: String = "list"
    public let data: [EmbeddingData]
    public let model: String
    public let usage: Usage

    public init(data: [EmbeddingData], model: String, usage: Usage) {
        self.data = data
        self.model = model
        self.usage = usage
    }
}

public struct EmbeddingData: Encodable, Sendable {
    public let object: String = "embedding"
    public let index: Int
    public let embedding: [Double]

    public init(index: Int, embedding: [Double]) {
        self.index = index
        self.embedding = embedding
    }
}

public struct Usage: Encodable, Sendable {
    public let promptTokens: Int
    public let totalTokens: Int

    enum CodingKeys: String, CodingKey {
        case promptTokens = "prompt_tokens"
        case totalTokens = "total_tokens"
    }

    public init(promptTokens: Int, totalTokens: Int) {
        self.promptTokens = promptTokens
        self.totalTokens = totalTokens
    }
}

// MARK: - Model Discovery Models

public struct ModelListResponse: Encodable, Sendable {
    public let object: String = "list"
    public let data: [ModelObject]

    public init(data: [ModelObject]) {
        self.data = data
    }
}

public struct ModelObject: Encodable, Sendable {
    public let id: String
    public let object: String = "model"
    public let created: Int = 1710000000
    public let ownedBy: String = "apple-natural-language"
    public let modelIdentifier: String?
    public let revision: Int?
    public let dimension: Int?
    public let maximumSequenceLength: Int?

    enum CodingKeys: String, CodingKey {
        case id
        case object
        case created
        case ownedBy = "owned_by"
        case modelIdentifier = "model_identifier"
        case revision
        case dimension
        case maximumSequenceLength = "maximum_sequence_length"
    }

    public init(
        id: String,
        modelIdentifier: String? = nil,
        revision: Int? = nil,
        dimension: Int? = nil,
        maximumSequenceLength: Int? = nil
    ) {
        self.id = id
        self.modelIdentifier = modelIdentifier
        self.revision = revision
        self.dimension = dimension
        self.maximumSequenceLength = maximumSequenceLength
    }
}

// MARK: - Model Metadata

public struct ModelMetadata: Encodable, Sendable {
    public let alias: String
    public let modelIdentifier: String
    public let revision: Int
    public let dimension: Int
    public let maximumSequenceLength: Int

    enum CodingKeys: String, CodingKey {
        case alias
        case modelIdentifier = "model_identifier"
        case revision
        case dimension
        case maximumSequenceLength = "maximum_sequence_length"
    }

    public init(
        alias: String,
        modelIdentifier: String,
        revision: Int,
        dimension: Int,
        maximumSequenceLength: Int
    ) {
        self.alias = alias
        self.modelIdentifier = modelIdentifier
        self.revision = revision
        self.dimension = dimension
        self.maximumSequenceLength = maximumSequenceLength
    }
}

// MARK: - OpenAI Error Models

public struct OpenAIErrorResponse: Encodable, Sendable {
    public let error: OpenAIError

    public init(error: OpenAIError) {
        self.error = error
    }
}

public struct OpenAIError: Encodable, Sendable {
    public let message: String
    public let type: String
    public let param: String?
    public let code: String?

    public init(message: String, type: String = "invalid_request_error", param: String? = nil, code: String? = nil) {
        self.message = message
        self.type = type
        self.param = param
        self.code = code
    }
}
