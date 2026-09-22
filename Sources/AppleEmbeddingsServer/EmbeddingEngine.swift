import Foundation
import NaturalLanguage
import Accelerate

public enum EmbeddingEngineError: Error, CustomStringConvertible, Sendable {
    case modelNotFound(String)
    case assetDownloadFailed(String)
    case emptyInput
    case sequenceLengthExceeded(current: Int, max: Int)
    case inferenceFailed(String)

    public var description: String {
        switch self {
        case .modelNotFound(let model):
            return "Unsupported or unavailable model: '\(model)'"
        case .assetDownloadFailed(let model):
            return "Failed to download assets for model: '\(model)'"
        case .emptyInput:
            return "Input text cannot be empty"
        case .sequenceLengthExceeded(let current, let max):
            return "Input sequence length (\(current)) exceeds model maximum sequence length (\(max))"
        case .inferenceFailed(let reason):
            return "Embedding inference failed: \(reason)"
        }
    }
}

/// Actor-isolated engine that manages NLContextualEmbedding models, asset requests,
/// mean pooling, L2 normalization, and metadata inspection.
public actor EmbeddingEngine {
    private let embeddingModel: NLContextualEmbedding
    public let alias: ModelAlias
    public let dimension: Int
    public let modelIdentifier: String
    public let revision: Int
    public let maximumSequenceLength: Int

    public init(alias: ModelAlias = .latin) async throws {
        self.alias = alias

        guard let model = NLContextualEmbedding(script: alias.script) else {
            throw EmbeddingEngineError.modelNotFound(alias.rawValue)
        }

        self.embeddingModel = model

        if !model.hasAvailableAssets {
            let assetResult = try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<NLContextualEmbedding.AssetsResult, Error>) in
                model.requestAssets { result, error in
                    if let error = error {
                        continuation.resume(throwing: error)
                    } else {
                        continuation.resume(returning: result)
                    }
                }
            }

            guard assetResult == .available else {
                throw EmbeddingEngineError.assetDownloadFailed(alias.rawValue)
            }
        }

        try model.load()

        self.dimension = model.dimension
        self.modelIdentifier = model.modelIdentifier
        self.revision = model.revision
        self.maximumSequenceLength = model.maximumSequenceLength
    }

    /// Internal initializer for testing or custom wrapped instances where NLContextualEmbedding mock/instance is supplied.
    internal init(embeddingModel: NLContextualEmbedding, alias: ModelAlias = .latin) throws {
        self.alias = alias
        self.embeddingModel = embeddingModel
        if !embeddingModel.hasAvailableAssets {
            // Assets assumption
        }
        try? embeddingModel.load()

        self.dimension = embeddingModel.dimension
        self.modelIdentifier = embeddingModel.modelIdentifier
        self.revision = embeddingModel.revision
        self.maximumSequenceLength = embeddingModel.maximumSequenceLength
    }

    /// Helper vector processing routines for mean pooling and L2 normalization using Accelerate/vDSP.
    public static func meanPoolAndL2Normalize(tokenVectors: [[Double]], expectedDimension: Int) throws -> [Double] {
        guard !tokenVectors.isEmpty else {
            throw EmbeddingEngineError.emptyInput
        }

        let count = Double(tokenVectors.count)
        var meanVector = [Double](repeating: 0.0, count: expectedDimension)

        for vec in tokenVectors {
            guard vec.count == expectedDimension else {
                throw EmbeddingEngineError.inferenceFailed("Token vector dimension mismatch. Expected \(expectedDimension), got \(vec.count)")
            }
            vDSP.add(vec, meanVector, result: &meanVector)
        }

        // Divide by vector count to get mean
        let scale = 1.0 / count
        meanVector = vDSP.multiply(scale, meanVector)

        // Compute L2 Norm: sqrt(sum(v_i^2))
        let sumSquares = vDSP.sumOfSquares(meanVector)
        let norm = sqrt(sumSquares)

        if norm > 0 {
            let invNorm = 1.0 / norm
            meanVector = vDSP.multiply(invNorm, meanVector)
        }

        return meanVector
    }

    /// Generates a normalized mean-pooled embedding vector for a given input text.
    public func generateEmbedding(for text: String) throws -> [Double] {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            throw EmbeddingEngineError.emptyInput
        }

        if text.count > maximumSequenceLength {
            throw EmbeddingEngineError.sequenceLengthExceeded(current: text.count, max: maximumSequenceLength)
        }

        let result = try embeddingModel.embeddingResult(for: text, language: nil)

        var tokenVectors: [[Double]] = []

        result.enumerateTokenVectors(in: text.startIndex..<text.endIndex) { vector, range in
            tokenVectors.append(vector)
            return true
        }

        guard !tokenVectors.isEmpty else {
            throw EmbeddingEngineError.inferenceFailed("No token vectors generated for input")
        }

        return try Self.meanPoolAndL2Normalize(tokenVectors: tokenVectors, expectedDimension: dimension)
    }

    /// Metadata summary structure for inspection
    public func getMetadata() -> ModelMetadata {
        return ModelMetadata(
            alias: alias.rawValue,
            modelIdentifier: modelIdentifier,
            revision: revision,
            dimension: dimension,
            maximumSequenceLength: maximumSequenceLength
        )
    }
}
