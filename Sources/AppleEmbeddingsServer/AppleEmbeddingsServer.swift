import Foundation
import ArgumentParser

@main
public struct AppleEmbeddingsServer: AsyncParsableCommand {
    public static let configuration = CommandConfiguration(
        commandName: "apple-embeddings-server",
        abstract: "Local embedding server and CLI powered by Apple's NaturalLanguage framework.",
        subcommands: [Embed.self, Serve.self],
        defaultSubcommand: nil
    )

    public init() {}
}

extension AppleEmbeddingsServer {
    struct Embed: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Generate a mean-pooled, L2-normalized float embedding vector for input text."
        )

        @Argument(help: "Text to generate embedding for.")
        var text: String?

        @Option(name: .shortAndLong, help: "Model alias to use (e.g. apple-nl-contextual-latin, apple-nl-contextual-cjk).")
        var model: String = ModelAlias.default.rawValue

        func run() async throws {
            let inputText: String
            if let text = text, !text.isEmpty {
                inputText = text
            } else {
                // Read from stdin to end of file if text is not provided as a positional argument
                let stdinData = (try? FileHandle.standardInput.readToEnd()) ?? Data()
                guard let stdinString = String(data: stdinData, encoding: .utf8),
                      !stdinString.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                    FileHandle.standardError.write(Data("Error: No input text provided via argument or stdin.\n".utf8))
                    throw ExitCode.failure
                }
                inputText = stdinString
            }

            guard let selectedAlias = ModelAlias(rawValue: model) else {
                FileHandle.standardError.write(Data("Error: Invalid model '\(model)'. Available models: \(ModelAlias.allCases.map { $0.rawValue }.joined(separator: ", "))\n".utf8))
                throw ExitCode.failure
            }

            do {
                let engine = try await EmbeddingEngine(alias: selectedAlias)
                let vector = try await engine.generateEmbedding(for: inputText)

                let encoder = JSONEncoder()
                let jsonBytes = try encoder.encode(vector)
                if let jsonString = String(data: jsonBytes, encoding: .utf8) {
                    print(jsonString)
                }
            } catch {
                FileHandle.standardError.write(Data("Error: \(error.localizedDescription)\n".utf8))
                throw ExitCode.failure
            }
        }
    }

    struct Serve: AsyncParsableCommand {
        static let configuration = CommandConfiguration(
            abstract: "Start the OpenAI-compatible HTTP embeddings server."
        )

        @Option(name: .shortAndLong, help: "Host address to bind the server.")
        var host: String = "127.0.0.1"

        @Option(name: .shortAndLong, help: "Port to bind the server.")
        var port: Int = 8080

        @Option(name: .shortAndLong, help: "Default model alias to host.")
        var model: String = ModelAlias.default.rawValue

        func run() async throws {
            guard let selectedAlias = ModelAlias(rawValue: model) else {
                FileHandle.standardError.write(Data("Error: Invalid model '\(model)'. Available models: \(ModelAlias.allCases.map { $0.rawValue }.joined(separator: ", "))\n".utf8))
                throw ExitCode.failure
            }

            FileHandle.standardError.write(Data("Initializing Apple NaturalLanguage embedding model '\(selectedAlias.rawValue)'...\n".utf8))
            let engine = try await EmbeddingEngine(alias: selectedAlias)
            let meta = await engine.getMetadata()

            FileHandle.standardError.write(Data("Model loaded successfully: identifier=\(meta.modelIdentifier), revision=\(meta.revision), dimension=\(meta.dimension), maxSeqLen=\(meta.maximumSequenceLength)\n".utf8))
            FileHandle.standardError.write(Data("Starting HTTP server on http://\(host):\(port)...\n".utf8))

            let server = EmbeddingServer(host: host, port: port, engine: engine)
            try await server.run()
        }
    }
}
