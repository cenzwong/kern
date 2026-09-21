# apple-embeddings-server Instructions

> **Project Goal:** Expose Apple's on-device `NLContextualEmbedding` (`NaturalLanguage` framework) as an efficient CLI tool and OpenAI-compatible HTTP embeddings server.

## Architecture & Layout

```text
apple-embeddings-server/
├── Package.swift
├── README.md
├── Sources/
│   └── AppleEmbeddingsServer/
│       ├── AppleEmbeddingsServer.swift  # ArgumentParser CLI entry point (@main)
│       ├── EmbeddingEngine.swift        # Actor isolating NLContextualEmbedding
│       ├── Models.swift                 # ModelAlias, OpenAI DTOs & errors
│       └── Server.swift                 # Hummingbird 2.x HTTP router
└── Tests/
    └── AppleEmbeddingsServerTests/
        └── AppleEmbeddingsServerTests.swift  # Swift Testing suite
```

## Non-Negotiable Principles

- **Swift 6 Strict Concurrency**: Use actors, `Sendable`, and explicit isolation.
- **Dynamic Model Properties**: Never hardcode vector dimensions or model revisions; always query `embedding.dimension`, `embedding.modelIdentifier`, `embedding.revision`, and `embedding.maximumSequenceLength`.
- **Mean Pooling & L2 Normalization**: Aggregate subword token vectors with mean pooling and scale via Accelerate/`vDSP`.
- **OpenAI Compatibility**: Clean JSON structures for `/v1/embeddings`, `/v1/models`, and OpenAI-style error responses.

## Build & Test

```bash
swift build -c release
swift test
```
