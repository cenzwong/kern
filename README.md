# apple-embeddings-server

A lightweight, high-performance CLI tool and OpenAI-compatible HTTP server written in Swift that leverages Apple's on-device `NLContextualEmbedding` (`NaturalLanguage` framework) on macOS (14.0+ / Sonoma or later).

## Key Features

- **100% Native & On-Device**: Uses Apple's `NLContextualEmbedding` framework natively on macOS.
- **Dynamic Dimension & Metadata**: Never hard-codes vector dimensions or revisions. Reads runtime `embedding.dimension`, `embedding.modelIdentifier`, `embedding.revision`, and `embedding.maximumSequenceLength`.
- **Mean Pooling & L2 Normalization**: Combines subword/token contextual vectors using Accelerate (`vDSP`) for high performance and scales the resulting vector to unit length (L2 norm = 1.0).
- **OpenAI-Compatible `/v1/embeddings` API**: Drop-in replacement for OpenAI embeddings API clients. Supports single strings or arrays of strings, with strict order preservation.
- **Actor-Isolated Architecture**: Thread-safe inference using Swift 6 actors and strict concurrency.
- **Model Aliases**: Exposes stable, script/language-mapped model aliases (`apple-nl-contextual-latin`, `apple-nl-contextual-cjk`, `apple-nl-contextual-cyrillic`, etc.).
- **Oversized Input Safeguard**: Validates text length against `maximumSequenceLength` and returns OpenAI-formatted 400 Bad Request errors for oversized inputs.

---

## Repository Structure

```text
apple-embeddings-server/
├── Package.swift
├── README.md
├── TODO.md
├── CLAUDE.md
├── AGENTS.md
├── Sources/
│   └── AppleEmbeddingsServer/
│       ├── AppleEmbeddingsServer.swift  # Command line interface (@main)
│       ├── EmbeddingEngine.swift        # Core actor wrapping NLContextualEmbedding
│       ├── Models.swift                 # DTOs, OpenAI API models, ModelAlias
│       └── Server.swift                 # Hummingbird 2.x HTTP router & handlers
└── Tests/
    └── AppleEmbeddingsServerTests/
        └── AppleEmbeddingsServerTests.swift  # Swift Testing test suite
```

---

## Usage

### 1. CLI Commands

#### Generate Embeddings directly (`embed`)

Outputs raw JSON array of float values to `stdout`. Diagnostics and errors go to `stderr`.

```bash
# Default Latin model
apple-embeddings-server embed "Hello world"

# CJK model
apple-embeddings-server embed --model apple-nl-contextual-cjk "你好世界"

# Pipe input via stdin
echo "Search query from pipeline" | apple-embeddings-server embed
```

#### Run HTTP Server (`serve`)

Starts the OpenAI-compatible HTTP server using Hummingbird 2.x.

```bash
apple-embeddings-server serve --host 127.0.0.1 --port 8080 --model apple-nl-contextual-latin
```

---

### 2. HTTP Endpoints

#### `GET /health`
Returns server health status.

```bash
curl http://127.0.0.1:8080/health
# {"status":"ok"}
```

#### `GET /metadata`
Returns metadata of the active model instance (dimension, model identifier, revision, max sequence length).

```bash
curl http://127.0.0.1:8080/metadata
```

#### `GET /v1/models`
Returns list of available model aliases in OpenAI format.

```bash
curl http://127.0.0.1:8080/v1/models
```

#### `POST /v1/embeddings`
Generates embeddings for a single string or an array of strings.

```bash
curl http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": ["Hello world", "Apple Neural Engine"],
    "model": "apple-nl-contextual-latin",
    "encoding_format": "float"
  }'
```

---

## Model Revision & Vector Database Best Practices

When persisting embedding vectors into a vector database (e.g. SQLite, LanceDB, Qdrant):
1. Save `model_identifier`, `revision`, and `dimension` along with your vector index.
2. Vectors generated across different Apple system revisions (e.g., Revision 1 vs Revision 2) are in different vector spaces and should not be directly compared.

---

## Building and Testing

### Requirements
- macOS 14.0+ (Sonoma or later)
- Swift 6.1+ / Xcode 16+

### Build

```bash
swift build -c release
```

### Run Tests

```bash
swift test
```

---

## License

MIT License
