You are an expert Systems and Swift Engineer. Build a high-performance, ultra-lightweight CLI and OpenAI-compatible HTTP server written in Swift (macOS 14+ / Sonoma or later) that leverages Apple's on-device `NLContextualEmbedding` (NaturalLanguage framework) running natively on the Apple Neural Engine (ANE).

### Objective
Create a single executable Swift package named `apple-embeddings-server` that can:
1. Run directly as a CLI to output embeddings for a given input string.
2. Run as an OpenAI-compatible HTTP server hosting the `/v1/embeddings` endpoint.

---

### Technical Requirements & Constraints

1. **Framework & Dependencies:**
   - Swift 5.10+ (macOS 14.0+).
   - Use `apple/swift-argument-parser` for CLI flag and sub-command routing.
   - Use `hummingbird-project/hummingbird` (v2.x preferred for lightweight, zero-dependency async HTTP routing without the heavy footprint of Vapor).
   - Use Apple's native `NaturalLanguage` framework (`NLContextualEmbedding`).

2. **Core Embedding Engine (`EmbeddingEngine.swift`):**
   - Initialize `NLContextualEmbedding(language: .undetermined)` or support language-specific models.
   - Ensure `embedding.hasAvailableAssets` check is performed; if not, invoke `embedding.requestAssets` with proper async/await or completion handling.
   - Implement a method: `generateEmbedding(for text: String) throws -> [Double]`.
   - Apple's `NLContextualEmbedding` outputs contextual token vectors (dimension is 512). Implement mean pooling across the token vectors to yield a single fixed-size 512-dimension vector representing the entire sentence/chunk.
   - Normalize the final pooled vector (L2 norm) so it matches standard cosine similarity expectations.

3. **OpenAI Compatibility Layer (`/v1/embeddings`):**
   - Implement standard OpenAI JSON payload structures with `Codable`:
     - **Request:**
       ```json
       {
         "input": "string or array of strings",
         "model": "apple-nl-contextual",
         "encoding_format": "float"
       }
       ```
       *Handle `input` flexibly: it can be either a single string or an array of strings.*
     - **Response:**
       ```json
       {
         "object": "list",
         "data": [
           {
             "object": "embedding",
             "index": 0,
             "embedding": [-0.0123, 0.0456, "..."]
           }
         ],
         "model": "apple-nl-contextual",
         "usage": {
           "prompt_tokens": 0,
           "total_tokens": 0
         }
       }
       ```
   - Health check endpoint: `GET /health` returning `{"status": "ok"}`.

4. **CLI Architecture (`apple-embeddings-server`):**
   - Root command with two subcommands:
     - `embed "<text>"`: Prints the raw JSON float array directly to stdout (useful for piping into `jq` or shell scripts).
     - `serve [--port 8080] [--host 127.0.0.1]`: Starts the HTTP server.

5. **Performance & Reliability:**
   - Ensure the `NLContextualEmbedding` instance is reused across requests (do not reinitialize the model per HTTP request to ensure zero cold-start penalty).
   - Thread-safe or actor-isolated embedding inference if accessed concurrently across HTTP requests.

---

### Deliverables Expected:
1. `Package.swift` with exact dependencies configured.
2. Complete, idiomatic Swift source files organized cleanly:
   - `Sources/EmbeddingEngine.swift`
   - `Sources/Models.swift` (OpenAI DTOs)
   - `Sources/Server.swift` (Hummingbird HTTP router)
   - `Sources/main.swift` (ArgumentParser entry point)
3. Step-by-step build & test instructions using `swift build -c release` and a verification `curl` command.