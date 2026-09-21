# TODO

- [x] Create Swift package structure for `apple-embeddings-server` (`swift-tools-version: 6.1`).
- [x] Implement core model aliases and OpenAI DTOs in `Models.swift`.
- [x] Implement actor-isolated `EmbeddingEngine` with mean pooling, L2 normalization (`vDSP`), asset loading, and sequence length checks.
- [x] Implement Hummingbird 2.x HTTP server with `/health`, `/metadata`, `/v1/models`, and `/v1/embeddings`.
- [x] Implement CLI command using `ArgumentParser` with `embed` and `serve` subcommands.
- [x] Implement unit and integration test suite using Swift Testing (`import Testing`).
- [x] Create comprehensive `README.md` and update `CLAUDE.md`.
