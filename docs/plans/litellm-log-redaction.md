# LiteLLM Log Redaction

## Plan

- [x] Reproduce the API key disclosure in LiteLLM request logs.
- [x] Add recursive credential redaction to standard and streaming logs.
- [x] Preserve the original arguments sent to providers.
- [x] Cover API keys, authorization headers, AWS credentials, Google API keys,
  Vertex service-account credentials, and nested token or secret fields.
- [x] Run focused tests, formatting, lint, type checks, and the full test suite.
- [x] Ask independent agents to review code conventions and documentation impact.

## Review

The helper stays private to the LiteLLM adapter because this pull request fixes
the observed LiteLLM request-log disclosure. A shared sanitizer for other model
adapters can be a separate change with adapter-specific tests.

The redactor uses credential-related key segments and keeps pricing fields that
contain `cost`. It replaces a complete credential value with `[redacted]`, so a
Vertex service-account JSON string is not parsed or partially exposed. Tests
also verify that providers receive the original values.

Validation passed after the review update:

- `uv run pytest packages/litellm/tests/test_client.py -q`: 7 tests.
- `/usr/bin/make lint`.
- `/usr/bin/make typecheck`.
- `/usr/bin/make tests`: 1,018 Python tests and 84 TypeScript tests.
