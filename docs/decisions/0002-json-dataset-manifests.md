# 2. Dataset manifests are JSON, not YAML

**Status:** accepted · **Date:** 2026-08-13

## Context

TRD §26 sketches the dataset manifest in YAML. Every other machine-readable artifact in this project is JSON, validated against a versioned JSON Schema before it is written and after it is read.

Supporting YAML would mean either adding a parser dependency for one file type, or maintaining two serialisation paths for the same validated object.

## Decision

Dataset manifests are JSON, validated against `schemas/dataset.schema.json` like every other artifact.

## Alternatives

- **PyYAML for manifests.** A dependency, a second parse path, and a format whose implicit typing rules (the Norway problem, sexagesimal literals, unquoted version strings becoming floats) are a poor fit for a file whose entire purpose is exact identity.
- **Both formats.** Two code paths, two sets of failure modes, and a question at every review about which one the manifest in hand uses.

## Consequences

- Manifests reuse the schema machinery: closed objects, checksum patterns enforced at the schema level, one validation error format.
- Manifests are less pleasant to hand-write than YAML. Acceptable: they are written once per dataset and are mostly digests, which nobody types by hand anyway.
- TRD §26 diverges from the implementation and should be corrected to JSON.

## Validation

`tests/test_datasets.py` loads, validates and rejects manifests through the same schema path as every other artifact; a manifest missing a checksum fails schema validation rather than being accepted and failing later.
