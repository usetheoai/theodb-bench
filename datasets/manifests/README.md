# Dataset manifests

One JSON file per dataset, validated against `schemas/dataset.schema.json`.

A dataset is identified by its **checksums**, never by a filename. The runner
refuses to measure over files it has not verified.

```
theodb-bench dataset list
theodb-bench dataset verify <id>
theodb-bench dataset fetch <id>
```

## Adding a manifest

1. Download the files once, by hand, from the publisher.
2. Compute `sha256sum` for each file.
3. Write the manifest with those digests, the licence, and the source URL.
4. Run `theodb-bench dataset verify <id>` against your local copy.

**No manifest may be committed with a checksum that was not computed from the
actual bytes.** A guessed digest turns verification into theatre: every future
fetch would fail, and the first person to "fix" it would do so by copying
whatever they happened to download.

`license.redistributable` records whether we may mirror the bytes. When it is
false the manifest points at the publisher and nothing is cached anywhere else.

This directory ships empty of real datasets: the checksums for the public ANN
and retrieval corpora have to be produced by whoever first fetches them, and
inventing them here would defeat the mechanism.
