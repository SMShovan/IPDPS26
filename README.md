# IPDPS Workspace

This workspace contains multiple related projects used for dynamic/temporal hypergraph motif benchmarking:

- `dynamicHyperGraph`
- `temporalDynamicHMotif`
- `THyMe-main`
- `cuhornet-main`
- `MoCHy-master`

## Project Structure

Each project keeps its own build/run documentation and `Makefile`:

- `dynamicHyperGraph/Makefile`
- `dynamicHyperGraph/docs/README.md`
- `temporalDynamicHMotif/Makefile`
- `temporalDynamicHMotif/README.md`

This top-level directory adds a wrapper `Makefile` for convenience.

## Top-Level Commands

From this directory:

```bash
make help
make build-all
make clean-all
```

### Dynamic project wrappers

```bash
make dynamic-build
make dynamic-run
make dynamic-figures
```

### Temporal project wrappers

```bash
make temporal-build
make temporal-run
make temporal-figures
```

`temporal-figures` runs figure pipelines for:

- figure12a
- figure12b
- figure13
- figure14
- figure15

