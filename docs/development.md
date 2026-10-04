# Development

Set `OCRUS_DATA_DIR` to choose the location of generated evaluation images, results, and fonts. Relative paths are resolved against the repository root; the default is the repository root. Character lists and `test_fonts.yml` remain in the repository, and execution logs are written to `logs/`.

Use `mise.toml` for tool versions and the Makefile for local and CI commands.

```bash
make setup      # Install the toolchain and dependencies
make ci         # Run formatting checks, clippy, compile checks and tests
make build      # Build all crates
make release    # Build the CLI release binary
make fmt        # Format the Rust code
make bench      # Run benchmarks
```

`make test` sets the Python library search path on macOS. Model-dependent tests skip when models are absent; passing unit tests alone does not verify OCR accuracy.

Build the Python wheel, install it and pytest into the test Python environment, and run tests from the repository root:

```bash
make wheel
make pytest
```

Running pytest from inside `python/` makes the source package shadow the installed wheel. Keep `pyo3/extension-module` out of the crate's default features so workspace tests can link against Python.

The OCR smoke test needs the model and takes about 20 seconds after a release build:

```bash
make smoke
```

Training, model conversion and large accuracy tests run separately. See [training](training.md) for those commands and [the maintenance record (Japanese)](maintenance-2026-10-04.md) for earlier validation results.
