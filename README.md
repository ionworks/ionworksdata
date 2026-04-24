# Ionworks Data Processing

> **This is a read-only mirror.** The source of truth is a private repo.

A library for processing experimental battery data into the common format used across Ionworks software. `ionworksdata` reads files from common battery cyclers (BaSyTec, BioLogic, Maccor, Neware, Novonix, Repower, generic CSV, BDF), normalizes units and sign conventions, and summarizes cycling data into step- and cycle-level tables.

## Installation

```bash
pip install ionworksdata
```

## Quick example

```python
import ionworksdata as iwd

# Reader is auto-detected from the file
data = iwd.read.time_series("path/to/file.mpt")
```

## Documentation

- **Preparing data for Ionworks Studio**: [docs.ionworks.com/data/preparing-data](https://docs.ionworks.com/data/preparing-data) — supported cyclers, custom column mappings, BDF, and troubleshooting.
- **API reference**: [data.docs.ionworks.com](https://data.docs.ionworks.com/) — complete reference for `read`, `write`, `transform`, `steps`, and `load`.
- **Changelog**: [docs.ionworks.com/changelog](https://docs.ionworks.com/changelog) — release notes across the platform and Python packages.

## Reporting issues

If a cycler file doesn't process correctly, please [open an issue](https://github.com/ionworks/ionworksdata/issues) with a **minimal working example**:

- **The data file** that reproduces the problem (anonymize it first if it contains anything sensitive — trimming to the smallest excerpt that still reproduces is ideal).
- **What you tried** — the exact `ionworksdata` call and any options you used.
- **The full error or incorrect output** you got.

By attaching a data file you grant us permission to use it in our regression test suite so the fix stays fixed. If you can't share the file under those terms, please include the smallest possible synthetic reproducer instead.
