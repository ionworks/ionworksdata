# Changelog — ionworksdata

All notable changes to this package are documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package follows [Semantic Versioning](https://semver.org/).

For platform-wide release notes (Studio, pipeline, SDK, and more),
see [docs.ionworks.com/changelog](https://docs.ionworks.com/changelog).

<!-- New release sections are prepended below by the release-packages skill. -->

## [0.10.0] - 2026-06-10

### Added
- ``CycleAgeing`` objective now supports ``experiment='from data'``,
  deriving the cycling experiment directly from the measured data
  instead of requiring it to be specified separately (#834).

### Changed
- The Arbin ``.res`` reader now bounds each ``mdb-export`` call with a
  300-second timeout, raising a clear ``RuntimeError`` instead of
  hanging on a corrupt or locked MDB file (#806).

### Fixed
- Corrected CC-discharge step mislabeling (#807) and an unsigned
  mixed-mode current sign error (#810) in step identification (#848).

## [0.9.3] - 2026-06-05

### Changed
- Canonical CSV column detection now warns when two columns collapse to
  the same whitespace-stripped key, making it clear that the later column
  shadows the earlier one (#718).

## [0.9.2] - 2026-06-01

### Changed
- Switched the ``polars`` dependency from ``polars-lts-cpu`` to the
  standard ``polars`` distribution (#768).

## [0.9.1] - 2026-05-29

### Changed
- Relaxed the ``numpy`` dependency bound to allow ``numpy>=2`` (#754).

## [0.9.0] - 2026-05-22

### Added
- Generic parquet reader (``ionworksdata.read.parquet``) that mirrors
  the existing CSV ingestion path: format auto-detection picks it
  up via ``from_path`` / ``from_file`` for ``.parquet`` inputs (#697).

### Changed
- Capacity and energy are now integrated with a per-step reset that
  matches the columns reported by the platform, instead of a single
  cumulative integral. Values inside a step are unchanged; the
  cumulative totals across resets will differ from previous
  releases (#687).

### Fixed
- Coin-cell ingestion now applies the correct current sign and
  capacity convention; previously some coin-cell sources were
  imported with flipped charge/discharge labels (#697).

## [0.8.0] - 2026-05-11

### Breaking changes
- `MeasurementValidationError.errors` is now `list[ValidationIssue]`
  (a frozen dataclass with stable `CheckName`, `severity`, `message`,
  and structured `payload`) instead of `list[str]`. Downstream
  callers that string-matched error messages should switch to
  `e.has_check(CheckName.X)` (#544).

### Changed
- `ionworksdata.read` auto-fix now keys off
  `e.has_check(CheckName.CURRENT_SIGN_CONVENTION)` rather than
  substring-matching the human-readable message (#544).

## [0.7.0] - 2026-04-30

### Added
- Arbin CSV/XLSX/RES reader.
- Maccor reader supports compact short-form column headers.

### Fixed
- `cycle-metrics` keeps the "Cycle count" column name in its output.
