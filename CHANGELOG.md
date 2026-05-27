# Changelog — ionworksdata

All notable changes to this package are documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package follows [Semantic Versioning](https://semver.org/).

For platform-wide release notes (Studio, pipeline, SDK, and more),
see [docs.ionworks.com/changelog](https://docs.ionworks.com/changelog).

<!-- New release sections are prepended below by the release-packages skill. -->

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
