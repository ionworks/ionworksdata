# Changelog — ionworksdata

All notable changes to this package are documented here. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this package follows [Semantic Versioning](https://semver.org/).

For platform-wide release notes (Studio, pipeline, SDK, and more),
see [docs.ionworks.com/changelog](https://docs.ionworks.com/changelog).

<!-- New release sections are prepended below by the release-packages skill. -->

## [0.15.0] - 2026-09-04

### Added
- Maccor header metadata is read and surfaced, rather than parsed for the test
  date and discarded. `Maccor.read_metadata()` returns the parsed header, the
  new module-level `read.metadata(filename, reader, options)` reaches it with
  reader auto-detection, and `measurement_details()` merges the keys into the
  measurement dict callers already build before upload. Most useful is
  `protocol["name"]`, from the file's `Procedure:` line, which identifies the
  schedule that produced a file and so can link a measurement to its protocol
  on upload; the export date, source path, source file id, and barcode comment
  land under `test_setup`. Merging uses `setdefault`, so a value the caller set
  is never overwritten by the header. `BaseReader.read_metadata()` reports the
  reader name and start time, so any reader can be asked without branching on
  reader type.
- Maccor columns recording the cycler's own execution are carried through
  instead of dropped, each named with the existing `... from cycler` suffix:
  `Mode from cycler` (`MD`), `End status from cycler` (`ES`), `Loop N from
  cycler` (`Loop1`..`Loop4`), and `VARn`/`FLAGn from cycler` (the `SetVar`
  registers). All are optional — a file without them reads as before. `MD` was
  previously mapped only to an internal working column that is dropped after
  the unsigned-current correction, so the mode never reached a caller; it is
  the only thing distinguishing Maccor's step-transition markers from genuine
  zero-current rest samples, and `VAR1` commonly holds the per-cell capacity
  the procedure used as its C-rate basis, which nothing else in the export
  records.

## [0.14.0] - 2026-08-31

### Added
- A Digatron CSV reader. `Time [s]` is derived from the file's `Timestamp` at
  microsecond resolution rather than from the vendor duration columns, whose
  `#s` header suffix can disagree with the values actually written. A
  cumulative duration column whose span disagrees with the wall clock is
  reported; pass `strict_duration_units=True` to refuse the file instead.
- Arbin electrochemical impedance sweeps are no longer dropped. `read.time_series()`
  on an Arbin workbook now splices the sweeps from the `ACIM_*` sheet into the
  cycling data in test-time order, as their own step with a null `Time [s]`,
  `Voltage [V]`, and `Current [A]` — a sweep is measured against frequency, so
  it carries no timestamp to record. Pass `options={"include_eis": False}` for
  the cycling data alone.
- Maccor exports that report current in mA and capacity/energy in mAHr/mWHr are
  read, with the values converted to A and A.h/W.h. Previously these files
  failed on a missing `Current [A]` column, and renaming the column by hand
  produced values 1000x too large.
- Maccor files carry the cycler's own DC internal-resistance column through to
  the output instead of dropping it.
- The CSV reader tolerates two structural quirks that are common across cyclers
  rather than specific to one vendor: a metadata preamble above the real header
  (each line a label padded out to the table width), and data rows carrying more
  fields than the header names — a trailing delimiter on every row, say. The
  header is located rather than assumed to be the first line, and the surplus
  fields are dropped before columns are matched.

### Changed
- Maccor header matching is canonical rather than literal: bracket style
  (`[]` vs `()`), `-`/`_`/space separators, padding around the unit, and unit
  spellings (`AHr`/`ah`, `WHr`/`wh`, `Sec`/`s`) are folded before lookup. Every
  header spelling that was recognised before still resolves, and punctuation
  variants of a known column now resolve without needing a new mapping.
- Worksheet selection is consistent across the Arbin, Neware, and Maccor Excel
  readers: each confirms a candidate sheet by its headers rather than trusting
  a fixed name or the first sheet.
- When several source columns map to the same output column, the winner is
  chosen by one shared rule across the readers instead of four private ones.
  No reader's output changes.
- Requires pybamm 26.8.0.0 or newer.

### Fixed
- Arbin `.xlsx` workbooks read the `Channel_*` data sheet instead of the leading
  `Global_Info` metadata sheet, which had made most multi-sheet Arbin workbooks
  unreadable.
- Arbin headers in the `Test_Time(s)` dialect no longer lose their units to a
  unit-stripper that did not fold underscores — this had dropped the time axis
  itself, leaving only voltage and current.
- Maccor bracketed-unit headers (`Test Time [s]`, `Current [A]`) are mapped
  instead of discarded, and a comma-separated header that carries its own units
  no longer has its first measurement row eaten as a units row.
- BioLogic exports written in a decimal-comma locale (`3,6955268E+000`) parse
  instead of failing before the first row.
- BaSyTec `.txt` result exports are supported, including latin1-encoded headers
  such as `T1[°C]` that previously failed as invalid UTF-8.
- Novonix `-9999` no-sensor sentinels are read as missing rather than as a real
  temperature.
- Neware exports whose vendor time column resets mid-file keep a monotonic
  `Time [s]`.

## [0.13.0] - 2026-08-25

### Added
- `DataLoader.from_db()` accepts a `time_range` window,
  `{"start": seconds, "end": seconds}`, carried through into `to_config()`.
  Bounds are elapsed `Time [s]` values from the first sample, not wall-clock
  datetimes, so the window stays meaningful for a measurement with no recorded
  start time and does not drift when that metadata is edited.

  It pins how much of a measurement a pipeline run uses. A measurement that is
  still being extended grows over time, so re-running an unpinned config
  silently reads more data than the first run did; a pinned config keeps
  re-runs comparable. The window is applied when the config is resolved for a
  run — a local `from_db()` still loads the full current series.

## [0.12.0] - 2026-08-11

### Breaking changes
- `DataLoader` now rejects an unrecognised option instead of ignoring it. The
  accepted set is closed, so a misspelt option can no longer silently skip the
  preprocessing you asked for — it raises at construction. Code that passed a
  stray or misspelt key and relied on it being dropped must remove it or
  correct the spelling.

### Added
- `DataLoader` accepts a `DataLoaderOptions` schema object wherever it accepts
  an options mapping, so a caller holding one no longer has to unwrap it by
  hand before merging it with keyword options.

### Changed
- `interpolate` is documented as taking a float or a list of floats; a numpy
  array is still accepted and normalised to a list.
- `first_step` / `last_step` are documented as taking a step index, a query
  selecting one row of the steps table, or the deprecated cycle/step dict.
- Added a dependency on `ionworks-schema>=0.18.0`, which now owns the
  validated definition of the loader's option surface.

## [0.11.6] - 2026-08-03

### Changed
- Raised the `fastexcel` lower bound to `>=0.20.2`.

## [0.11.5] - 2026-07-29

### Changed
- Raised the `polars` lower bound to `>=1.43.0`.

## [0.11.4] - 2026-07-28

### Changed
- `Power [W]` is now computed during raw→parquet processing rather than
  derived on every read. A new `set_power()` transform mirrors the existing
  `set_capacity`/`set_energy` pattern, computing
  `Power [W] = Voltage [V] * Current [A]` and inheriting the current's sign.
  It is a no-op when `Power [W]` is already present or the source columns are
  missing, so previously processed measurements are unaffected.

### Fixed
- The current-sign auto-flip retry path now drops and recomputes `Power [W]`
  alongside the capacity and energy columns; previously power silently kept
  the pre-flip sign.

## [0.11.3] - 2026-07-24

### Changed
- Migrated to numpy 2 and pandas 3.
- Bumped pybamm to 26.7.1.0 and adapted UCP power-mode simulations and
  `FunctionForExport` serialization accordingly.

### Fixed
- Incremented the numpy lower pin.

## [0.11.2] - 2026-07-20

### Changed
- Raised the ``polars`` dependency floor from ``>=1.33.1`` to
  ``>=1.42.1`` (#1284).

## [0.11.1] - 2026-07-10

### Fixed
- BioLogic EIS reader: flip the reported ``-Im(Z)`` into the canonical
  ``Z_Im [Ohm]`` via an intermediate column instead of negating in place,
  avoiding a column-name collision during the rename step (#1133).

## [0.11.0] - 2026-07-08

### Breaking changes
- ``PiecewiseLinearTimeseries`` is now an abstract base class. Build a
  piecewise-linear timeseries via the new ``piecewise_linear_timeseries()``
  factory (or the concrete ``PiecewiseLinearTimeseriesCompressed`` /
  ``PiecewiseLinearTimeseriesLossless`` subclasses); constructing the base
  class directly now raises ``NotImplementedError`` (#1104).

### Added
- ``piecewise_linear_timeseries`` factory for constructing a piecewise-linear
  timeseries, with a lossless variant that reproduces the input current
  samples exactly instead of compressing them within the tolerances (#1104).

### Fixed
- Neware BTSDA workbooks now auto-select the ``record`` sheet when the
  active sheet is not the data sheet (#1113).
- Maccor exports now read the cycle number from ``"Cycle C"`` rather than
  ``"Cycle P"`` (#1112).

## [0.10.1] - 2026-06-26

### Fixed
- Corrected the documented Neware column mappings: ``"Current(A)"`` maps
  to ``"Current [A]"`` (not ``"Current [mA]"``) (#969).

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
