# Reproducibility status

## Present in this release candidate
- Recovered source files for Phases 1–7 are present in `src/`.
- Phase 6 includes the independent G2 generator/replication workflow.
- Phase 7 includes audit, reconstruction/linkage, held-out-domain modeling,
  matched-reference analysis, and severity-sensitivity analysis.
- Selected frozen manuscript-facing reports/tables are under
  `results/frozen_manuscript_evidence/`.
- Raw Phase 7 data are not redistributed; verified citation/provenance are documented.
- Scientific interpretation safeguards are in `docs/SCIENTIFIC_LOCK.md`.

## Validation completed
- Every Python source file was syntax-compiled.
- Packaging-wrapper marker checks were run.
- Checksums were regenerated after assembly.

## Not yet claimed
This release candidate has NOT been certified as a clean-environment,
end-to-end reproduction of every numerical manuscript result. Some workflows
require raw or generated data not bundled here. The dependency lock records the
packaging environment, not necessarily the historical experiment environment.

Before a permanent DOI: run the workflow in a clean environment, retain logs,
compare manuscript-facing outputs with frozen evidence, select a software
license, and replace repository/DOI placeholders.
