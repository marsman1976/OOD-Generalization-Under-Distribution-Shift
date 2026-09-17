# OOD Generalization Research — Complete Source Repository

Code associated with the study:

**Training-Data Composition, Augmentation, Model Configuration, and Generalization Under Distribution Shift**

This release candidate contains the recovered source code currently available for **Phases 1 through 7**, including the Phase 6 independent G2 generator/replication workflow and the Phase 7 real-data audit, reconstruction, OOD modeling, matched analysis, and final severity-sensitivity analysis.

## Why `src/` stays flat

The original project used a flat `src/` directory, project-root-relative paths, and imports such as `src.model` and `src.augmentation`. The GitHub package preserves that structure instead of reorganizing the scientific code and risking changes to behavior.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Run commands from the repository root. Example:

```bash
python -m src.run_experiment
```

See `docs/RUN_ORDER.md` for the phase-by-phase script map.

## Repository contents

- `src/` — complete recovered Python source tree for Phases 1–7
- `docs/RUN_ORDER.md` — phase-by-phase code map
- `docs/SCIENTIFIC_LOCK.md` — frozen interpretation safeguards
- `docs/VALIDATION.txt` — packaging-time syntax validation
- `data/README.md` — data/release guidance
- `results/README.md` — result-output guidance
- `requirements.txt`
- `.gitignore`

## Phase 7 data note

The real-data workflow expects the original HydroGrowNet files under the project's Phase 7 data path. Raw source data are not included in this ZIP. Redistribution should follow the original dataset license.

## Reproducibility before journal release

Before assigning a DOI or calling the repository a permanent reproducibility archive:

1. Run the full workflow in the final clean environment.
2. Freeze exact dependency versions from that environment.
3. Add the original dataset citation and license.
4. Add selected frozen result files/checksums used by the manuscript.
5. Add author metadata, a repository license, and `CITATION.cff`.
6. Tag the exact commit corresponding to the submitted manuscript.

## Scientific integrity

The repository preserves negative findings, partial replication, and sensitivity-qualified results. Do not alter the frozen analyses solely to strengthen statistical significance.


## Release-candidate additions
- `CITATION.cff` — citation metadata; permanent repository field remains a placeholder.
- `LICENSE_NOT_SELECTED.md` — software license still requires the author's choice.
- `REPRODUCIBILITY_STATUS.md` — completed versus outstanding validation.
- `data/DATASET_CITATION.md` — Phase 7 dataset citation, DOI, and license.
- `results/frozen_manuscript_evidence/` — recovered manuscript-facing evidence.
- `requirements-lock-packaging-environment.txt` — packaging environment versions.

Do not describe this release candidate as a fully verified end-to-end reproduction
until the clean-environment validation in `REPRODUCIBILITY_STATUS.md` is complete.
