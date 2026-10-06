# Verification evidence

## Historical local verification — 5 October 2026

At this pre-publication checkpoint, the complete workflow executed on real public source files; actual reports and plots were generated. The executed notebook's stored outputs contain no errors. Ruff lint/format, meaningful unit tests and dependency consistency passed.

A clean local Git clone was installed in a separate Python 3.12 study environment. The source acquisition and complete workflow ran from that clone without copied models or raw sources. Source scientific content was verified; generated JSON reports match exactly and tabular outputs reproduce within atol=rtol=1e-10. Clone tests, lint, notebook validation and pip check passed. This environment is shared by the four final studies with their scoped dependencies; it is not a dedicated fresh environment for each study.

All results are retrospective and limited to the documented source population. The repository has since been [published](https://github.com/idrisslemnouni-crypto/seed-variety-classification); see its [GitHub Actions history](https://github.com/idrisslemnouni-crypto/seed-variety-classification/actions) for remote results by revision. No independent field validation, production deployment or invented metrics are claimed. Raw source files and any derived database/model are saved locally and ignored in Git. Source archives include code, documentation and actual reports, with acquisition commands to reconstruct ignored files.

## Input-contract verification — 6 October 2026

Added inclusive marginal feature ranges calculated from the training partition only. The trusted local artifact was upgraded without fitting after checking the pinned source checksum/audit (13,611 source rows), all recorded split IDs/classes (9,480/2,031/2,032), experiment/artifact metadata and SVM support vectors against the ordered training inputs. A backup remains ignored at `models/selected.before-support.joblib`. The estimator's joblib hash, class order, calibration temperature and every example probability remain exactly unchanged; the only added artifact key is `training_feature_ranges`.

Nine local tests pass, including training/holdout range separation, inclusive boundaries, extrapolation feature listing, legacy metadata fallback, unchanged predictions, multiclass score/probability batch invariance, and rejection of a mismatched recorded split before touching an artifact. Ruff lint/format and Git whitespace checks pass. No retraining or revised test metric was performed. Marginal range membership cannot establish joint geometry support, independent batch validity or unknown-variety rejection; the original test evidence remains historically inspected. These local checks do not assert a remote CI result for the new changes; consult Actions for each pushed revision.
