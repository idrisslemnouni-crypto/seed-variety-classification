# Local verification — 5 October 2026

The complete workflow executed on real public source files; actual reports and plots were generated. The executed notebook's stored outputs contain no errors. Ruff lint/format, meaningful unit tests and dependency consistency passed.

A clean local Git clone was installed in a separate Python 3.12 study environment. The source acquisition and complete workflow ran from that clone without copied models or raw sources. Source scientific content was verified; generated JSON reports match exactly and tabular outputs reproduce within atol=rtol=1e-10. Clone tests, lint, notebook validation and pip check passed. This environment is shared by the four final studies with their scoped dependencies; it is not a dedicated fresh environment for each study.

All results are retrospective and limited to the documented source population. Public GitHub CI is pending the scheduled publication day; no independent field validation, production deployment or invented metrics are claimed. Raw source files and any derived database/model are saved locally and ignored in Git. Source archives include code, documentation and actual reports, with acquisition commands to reconstruct ignored files.
