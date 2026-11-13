# reports/

Strategy research reports, built from one version-controlled marimo template.

```
reports/
├── README.md                    tracked
├── templates/                   tracked
│   ├── strategy_report.py       the marimo notebook every report runs
│   └── configs/*.yaml           example report configs
└── <name>/                      NOT tracked (proprietary results)
    ├── config.yaml              what to test: strategy, universe, period, search
    ├── report.py                copy of the template
    ├── notes.md                 analyst conclusions, rendered in section 7
    ├── report.html              static export (self-contained)
    ├── tearsheet.html           Nautilus tearsheet of the validated OOS returns
    ├── optuna.db                screening + Nautilus-validation studies
    └── trials/                  per-trial out-of-sample returns (parquet)
```

```bash
algo report new tsmom-sp500 --from reports/templates/configs/tsmom_sp500.yaml
algo report build tsmom-sp500     # → reports/tsmom-sp500/report.html
algo report edit tsmom-sp500      # interactive marimo
algo report list
```

See `docs/guides/strategy_reports.md` for the full workflow.
