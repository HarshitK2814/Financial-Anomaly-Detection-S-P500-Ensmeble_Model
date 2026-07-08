# Explainable Regime-Conditioned Mixture of Experts for Financial Anomaly Detection

Code accompanying the paper *"Explainable Regime Conditioned Mixture of
Experts for Financial Anomaly Detection, Risk Capital Calibration, and Cross
Market Validation"* (Hemrajani, Kumar, Kumar, Shekhar, Himthani, Mahalik,
Bohra — Manipal University Jaipur / Bennett University).

## Abstract

Conventional financial anomaly detection relies on static threshold models
that ignore volatility regime dynamics, causing substantial mis-calibration
of capital across varied market conditions. This work proposes a
regime-conditioned mixture of experts (RC-MoE) architecture that partitions
the market into volatility regimes via realised-volatility quantiles and
trains individual experts per regime on eight interpretable features
spanning trend, momentum, volatility, market state, and tail risk over a
128-day lookback. Experiments are conducted on 2,513 daily observations from
the S&P 500 (US) over 9.7 years (2016-2025) under a leakage-free protocol,
and extended to NASDAQ (US), Russell 2000 (US), and Bitcoin (Global). Key
findings: (1) unconditional VaR₉₉ fails Kupiec coverage in high-volatility
regimes while overcharging capital in stable ones, across all four markets;
(2) RC-MoE reduces signal frequency by 3.2× relative to GARCH (6.2% vs.
19.6% of trading days), improving hedging efficiency by 85%. Factor
regression confirms a low market beta (0.63) and near-zero alpha, and
Explainability Consistency Index (ECI) bootstrapping shows anomaly drivers
shift structurally across regimes.

## Repository structure

```
src/
  data/          Data fetching (yfinance), factor construction, preprocessing, dataset loaders
  models/        Expert models: LightGBM/XGBoost/RF experts, LSTM/GRU attention, hybrid,
                 regime detector, meta-ensemble, adaptive gating
  training/      Training entrypoints for the journal model / multi-model / SOTA baselines
  evaluation/    Statistical tests, ablations, ECI validation, EVT/Basel capital calibration,
                 factor-alpha (Fama-French) tests, multi-market detection pipelines
  baselines/     Isolation Forest and PCA baselines
  xai/           Explainability: tree/deep explainers, regime explainer, unified reporting
  utils/         Seeding, logging, volatility utilities, figure generation scripts

run_daily_pipeline.py            End-to-end daily pipeline (fetch -> factors -> train -> evaluate)
run_complete_evaluation.py       Full walk-forward evaluation
run_leakage_free_evaluation.py   Leakage-free walk-forward evaluation protocol
run_sensitivity.py               L (window length) x K (number of experts) sensitivity sweep

final.tex                        Manuscript source
final_figures/, kronos_style_figures/, correct figures/   Paper figures
backend/, frontend/              Interactive dashboard demo (see dashboard_demo.zip)
```

## Data

Raw market data (S&P 500, NASDAQ, Russell 2000, Bitcoin OHLCV) is fetched at
runtime via the [`yfinance`](https://pypi.org/project/yfinance/) Python
package. Fama-French risk-factor data is sourced from the
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).

The derived/engineered dataset — rolling-window features, factor channels,
trained expert models, and every value backing the paper's reported means,
standard deviations, tables, and figures — is archived separately with a
persistent DOI:

**Replication dataset**: https://doi.org/10.5281/zenodo.21257024 (CC-BY 4.0)

## Setup

```bash
conda env create -f environment.yml   # or: pip install -r requirements.txt
```

## Reproducing results

```bash
python run_daily_pipeline.py              # regenerate factors and per-regime experts
python run_leakage_free_evaluation.py      # walk-forward leakage-free evaluation
python run_sensitivity.py                  # L x K sensitivity sweep (Section 6.4)
python src/evaluation/eci_validation.py    # bootstrap-validated ECI per regime
python src/evaluation/factor_alpha.py      # Fama-French 5 + Momentum alpha test
python src/evaluation/multimarket_eventwise_canonical.py   # cross-market event-wise detection
```

## Citation

```bibtex
@article{hemrajani2026rcmoe,
  title   = {Explainable Regime Conditioned Mixture of Experts for Financial
             Anomaly Detection, Risk Capital Calibration, and Cross Market
             Validation},
  author  = {Hemrajani, Prashant and Kumar, Harshit and Kumar, Sarthak and
             Shekhar, Shashank and Himthani, Varsha and Mahalik, Sayan and
             Bohra, Manoj Kumar},
  year    = {2026}
}
```

Please also cite the replication dataset (DOI above) if you reuse the data.

## License

Code is released under the [MIT License](LICENSE). The replication dataset
is released separately under CC-BY 4.0.
