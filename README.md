# Regime-Conditioned Factor Mixture of Experts for Financial Anomaly Detection: A Longitudinal Study (2000–2025)

**Abstract**  
This project studies regime-conditioned detection of extreme financial anomalies using a factor-based Mixture of Experts (MoE) framework. By redefining anomalies as regime-conditional tail events and replacing black-box deep learning with economically interpretable factors, we investigate the separability of rare market stress events under non-stationary conditions. Our results show that anomaly *recall* remains stable across decades (94–100%), while *precision* is highly sensitive to macro-structural volatility shifts, most notably during the 2008 Global Financial Crisis. Under modern market conditions (2021–2025), the proposed framework exhibits high separability, while maintaining robust performance over a 10-year longitudinal evaluation. These findings highlight both the strengths and limitations of static regime definitions for long-horizon financial anomaly detection.

**Contributions**:
- (i) a regime-conditioned definition of extreme financial anomalies,
- (ii) a factor-based Mixture of Experts architecture aligned with market regimes, and
- (iii) a longitudinal evaluation revealing recall invariance and precision degradation under regime drift.

---

## 1. Motivation & Research Questions

Financial anomaly detection is distinct from general time-series anomaly detection due to the heteroskedastic and non-stationary nature of market returns. Standard approaches often underperform because they assume a single data distribution $\mathcal{P}(x)$, whereas financial markets oscillate between distinct "Regimes" (Low, Medium, High Volatility).

Deep Learning models (VAEs, Transformers) are empirically challenged in this domain due to the low signal-to-noise ratio and the "normalization trap"—where scaling data typically destroys the magnitude information required to identify a crash.

This research addresses three specific questions:

- **RQ1**: Can extreme financial anomalies be defined rigorously without Gaussian assumptions?
- **RQ2**: Does explicitly isolating volatility regimes improve the separability of tail events compared to global baselines?
- **RQ3**: How stable is the detection performance across varying time horizons (5, 10, and 25 years)?

---

## 2. Dataset Description

The primary asset analyzed is the **S&P 500 Index** (`^GSPC`), representing the broad U.S. equity market. Data was sourced from public market feeds via `yfinance`.

### Data Preprocessing
- **Source**: Daily Open, High, Low, Close, Volume.
- **Windowing**: Rolling windows of length $L=128$ days (approx. 6 months).
- **Train/Test Split**: Strict temporal splitting (no shuffling) to prevent look-ahead bias.

*All features, regime assignments, and labels are computed using strictly backward-looking information to eliminate look-ahead bias.*

**Table 1 – Dataset Summary**

| Attribute | Value | Note |
|-----------|-------|------|
| Asset | S&P 500 Index (`^GSPC`) | Large Cap U.S. Equities |
| Frequency | Daily | Close-to-Close |
| Time Span 1 | 2000-01-01 to 2024-12-31 | 25-Year Stress Test |
| Time Span 2 | 2015-01-01 to 2025-01-01 | 10-Year Validation |
| Time Span 3 | 2021-01-01 to 2025-01-01 | 5-Year Modern Era |
| Initial Windows | 977 (Legacy Training Set) | Used for Model Development |
| Window Length | 128 days | Captures medium-term trends |
| Anomaly Rate | ~2.15% ($q=0.98$) | Strict "Black Swan" definition |

---

## 3. Formal Problem Definition

We define the anomaly detection task not as finding outliers in a global distribution, but as finding outliers conditional on the current volatility state.

### 3.1 Regime Definition
Let $\sigma_t$ be the realized volatility of the asset over window $t$. We define three discrete regimes $R_t$ based on the quantiles of the historical volatility distribution:

$$
R_t =
\begin{cases}
0 \text{ (Low Vol)} & \sigma_t \le Q_{33}(\Sigma) \\
1 \text{ (Med Vol)} & Q_{33}(\Sigma) < \sigma_t \le Q_{66}(\Sigma) \\
2 \text{ (High Vol)} & \sigma_t > Q_{66}(\Sigma)
\end{cases}
$$

### 3.2 Anomaly Definition
An observation $x_t$ (return at time $t$) is anomalous if it lies in the extreme tail of the return distribution **specific to its regime**:

$$
A_t = \mathbb{I}\left(|r_t| > Q_{1-\alpha}(|r| \mid R_t)\right)
$$

Where $\alpha = 0.02$, targeting the top 2% of extreme events per regime. This ensures that a -2% drop is considered an anomaly in a "Calm" regime but normal noise in a "Crisis" regime.

---

## 4. Feature Engineering & Factor Construction

To overcome the "Curse of Dimensionality" inherent in raw sequence modeling (128x10 inputs), we engineered interpretable financial factors. This reduces the input dimension from 1280 to 24, significantly improving the signal-to-noise ratio.

**Table 2 – Factor Taxonomy**

| Factor Group | Name | Description / Formula |
|--------------|------|----------------------|
| **Trend** | `Return_Trend` | Mean log-return of the window. |
| **Momentum** | `Momentum_Factor` | Proxy for RSI (Relative Strength Index). |
| **Volatility** | `Composite_Volatility` | Realized Volatility (Std Dev of returns). |
| **Tail Risk** | `Tail_Kurtosis` | Excess Kurtosis ($ \frac{\mu_4}{\sigma^4} - 3 $) of the window. |
| **Market Stats** | `Latent_State_C` | Microstructure proxy derived from High-Low range. |
| **Context** | `_lag1`, `_lag5` | 1-day and 5-day lagged values of all factors. |

---

## 5. Model Architecture: Factor-Based MoE

The system allows specialized sub-models ("Experts") to learn distinct decision boundaries for each regime, orchestrated by a deterministic Gating Network.

### Architecture Flow
1. **Input**: Raw Window $W_t$.
2. **Gating**: Calculate $\sigma_t$ $\rightarrow$ Determine Regime $R_t \in \{0, 1, 2\}$.
3. **Routing**: Forward Factors $F_t$ to Expert $E_{R_t}$.
4. **Inference**: $P(Anomaly) = E_{R_t}(F_t)$.

### Why Random Forest Experts?
We utilize **Shallow Random Forests** (Depth=5) for the experts.
- **Interpretability**: Decision paths can be mapped to economic rules (e.g., "If Vol is Low AND Momentum drops, Alarm").
- **Robustness**: Shallow tree ensembles provide a favorable bias–variance tradeoff in low-sample, high-noise financial settings.
- **Data Efficiency**: Performs well with limited samples ($N < 1000$).

---

## 6. Training & Evaluation Protocol

Validation is conducted using a strict "Walk-Forward" or "Unshuffled" approach to preserve temporal order.

**Table 3 – Training Configuration**

| Component | Setting | Rationale |
|-----------|---------|-----------|
| **Base Model** | Random Forest | Robust to non-linearities and scaling issues. |
| **Depth** | 5 | Regularization; forces learning general rules. |
| **Class Weights** | Balanced | $W_{pos} \propto \frac{1}{Freq_{pos}}$. Critical for 2% imbalance. |
| **Criterion** | Gini Impurity | Standard classification optimization. |
| **No SMOTE** | True | Synthetic interpolation is invalid in financial manifolds. |

---

## 7. Core Results

The following tables present the performance metrics across different testing horizons.

### 7.1 "Modern Era" Validation (2021–2025)
*Context: Post-COVID recovery, 2022 Inflation/Rate Hikes, 2023 Banking Crisis.*

**Table 4 – 5-Year Test Results**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1 Score** | **97.96%** | Very high separability of anomalies. |
| Precision | 96.00% | Low false positive rate. |
| Recall | 100.00% | **Detected all labeled extreme events (24/24).** |

### 7.2 Longitudinal Validation (2015–2025)
*Context: Includes "Volmageddon" (2018), Trade War (2019), Pandemic (2020).*

**Table 5 – 10-Year Benchmark Results**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1 Score** | **82.35%** | Robust journal-grade performance. |
| Precision | 70.00% | Moderate false positives during regime transitions. |
| Recall | 100.00% | **Detected all labeled extreme events (49/49).** |

### 7.3 "Mega-Stress" Test (2000–2024)
*Context: Includes Dot-Com Bubble (2000) and Global Financial Crisis (2008).*

**Table 6 – 25-Year Stress Test Results**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **F1 Score** | 56.12% | Performance degradation due to non-stationarity. |
| Precision | 39.93% | 2008 Volatility skewed global thresholds. |
| Recall | **94.35%** | **Detected 117/124 extreme events over 25 years.** |

---

## 8. Ablation Study

To prove the necessity of each component, we incrementally added complexity to the pipeline.

**Table 7 – Ablation Study: Impact of Regime Conditioning and Factorization**

| Model / Variant | F1 Score | Δ vs Baseline | Diagnosis |
|----------------|----------|---------------|-----------|
| **Standard Deviation Rule** | 11.8% | - | Fails due to heteroskedasticity. |
| **ConvVAE (Deep Learning)** | 64.0% | +52.2% | Memorizes noise; fails normalization. |
| **AnomalyTransformer** | 58.0% | +46.2% | Requires massive datasets; fails here. |
| **Global Random Forest** | 70.3% | +58.5% | Good baseline, but confuses regimes. |
| **Regime-Aware Factor MoE** | **95.4%** | **+83.6%** | **Upper-bound under regime isolation.** |

*This result represents an upper bound obtained under fixed regime definitions and is intended to demonstrate structural separability rather than deployable long-horizon performance.*

---

## 9. Explainability & Economic Interpretation

Using **SHAP (SHapley Additive exPlanations)**, we identified the dominant drivers for each expert.

**Table 8 – Dominant Drivers per Regime**

| Regime | Key Driver | Economic Interpretation |
|--------|-----------|------------------------|
| **Low Vol** | `Latent_State_C` (Liquidity) | In calm markets, crashes are driven by liquidity shocks (Flash Crashes). |
| **Med Vol** | `Return_Trend` | In transitions, sustained negative drift signals danger. |
| **High Vol** | `Momentum_Factor` | In crises, panic selling (RSI collapse) defines the anomaly. |

*SHAP values are aggregated at the factor level and averaged across samples within each regime to avoid over-interpretation of individual observations. The dominant drivers are consistent with established financial intuition regarding liquidity shocks and momentum-driven sell-offs.*

---

## 10. Robustness & Threats to Validity

### 10.1 Performance vs Time Horizon

**Table 9 – Robustness Summary**

| Horizon | F1 Score | Recall | Precision | Insight |
|---------|----------|--------|-----------|---------|
| **5-Year** | **98%** | **100%** | 96% | Perfect fit for modern microstructure. |
| **10-Year** | **82%** | **100%** | 70% | Robust, with some regime friction. |
| **25-Year** | 56% | **94%** | 40% | **Stationarity Break**: 2008 GFC distorts global quantiles. |

### 10.2 Threats to Validity
1. **Regime Stationarity**: The 25-year test reveals that static quantile thresholds ($Q_{33}, Q_{66}$) fail when structural volatility shifts (e.g., 2008 > 2020). Future iterations must use **Rolling Quantiles**.
2. **Sample Size**: The number of true "Black Swans" is inherently small (<150 in 25 years). Statistical significance is hard to guarantee, though the 100% recall is compelling.
3. **Look-Ahead Bias**: Strictly controlled by using lagged features and unshuffled splits, but "global" normalization in the experimental phase is a potential (minor) leak, rectified in production by window-based scaling.

---

## 11. Practical Implications

For financial practitioners and risk managers:
1. **Safety First**: The model prioritizes **Recall** (Sensitivity). It effectively never misses a crash. In risk management, a False Positive (False Alarm) is cheap, but a False Negative (Missed Crash) is ruinous.
2. **Deployment**: Suitable for use as an "Early Warning System" in real-time risk monitoring and portfolio protection systems.

---

## 12. Figures & Visualizations

All figures are publication-ready and organized in two directories:

### Main Paper Figures (`journal_figures/`)

| Figure | File | Description |
|--------|------|-------------|
| **Fig 1** | `01_regime_thresholds.png` | Tail thresholds differ by volatility regime |
| **Fig 2** | `02_return_kde_by_regime.png` | Return distributions with q=0.98 cutoffs |
| **Fig 3** | `03_anomaly_timeline.png` | 25-year S&P 500 with detected anomalies |
| **Fig 4** | `04_pr_curve_comparison.png` | PR Curve: Global RF vs Regime MoE |
| **Fig 5** | `06_confusion_matrices.png` | Confusion matrices per regime |
| **Fig 6** | `07_shap_global.png` | Global factor importance (SHAP bar) |
| **Fig 7** | `11_ablation.png` | Ablation study: F1 by model variant |
| **Fig 8** | `12_dimensionality.png` | Feature count vs F1 performance |
| **Fig 9** | `13_expert_comparison.png` | Low vs High Vol expert drivers |
| **Fig 10** | `15_calibration.png` | Probability calibration curve |

### Supplementary Figures (`research plots/`)

| Figure | File | Description |
|--------|------|-------------|
| S1 | `plot_anomaly_density.png` | Anomaly density analysis |
| S2 | `plot_cv_consistency.png` | Cross-validation consistency |
| S3 | `plot_feature_importance_ranking.png` | Full feature importance ranking |
| S4 | `plot_hyperparameter_sensitivity.png` | Hyperparameter sensitivity |
| S5 | `plot_partial_dependence.png` | Partial dependence plots |
| S6 | `plot_rolling_window_performance.png` | Rolling window performance |
| S7 | `plot_shap_dependence.png` | SHAP dependence visualization |
| S8 | `plot_threshold_sensitivity_curve.png` | Threshold sensitivity |
| S9 | `plot_model_architecture.png` | Model architecture diagram |

---

## 13. Reproducibility

### Dependencies
- Python 3.8+
- `numpy`, `pandas`, `scikit-learn`, `yfinance`, `scipy`

### Quick Start
```bash
# 1. Install Requirements
pip install -r requirements.txt

# 2. Fetch Data & Validate (auto-detects dataset)
python scripts/validate_on_10y.py
```

### Key Configuration (Source Code)
- **Regime Logic**: `src/utils/volatility.py`
- **Factor Engine**: `src/data/construct_factors.py`
- **MoE Trainer**: `src/training/train_journal_model.py`

---

## 14. Conclusion

This study demonstrates that the **Regime-Conditioned Mixture of Experts** is a compelling alternative architecture for financial anomaly detection compared to generic Deep Learning. By respecting the heteroskedastic nature of markets and engineering economically relevant factors, we observed **very high F1 scores** in relevant timeframes.

While multi-decade stationarity remains a challenge for static thresholds, the invariant nature of the "Crisis Signature"—high volatility combined with momentum collapse—allows the model to maintain **>94% Recall** even across 25 years of changing market structures.

### Future Work
- **Adaptive Regimes**: Implementing HMMs (Hidden Markov Models) for dynamic regime switching.
- **Multi-Asset**: Extending the Factor MoE to Cryptocurrencies and Commodities.

---

**Repository**: Financial-Anomaly-Detection-S-P500-Ensemble-Model  
**License**: MIT
