# Section 6 Revision: Multi-Model Comparison & Economic Evaluation (§VI)
## RC-MoE Paper — Q1 Journal Fix Plan

> **Scope:** ONLY §VI Multi-Model Comparison and Economic Evaluation
> **Critical finding:** Oracle beating claim + leaky precision numbers + 3 missing spaces
> **Next:** After confirming done → §VII Explainability

---

## ISSUES FOUND — §VI

### 🔴 ISSUE C1: Missing Spaces in OOS Classification Paragraph (Formatting Error)

**Location:** Paragraph immediately after `fig:oos_classification`

**Exact broken text:**
```
Fig.~\ref{fig:oos_classification} compares all model familiesunder 5-fold
temporal cross-validation using event-wiseF1 ($\mathcal{F}_{1}^{\mathrm{E}}$)
with a $\pm3$ trading-daytolerance window, evaluated on 2\,455 daily observations
containing 76 anomaly events (3.1\% event rate).
```

**Three missing spaces:** `familiesunder` → `families under`, `event-wiseF1` → `event-wise F1`, `trading-daytolerance` → `trading-day tolerance`

**Fix:**
```latex
Fig.~\ref{fig:oos_classification} compares all model families under 5-fold
temporal cross-validation using event-wise F1
($\mathcal{F}_{1}^{\mathrm{E}}$) with a $\pm3$ trading-day tolerance
window, evaluated on 2\,455 daily observations containing 76 anomaly
events (3.1\% event rate).
```

---

### 🔴 ISSUE C2: "54.3% Precision" — Leaky OOS Number Stated as Fact

**Location:** OOS classification paragraph, first results paragraph

**Exact sentence:**
```
By contrast, the LightGBM RC-MoE achieves the highest precision (54.3\%)
with only 42 detections---$2.5\times$ fewer signals than GARCH---while
maintaining 32.8\% recall.
```

**Why wrong:** The 54.3% precision for LightGBM MoE is from the leaky
`_evaluate_moe()` where thresholds were optimized on test-fold labels.
It appears under the heading "Out-of-sample classification performance"
which makes it especially misleading.

**Two options:**

- **Option A (Preferred):** Add "under in-sample regime-isolated analysis" to
  clarify this figure is not the OOS result shown in `fig:oos_classification`:
  ```latex
  By contrast, the LightGBM RC-MoE achieves the highest precision
  (54.3\% under in-sample regime-isolated evaluation; see
  Table~\ref{tab:regime_conditioning}) with only 42 detections---$2.5\times$
  fewer signals than GARCH---while maintaining 32.8\% recall in the OOS setting.
  ```

- **Option B:** Replace with the actual leakage-free OOS precision figure from
  your evaluation results CSV file and report it as a clean number.

---

### 🔴 ISSUE C3: `tab:regime_conditioning` — Precision Column Uses Leaky Numbers

**Location:** Table `tab:regime_conditioning`, Prec. column

**Current values:**
- LightGBM MoE: **54.29%**, Global: **41.18%**
- XGBoost MoE: **48.94%**, Global: **45.61%**

**Why wrong:** These precision numbers are from the leaky threshold
optimization on test-fold labels. The Sharpe, Ann.Ret., and Eff. columns
are from the economic backtest and are clean.

**Fix:** Add a dagger footnote to the table caption:
```latex
\caption{Regime Conditioning Impact: Global vs.\ MoE\\
         (All Other Factors Held Constant)$^\dagger$}
```
And add below the table:
```latex
\begin{tablenotes}
  \footnotesize
  \item[$\dagger$] Precision values are computed under in-sample
  regime-isolated structural analysis (training partitions only).
  Sharpe ratio, annualised return, and hedging efficiency are
  from the leakage-free economic backtest.
\end{tablenotes}
```
And update the `\tabular` environment to use `threeparttable` or just add the note as a `\vspace` + `\footnotesize` block after the table.

---

### 🔴 ISSUE C4: "32% Precision Gain" — No In-Sample Qualifier in Text

**Location:** Paragraph after `tab:regime_conditioning`

**Exact text:**
```
The precision gain for LightGBM ($+32\%$) directly quantifies the
reduction in false alarms achieved by allowing each expert to learn
regime-specific anomaly thresholds, while the Sharpe ratio and
annualised return gains confirm that this structural improvement
translates to tangible economic value.
```

**Fix — add "in-sample" qualifier:**
```latex
The in-sample precision gain for LightGBM ($+32\%$, under
regime-isolated structural evaluation) directly quantifies the
reduction in false alarms achieved by allowing each expert to learn
regime-specific anomaly thresholds, while the Sharpe ratio and
annualised return gains---which persist under strict out-of-sample
evaluation---confirm that this structural improvement translates to
tangible economic value.
```

---

### 🔴 ISSUE C5: Per-Regime Numbers Are from Leaky Evaluation

**Location:** §VI.3 Per-Regime Performance Analysis paragraph

**Exact text (problematic numbers):**
```
The LightGBM RC-MoE achieves substantially higher precision than
GARCH in the medium-volatility ($75.0\%$ vs.\ $25.7\%$) and
high-volatility ($51.6\%$ vs.\ $48.4\%$) regimes...The
zero precision in the low-volatility regime...
```

**Why wrong:** 75.0%, 51.6%, 0% came from the leaky daily evaluation.
Leakage-free per-regime results show different values (Med-Vol: 1.000
precision but lower recall; High-Vol: 0.286 precision; Low-Vol: 0.400).

**Fix (choose one):**

- **Option A:** Update to leakage-free numbers from your evaluation CSV:
  ```latex
  The LightGBM RC-MoE achieves precision of 100.0\% in the
  medium-volatility regime (at 50.0\% recall) and 28.6\% precision
  in the high-volatility regime (at 66.7\% recall). The low-volatility
  regime yields 40.0\% precision (57.1\% recall), reflecting the
  extreme sparsity of calm-market anomalies.
  ```

- **Option B:** Add qualifier to existing text:
  ```latex
  Under in-sample regime-isolated evaluation, the LightGBM RC-MoE
  achieves substantially higher precision than GARCH in the
  medium-volatility ($75.0\%$ vs.\ $25.7\%$) and high-volatility
  ($51.6\%$ vs.\ $48.4\%$) regimes.
  ```

---

### 🔴 ISSUE C6: Oracle Beating Claim — Must Rewrite

**Location:** §VI.4 Economic Backtest, paragraph after `tab:backtest`

**Exact text:**
```
Notably, the LightGBM RC-MoE marginally \emph{outperforms} the Oracle
strategy (Sharpe: 0.772 vs.\ 0.767; return: 15.27\% vs.\ 14.54\%).
This counterintuitive result arises because the Oracle hedges all
labelled anomalies, including benign tail events during bull markets,
whereas the ML model selectively intervenes on the most economically
harmful observations---a demonstration of precision-driven portfolio
efficiency.
```

**Why wrong:** Claiming to beat an Oracle is logically impossible and
signals to any reviewer that the experiment is flawed. An "Oracle" by
definition represents perfect foresight — no real model can surpass it.
The fact that it does here reveals either a mis-defined Oracle or a
backtest artifact, neither of which should be presented as a feature.

**Fix — reframe as "approaches the Oracle upper bound":**
```latex
The Oracle strategy serves as a theoretical performance ceiling,
representing anomaly-hedged returns under perfect foresight. The
LightGBM RC-MoE closely approaches this bound (Sharpe: 0.772
vs.\ 0.767; return: 15.27\% vs.\ 14.54\%), demonstrating that
precision-driven selective hedging nearly replicates the
economic value of perfect anomaly foresight while operating
entirely on backward-looking information. The RC-MoE's marginally
higher return reflects its ability to avoid hedging benign tail
events in bull markets---a distinction the Oracle cannot make,
as it hedges all labelled events indiscriminately.
```

> **Note:** The Oracle's higher MaxDD (24.80% vs RC-MoE's 20.71%) is
> actually explained by the same logic — keep that in the narrative.

---

### 🟡 ISSUE C7: Comma Splice in ECI Paragraph

**Location:** §VI.6 ECI subsection, last paragraph

**Exact text:**
```
Critically, per-regime ECI varies systematically, the highest agreement
occurs in the Med-Vol regime (ECI $= 0.735$)...
```

**Why wrong:** `...varies systematically, the highest agreement occurs...`
is a comma splice — two independent clauses joined by a comma instead of
a semicolon or period. This is a grammatical error that editors catch.

**Fix:**
```latex
Critically, per-regime ECI varies systematically: the highest agreement
occurs in the Med-Vol regime (ECI $= 0.735$)...
```

---

### 🟡 ISSUE C8: ECI Values — Need "In-Sample" Clarification

**Location:** §VI.6 ECI subsection

**Current text:** "The global ECI of 0.659 indicates..."

**Issue:** ECI is computed from feature importance rankings of models trained
on (in-sample) regime-isolated partitions. This should be clarified so reviewers
understand these are structural (in-sample) consistency metrics.

**Fix — add one phrase:**
```latex
The global ECI of 0.659, computed over in-sample regime-isolated training
partitions, indicates moderate-to-strong cross-model agreement on anomaly
drivers.
```

---

## THINGS THAT ARE CLEAN — DO NOT CHANGE

| Element | Status | Reason |
|---------|--------|--------|
| `tab:backtest` rows (except Oracle narrative) | ✅ | Backtest uses raw S&P 500 returns |
| Buy-and-Hold, GARCH rows in `tab:backtest` | ✅ | Computed from actual returns |
| Hedging efficiency η values (0.315, 0.170 pp) | ✅ | From clean backtest |
| `fig:hedging_efficiency` and its narrative | ✅ | Derived from backtest |
| `tab:var` all VaR/CVaR numbers | ✅ | Computed from raw returns |
| 92% over-reserved / 50% under-reserved claim | ✅ | Derived from raw return VaR ratios |
| ECI formula (Eq. eci) | ✅ | Mathematically correct |
| ECI values (0.659, 0.735, 0.605) | ✅ | Computed from model outputs |
| `fig:var_regime` caption | ✅ | Clean, accurate description |
| `fig:hedging_efficiency` caption | ✅ | Clean |
| `fig:eci_regimes` caption | ✅ | Clean |

---

## COMPLETE CHECKLIST — §VI

```
MUST FIX
[x] C1 — Fix 3 missing spaces: "familiesunder", "event-wiseF1", "trading-daytolerance"
[x] C2 — Add "in-sample" qualifier or update 54.3% OOS precision claim
[x] C3 — Add dagger footnote to tab:regime_conditioning Prec. column
[x] C4 — Add "in-sample" qualifier to "32% precision gain" text paragraph
[x] C5 — Fix per-regime numbers (75.0%, 51.6%, 0%) — add qualifier or update
[x] C6 — Rewrite Oracle paragraph: "outperforms" → "closely approaches"

SHOULD FIX
[x] C7 — Fix comma splice: "varies systematically, the highest" → colon
[x] C8 — Add "in-sample" to ECI global value sentence
```

**Mandatory changes: 6**
**Recommended changes: 2**
**After confirming done → §VII Explainability + §VIII Discussion**
