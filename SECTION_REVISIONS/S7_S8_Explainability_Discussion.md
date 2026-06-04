# Section 7+8 Revision: Explainability (§VII) + Discussion (§VIII)
## RC-MoE Paper — Q1 Journal Fix Plan

> **Scope:** §VII Explainability and Economic Interpretation + §VIII Discussion
> **Critical finding:** fig:shap_regime caption directly contradicts tab:shap — factual error
> **Next:** After confirming done → §IX Limitations + §X Conclusion

---

## §VII EXPLAINABILITY — ISSUES FOUND

### 🔴 ISSUE X1: Caption of `fig:shap_regime` Contradicts `tab:shap` — Factual Error

**Location:** `\caption{}` for `fig:shap_regime`

**Current caption says:**
```
Trend momentum dominates low-volatility periods, whereas market
volatility, liquidity stress, and tail-risk skew become the primary
explanatory drivers during high-volatility regimes.
```

**But `tab:shap` says:**
| Regime | Key Driver |
|--------|-----------|
| Low Volatility | **Latent State C (Liquidity)** |
| Med Volatility | **Return Trend** |
| High Volatility | **Momentum Factor** |

**The contradiction:**
- Caption says "Trend momentum dominates **low-volatility**" — Table says Low-Vol driver is **Liquidity (Latent State C)**, not trend momentum
- Caption says "market volatility, liquidity stress... primary drivers during **high-volatility**" — Table says High-Vol driver is **Momentum Factor**

The caption describes the exact opposite regime mapping from what the table states. One of them is wrong. Assuming the table is correct (it has more detail):

**Fix — rewrite caption to match the table:**
```latex
\caption{Regime-conditioned SHAP feature attribution across volatility
environments (aggregated over in-sample training partitions). Liquidity
microstructure factors dominate low-volatility periods, sustained return
trend drives medium-volatility anomalies, and momentum collapse emerges
as the primary driver during high-volatility crisis episodes.}
```

---

### 🟡 ISSUE X2: Em-dash Typography — `--` Should Be `---`

**Location:** §VII.1 SHAP opening sentence

**Exact text:**
```
SHAP values are computed separately for each regime expert and aggregated
at the factor level--averaging mean absolute SHAP contributions...
```

**Problem:** `--` is an en-dash (used for ranges like 2016--2025). An em-dash `---` is required for a parenthetical break in a sentence.

**Fix:** Change `level--averaging` to `level---averaging`

---

### 🟡 ISSUE X3: "In-Regime Observations" Should Clarify Training-Only

**Location:** §VII.1 SHAP opening sentence

**Current text:**
```
...aggregated at the factor level---averaging mean absolute SHAP
contributions across all in-regime observations.
```

**Issue:** "In-regime observations" is ambiguous — does this include both train and test observations? SHAP values should only be computed on training data to avoid any data leakage in model explanations.

**Fix — one word addition:**
```latex
...aggregated at the factor level---averaging mean absolute SHAP
contributions across all in-regime \textbf{training} observations.
```

---

### 🟡 ISSUE X4: "Representative Decision Rule" — Is 0.42 Real or Illustrative?

**Location:** §VII.3 Interpretability vs. Deep Learning

**Current text:**
```
A representative decision rule derived from the High-Vol expert is:
\emph{``If realised volatility exceeds Q$_{66}$ and momentum falls
below 0.42, flag as anomalous.''}
```

**Issue:** Where does 0.42 come from? If this is an actual threshold extracted from the model, it needs a reference or explanation of how it was extracted. If it is illustrative/approximate, it should be labelled as such. Reviewers will ask: "how was this threshold derived?"

**Fix (two options):**
- **Option A (if real):** Add: *"...momentum falls below 0.42 (the 33rd percentile of momentum in the High-Vol regime), flag as anomalous."*
- **Option B (if illustrative):** Add: *"A representative (illustrative) decision rule derived from the High-Vol expert is: ..."*

---

### ✅ THINGS THAT ARE FINE — §VII

| Element | Status | Reason |
|---------|--------|--------|
| SHAP aggregation rationale | ✅ | "Instance-level unstable for rare events" is correct |
| `tab:shap` economic interpretations | ✅ | Aligned with financial theory |
| Economic Interpretation subsection | ✅ | Clean, theory-grounded |
| "no single global factor adequately describes..." | ✅ | Valid conclusion from ECI/SHAP |
| Last paragraph ("SHAP values are structural explanations...") | ✅ | Excellent — explicitly says not causal predictors |

---

## §VIII DISCUSSION — ISSUES FOUND

### 🔴 ISSUE D1: "32% Precision Improvement" — No In-Sample Qualifier

**Location:** §VIII.B Role of Regime Conditioning, last sentence

**Exact text:**
```
The ablation results (Section~\ref{sec:ablation}) provide direct
causal evidence: the transition from a global Random Forest to
RC-MoE---with no change to features, labels, or model
complexity---yields a 32\% precision improvement, isolating regime
conditioning as the primary performance driver.
```

**Why wrong:** This is the same issue as C4 in §VI. The 32% precision
improvement is an in-sample structural separability result. Without the
qualifier, it reads as a general OOS finding.

**Fix:**
```latex
The ablation results (Section~\ref{sec:ablation}) provide direct
structural evidence: the transition from a global Random Forest to
RC-MoE---with no change to features, labels, or model
complexity---yields a 32\% in-sample precision improvement under
regime-isolated evaluation, isolating regime conditioning as the
primary source of structural separability gains.
```

---

### 🟡 ISSUE D2: "With High Recall" — Imprecise Phrasing for 95.5% F1

**Location:** §VIII.A Structural Separability vs. Predictive Accuracy

**Current text:**
```
...factor representations separate extreme events with high recall,
approaching an F1 score of 95.5\% under the RC-MoE architecture.
```

**Issue:** The sentence says "with high recall" then immediately reports an F1 score. Recall and F1 are different metrics. The 95.5% is F1, not recall. Saying "high recall" then citing F1 will confuse readers and invite a reviewer to ask "what is the actual recall figure?"

**Fix — be precise:**
```latex
...factor representations separate extreme events with strong
discriminability, approaching an in-sample F1 score of 95.5\%
under the RC-MoE architecture.
```

---

### 🟡 ISSUE D3: Deep Learning F1 Comparison — In-Sample Qualifier Missing

**Location:** §VIII.C Implications for Deep Learning, first paragraph

**Current text:**
```
The ConvVAE and Anomaly Transformer variants...achieve substantially
lower F1 scores (64.0\% and 58.0\% respectively, vs.\ 95.5\% for RC-MoE).
```

**Issue:** All three F1 values (64%, 58%, 95.5%) are in-sample structural
separability scores from the ablation study. The Discussion section does not
explicitly say this. While context makes it partially clear, a standalone
reader of §VIII could misread these as OOS numbers.

**Fix — add two words:**
```latex
The ConvVAE and Anomaly Transformer variants...achieve substantially
lower \textbf{in-sample} F1 scores (64.0\% and 58.0\% respectively,
vs.\ 95.5\% for RC-MoE under regime-isolated structural evaluation).
```

---

### 🟡 ISSUE D4: Citation Inconsistency — `darban2024deep` vs `darban2023deep`

**Location:** §VIII.C, sentence about synthetic augmentation

**Current text:**
```
...severe class imbalance that deep architectures struggle to handle
without synthetic augmentation, which introduces fictitious tail
observations~\cite{darban2024deep}.
```

**Issue:** In §II Related Work, the same Darban et al. survey is cited as
`\cite{darban2023deep}`. Two different citation keys for the same author
suggests either:
1. Two different papers (need to verify), OR
2. A duplicate bib entry with different keys (will cause compilation warning
   or bibliography inconsistency)

**Fix:** Check `references.bib` — if same paper, unify to one key
(`darban2023deep` or `darban2024deep`, whichever matches the actual year).

---

### 🟡 ISSUE D5: "Robust Within-Regime Recall" — Vague and Overpositive

**Location:** §VIII.D Risk Management Applications, opening sentence

**Current text:**
```
The demonstrated hedging efficiency ($\eta = 0.315$~pp per detection)
and robust within-regime recall define three practical deployment
contexts for RC-MoE.
```

**Issue:** "Robust within-regime recall" is stated without a number and
without qualification. OOS per-regime recall varies substantially:
High-Vol 66.7%, Low-Vol 57.1%, Med-Vol 50.0% (leakage-free results).
Calling this "robust" without citation or number is vague.

**Fix — either add the actual recall range or replace with a precise claim:**
```latex
The demonstrated hedging efficiency ($\eta = 0.315$~pp per detection)
and structural within-regime separability (in-sample F1: 95.5\%)
define three practical deployment contexts for RC-MoE.
```

---

### ✅ THINGS THAT ARE FINE — §VIII

| Element | Status | Reason |
|---------|--------|--------|
| §VIII.A framing of in-sample vs OOS divergence | ✅ | Well-argued, honest |
| Two mechanisms explaining OOS degradation | ✅ | Scientifically sound |
| §VIII.B core argument about regime conditioning | ✅ | Correct reasoning |
| §VIII.C three structural properties of financial data | ✅ | Well-supported by literature |
| §VIII.D three deployment contexts | ✅ | Reasonable and practical |
| Hedging efficiency η = 0.315 pp | ✅ | From clean backtest |
| "3.2× fewer false interventions than GARCH" | ✅ | From clean backtest (19.6%/6.2%) |
| Basel III/IV compliance argument | ✅ | Based on clean VaR analysis |
| COVID-19 example for regime drift | ✅ | Correct illustrative example |

---

## COMPLETE CHECKLIST — §VII + §VIII

```
§VII EXPLAINABILITY
[x] X1 — Fix fig:shap_regime caption (contradicts tab:shap — CRITICAL)
          Low-Vol driver is Liquidity, not Trend Momentum
[x] X2 — Fix em-dash: "level--averaging" → "level---averaging"
[x] X3 — Add "training" to "in-regime observations"
[ ] X4 — Clarify whether 0.42 momentum threshold is real or illustrative

§VIII DISCUSSION
[x] D1 — Add "in-sample" qualifier to "32% precision improvement" in §VIII.B
[x] D2 — Replace "with high recall" → "with strong discriminability"
          for the 95.5% F1 sentence
[x] D3 — Add "in-sample" before F1 values in DL comparison paragraph
[x] D4 — Check references.bib: darban2024deep vs darban2023deep
[x] D5 — Replace "robust within-regime recall" with precise claim
```

**§VII mandatory changes: 3 (X1 is critical)**
**§VIII mandatory changes: 1, recommended: 4**
**After confirming done → §IX Limitations + §X Conclusion**
