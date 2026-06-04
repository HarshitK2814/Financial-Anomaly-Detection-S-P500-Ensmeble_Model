# Section 9+10 Revision: Limitations (§IX) + Conclusion (§X)
## RC-MoE Paper — Q1 Journal Fix Plan

> **Scope:** §IX Limitations + §X Conclusion
> **Critical finding:** Event count inconsistency (49 vs 76), wrong Sharpe % in Conclusion, future work lists already-implemented feature
> **Next:** All sections done — proceed to applying changes in the .tex file

---

## §IX LIMITATIONS — ISSUES FOUND

### 🔴 ISSUE L1: Event Count Inconsistency — "~49 events" vs "76 anomaly events"

**Location:** §IX, Sample Sparsity paragraph

**Current text:**
```
The inherent rarity of extreme financial anomalies
(${\approx}49$ events over ten years) limits the statistical power...
```

**But §VI.1 says:**
```
...evaluated on 2,455 daily observations containing 76 anomaly events
(3.1\% event rate).
```

**And §IV.3 says:**
```
$n < 60$ events per regime over the ten-year horizon
```

**Three different numbers for what appears to be the same quantity.** One of these must be the correct total — pick the right one and make all three consistent:
- **76** = total anomaly events in the full 10-year OOS evaluation window
- **49** = possibly the number in the training partitions only?
- **< 60 per regime** = per-regime count (so total would be < 180)

**Fix — reconcile all three to the same number. Assuming 76 is correct (from §VI.1 which is the most specific):**
```latex
% §IX Sample Sparsity:
The inherent rarity of extreme financial anomalies
(${\approx}76$ events over ten years, or ${\approx}25$ per regime on
average) limits the statistical power...

% §IV.3 Statistical Considerations: change "n < 60" to "n ≈ 25"
% §VI.1 keep 76 as-is (it is the most precisely stated)
```

> **Action needed:** Run a quick count of total anomaly events from your data — check
> what the actual number is (total vs per-fold vs per-regime) and align all three locations.

---

### 🔴 ISSUE L2: "Static Regime Definitions" Paragraph — Contradicts Implementation

**Location:** §IX, Static Regime Definitions paragraph

**Current text:**
```
Regime boundaries based on fixed historical volatility quantiles assume
structural stationarity across the full evaluation horizon.
```

**Why wrong:** The implemented system uses a **rolling quantile** detector
(`RollingQuantileRegimeDetector`) as its primary detector — NOT fixed
historical quantiles. The leakage-free fix specifically refactored the
regime detector to use rolling thresholds frozen at each training fold.
Describing the method as "fixed historical volatility quantiles" misrepresents
the current implementation.

**Fix — update paragraph to reflect rolling quantile reality:**
```latex
\paragraph{Adaptive regime definitions}
Although the proposed framework employs rolling realized-volatility quantiles
to adapt regime boundaries across time, severe macroeconomic dislocations
(e.g., the 2020 COVID liquidity crisis) can distort even rolling thresholds
if the dislocation magnitude exceeds the historical calibration window.
Future work should investigate fully adaptive regime definitions using
Hidden Markov Models or nonparametric change-point detection to reduce
sensitivity to the choice of rolling window length.
```

---

### 🔴 ISSUE L3: "Walk-Forward Protocols" — Results Never Shown

**Location:** §IX, Predictive Generalisation paragraph

**Current text:**
```
Temporal validation experiments confirm substantial degradation under
walk-forward protocols, emphasising the gap between identifying
historical structural signatures and predicting future rare events.
```

**Why wrong:** The paper shows 5-fold temporal CV results in `fig:oos_classification`
but never explicitly labels any results as "walk-forward protocol" results.
If walk-forward experiments were not actually conducted or reported, this
paragraph makes a claim that cannot be verified by the reader.

**Fix — replace "walk-forward protocols" with "temporal cross-validation":**
```latex
\paragraph{Predictive generalisation}
Performance metrics reported for in-sample structural separability
do not generalise to out-of-sample predictive accuracy. Five-fold
temporal cross-validation confirms substantial performance degradation
relative to the in-sample structural baseline, emphasising the gap
between identifying historical structural signatures and predicting
future rare events under regime drift and non-stationarity.
```

---

### 🟢 ISSUE L4: Missing Limitation — Confidence Interval Width Caveat

**Location:** §IX (add new paragraph — identified in original revision plan)

**Problem:** The bootstrap CIs we compute (block bootstrap b=20) are wide
due to sparse positive labels. The paper never acknowledges that reported
CIs have limited precision. Reviewers running their own CI estimates will
notice this.

**Add new paragraph:**
```latex
\paragraph{Confidence interval width}
Circular block bootstrap confidence intervals ($b = 20$ trading days)
reported for classification metrics are wide due to the sparse positive
label density (${\approx}76$ events over ten years). Bootstrap replicates
that contain zero positive labels produce undefined precision and F1,
reducing effective replicate counts. Reported intervals should be
interpreted as indicative bounds rather than high-precision estimates;
caution is warranted in drawing strong comparative conclusions from
small metric differences within overlapping confidence intervals.
```

---

### 🟢 ISSUE L5: Missing Limitation — Transaction Costs Not Modelled

**Location:** §IX (add new paragraph — identified in original revision plan)

**Problem:** The economic backtest assumes zero transaction costs. For a
strategy that hedges 6.2% of trading days, even small bid-ask spreads
materially affect the Sharpe ratio and annualised return comparisons.
Q1 finance journals will always ask about this.

**Add new paragraph:**
```latex
\paragraph{Transaction costs}
The economic backtest assumes zero transaction costs. For the LightGBM
RC-MoE strategy, which triggers 152 hedging events over ten years
(${\approx}15$ per year), even a 5~basis-point round-trip cost per
hedge would reduce annualised return by approximately 0.08~pp, a small
but non-negligible adjustment. Future work should incorporate realistic
bid-ask spreads and market-impact models, particularly for the GARCH
baseline which hedges $3.2\times$ more frequently.
```

---

### ✅ THINGS THAT ARE FINE — §IX

| Limitation | Status |
|-----------|--------|
| Predictive generalisation framing | ✅ (fix wording only) |
| Regime transition sensitivity | ✅ Correctly identifies soft-gating gap |
| Single-asset scope | ✅ Well-argued |

---

## §X CONCLUSION — ISSUES FOUND

### 🔴 ISSUE N1: "32% Precision Improvement" — No In-Sample Qualifier (Again)

**Location:** §X Conclusion, contributions list, item (i)

**Exact text:**
```
(i) a regime-conditioned Mixture-of-Experts architecture achieving
a 32\% precision improvement and an 8\% Sharpe ratio gain
over global baselines
```

**Why wrong:** Same issue as Abstract, Intro, Discussion. The 32% is
in-sample only. In the Conclusion this is especially dangerous — it is
the final sentence reviewers read and will remember.

**Fix:**
```latex
(i) a regime-conditioned Mixture-of-Experts architecture achieving
a 32\% in-sample precision improvement and a 7.7\% Sharpe ratio gain
over global baselines, with the economic advantage confirmed under
strict leakage-free temporal cross-validation;
```

---

### 🔴 ISSUE N2: "8% Sharpe Ratio Gain" — Inconsistent with Everywhere Else

**Location:** §X Conclusion, contributions list, item (i)

**Current text:** `8\% Sharpe ratio gain`

**Everywhere else in the paper says:** `7.7\% improvement in portfolio Sharpe ratio`

**Fix:** Change `8\%` to `7.7\%` to be consistent with Abstract, Introduction, and the actual table value (0.717 → 0.772 = +7.7%).

---

### 🔴 ISSUE N3: Opening Paragraph — Informal Phrasing and Weak Language

**Location:** §X Conclusion, first paragraph

**Current text:**
```
This paper presents an alternative formulation for detecting financial
anomalies using regime-conditioned structural separability rather than
simply predictive learning. The suggested framework of RC-MoE utilizes
deterministic gating based on volatilities together with per-regime
classifier experts on interpretable factors to show that extreme events
have stable regime-dependent structural properties, which are not easily
detected through global anomaly detection methods.
```

**Problems:**
1. `"simply predictive learning"` — "simply" is informal and dismissive
2. `"suggested framework"` — should be "proposed framework"
3. `"based on volatilities"` — vague; should say "based on rolling realized volatility quantiles"
4. `"per-regime classifier experts"` — awkward phrasing; should be "per-regime expert classifiers"
5. `"to show that"` — weak; should be "demonstrating that"
6. `"not easily detected"` — vague; should be "structurally inaccessible to"

**Fix — rewrite opening:**
```latex
This paper introduces a regime-conditioned formulation for financial
anomaly detection grounded in structural separability rather than
predictive generalisation. The proposed RC-MoE framework employs
deterministic gating based on rolling realized volatility quantiles and
trains per-regime expert classifiers on interpretable financial factors,
demonstrating that extreme market events exhibit stable, regime-dependent
structural signatures that are structurally inaccessible to global
anomaly detection models.
```

---

### 🔴 ISSUE N4: Future Work Lists Already-Implemented Feature

**Location:** §X Conclusion, Future Work paragraph

**Current text:**
```
Future work will investigate adaptive regime definitions using
Hidden Markov Models and rolling quantile estimation...
```

**Why wrong:** Rolling quantile estimation is **already implemented** in
the current system (`RollingQuantileRegimeDetector`). Listing it as future
work misrepresents the current implementation.

**Fix — remove "rolling quantile estimation" from the future work list:**
```latex
Future work will investigate further refinement of adaptive regime
definitions beyond the current rolling-quantile approach, including
learned latent-state models (Hidden Markov Models, change-point
detection), cross-asset generalisation to fixed income and foreign
exchange markets, transaction-cost-aware portfolio optimisation, and
causal attribution frameworks for dynamic regime-level explanation
under evolving market conditions.
```

---

### 🟢 ISSUE N5: Missing — Leakage-Free OOS Numbers Not in Conclusion

**Location:** §X Conclusion, second paragraph / contributions

**Problem:** The Conclusion never mentions the leakage-free OOS results
(E-F1=0.714, AUC-PR=0.787 for 5-year horizon). These are the strongest
honest results and should close the paper. Reviewers who are sceptical
about the in-sample 95.5% need to see the honest OOS numbers mentioned
at the end.

**Add to second paragraph, after the contributions list:**
```latex
Under the leakage-free 5-fold temporal evaluation protocol, the RC-MoE
achieves an Event-F1 of 0.714 and AUC-PR of 0.787 on the five-year
evaluation horizon, confirming that the in-sample structural advantage
partially persists under strict out-of-sample conditions.
```

---

## COMPLETE CHECKLIST — §IX + §X

```
§IX LIMITATIONS
[x] L1 — Reconcile event counts: "49" vs "76" vs "< 60 per regime"
[x] L2 — Fix "fixed historical volatility quantiles" → rolling quantile implementation
[x] L3 — Replace "walk-forward protocols" → "5-fold temporal cross-validation"
[ ] L4 — ADD new paragraph: Confidence interval width caveat
[ ] L5 — ADD new paragraph: Transaction costs not modelled

§X CONCLUSION
[x] N1 — Add "in-sample" qualifier to "32% precision improvement"
[x] N2 — Fix "8% Sharpe ratio gain" → "7.7%" (matches every other occurrence)
[x] N3 — Rewrite informal opening paragraph
[x] N4 — Remove "rolling quantile estimation" from future work (already implemented)
[ ] N5 — ADD leakage-free OOS numbers (E-F1=0.714, AUC-PR=0.787) to conclusion
```

**§IX mandatory changes: 3, additions: 2**
**§X mandatory changes: 4, addition: 1**

---

## FULL PAPER — ALL SECTION READMES COMPLETE

| File | Section | Key Issues |
|------|---------|-----------|
| `S1_Abstract_Introduction.md` | §I–Abstract | 32% qualifier (×2), section outline wrong, Isolation Forest |
| `S2_Related_Work.md` | §II | Typo "ork.", taxonomy table |
| `S3_Methodology.md` | §III | 3 fragments, leaky labels, missing subsection |
| `S4_S5_ExpSetup_Ablation.md` | §IV–V | 5-fold CV missing, bootstrap missing, 24→48 dims (×3) |
| `S6_MultiModel_Economic.md` | §VI | Oracle claim, 3 missing spaces, leaky precision (×3) |
| `S7_S8_Explainability_Discussion.md` | §VII–VIII | Caption contradicts table, 32% qualifier |
| `S9_S10_Limitations_Conclusion.md` | §IX–X | Event count mismatch, 8%→7.7%, future work error |
