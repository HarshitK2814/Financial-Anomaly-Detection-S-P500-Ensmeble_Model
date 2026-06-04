# Section 4+5 Revision: Experimental Setup (§IV) + Ablation Study (§V)
## RC-MoE Paper — Q1 Journal Fix Plan

> **Scope:** §IV Experimental Setup AND §V Ablation Study
> **Critical finding:** Ablation section has wrong dimensions (24 vs 48) appearing 3 times
> **Next:** After confirming done → §VI Multi-Model Comparison

---

## §IV EXPERIMENTAL SETUP — ISSUES FOUND

### 🔴 ISSUE E1: 5-Fold Temporal CV Is Missing — Section Describes Different Protocols

**Location:** §IV.1 Evaluation Framework, OOS subsubsection

**Current text describes two OOS protocols:**
1. Blocked Temporal Split (70% / 30% chronological)
2. Walk-Forward Validation (train on all up to $t$, evaluate on $t+1$)

**Why wrong:** The actual main OOS results (shown in `fig:oos_classification` and
`tab:regime_conditioning`) use **5-fold temporal cross-validation**, which is
described in the figure captions ("5-fold temporal cross-validation") but
**never described in the Experimental Setup section**. A reviewer reading §IV
and then seeing "5-fold temporal CV" in the results will immediately ask:
"Where was this protocol described?"

**Fix — add a third bullet (or rewrite as the primary protocol):**
```latex
\item \textbf{5-Fold Temporal Cross-Validation (Primary OOS Protocol):}
      The full evaluation horizon is partitioned into 5 non-overlapping
      temporal folds. Models are trained on folds $1, \ldots, k-1$ and
      evaluated on fold $k$, for $k \in \{2,3,4,5\}$. Fold boundaries
      respect strict chronological ordering; no shuffling is applied.
      This protocol is used for all results reported in
      Section~\ref{sec:multimodel} and Table~\ref{tab:regime_conditioning}.
```

---

### 🔴 ISSUE E2: Statistical Considerations — Block Bootstrap Missing

**Location:** §IV.3 Statistical Considerations

**Current text:**
```
Given the inherently low frequency of extreme financial anomalies
($n < 60$ events per regime over the ten-year horizon), confidence
intervals and formal hypothesis tests possess limited statistical power.
All results are therefore reported with explicit contextualisation of
per-regime sample sizes and are interpreted as structural diagnostics
rather than population-level performance guarantees.
```

**Why wrong:** The leakage-free code implements circular block bootstrap
($b=20$) for confidence intervals. The section says results are reported
"with explicit contextualisation" but makes no mention of CIs or how they
are computed. A reviewer will ask: "Are confidence intervals provided? What
method?"

**Fix — add two sentences:**
```latex
Given the inherently low frequency of extreme financial anomalies
($n < 60$ events per regime over the ten-year horizon), confidence
intervals and formal hypothesis tests possess limited statistical power.
Confidence intervals are computed via circular block bootstrap with block
size $b = 20$ trading days, which preserves the autocorrelation structure
of financial return series. All results are reported with explicit
contextualisation of per-regime sample sizes and are interpreted as
structural diagnostics rather than population-level performance guarantees.
```

---

### 🟡 ISSUE E3: "Walk-Forward Validation" — Results Never Explicitly Shown

**Location:** §IV.1, Walk-Forward bullet

**Problem:** Walk-Forward Validation is described as a protocol but no results
table or figure is explicitly labelled as walk-forward results. If this protocol
is not actually used in the reported results, describing it creates false
expectations.

**Fix (choose one):**
- **Option A:** Remove the walk-forward bullet if it was not used
- **Option B:** Add note: *"Walk-forward results are consistent with 5-fold CV
  results and are omitted for brevity; they are available in the supplementary material."*

---

### ✅ THINGS THAT ARE FINE — §IV

| Element | Status |
|---------|--------|
| In-Sample evaluation description | ✅ Well framed as diagnostic |
| Precision/Recall/F1/Event-F1 metric definitions | ✅ Correct |
| Class imbalance handling | ✅ Correctly justifies dropping accuracy |
| Recall weighting rationale | ✅ Economically sound |

---

## §V ABLATION STUDY — ISSUES FOUND

### 🔴 ISSUE A1: WRONG DIMENSION — "24-dimensional" Factor Vector (Appears 3 Times)

**This is the most critical error in the ablation section.**

The factor representation is 48-dimensional (16 base factors × 3 lags: base + lag-1 + lag-5), as defined in Eq.~(9) and Table~4 of §III. But the ablation section says **24-dimensional** three times:

**Occurrence 1 — Ablation Design, item 5:**
```
\textbf{Global Random Forest:} A single Random Forest classifier trained
on the \textbf{24-dimensional} factor representation \emph{without} regime
separation
```

**Occurrence 2 — Ablation Analysis, observation 1:**
```
Replacing raw 128-day return windows with the \textbf{24-dimensional}
engineered factor vector yields a substantial improvement over deep
architectures (ConvVAE: 64.0\%; Global RF: 70.3\%).
```

**Occurrence 3 — Ablation Analysis, observation 1:**
```
This confirms that dimensionality reduction from 1\,280 to \textbf{24}
features improves the signal-to-noise ratio
```

**Fix — change all three to 48:**
- Item 5: `24-dimensional` → `48-dimensional`
- Analysis obs. 1: `24-dimensional` → `48-dimensional`
- Analysis obs. 1: `1\,280 to 24` → `640 to 48`

> **Note on "1,280":** Raw OHLCV = 5 channels × 128 timesteps = 640 dimensions,
> not 1,280. Change to `640 to 48` unless you have 10 raw channels (in which case
> document what they are).

---

### 🔴 ISSUE A2: Ablation F1 Scores — No "In-Sample" Qualifier Anywhere

**Location:** §V.2 Results + §V.3 Analysis

**Specific sentences:**
```
The improvement from Global RF (70.3\%) to RC-MoE (95.5\%)---driven
solely by the three-regime partitioning with no change to features,
anomaly labels, or model complexity---demonstrates that per-regime
expert specialisation...is the dominant source of separability gains.
```

**Why wrong:** 95.5% F1 is an in-sample structural separability diagnostic.
Without stating "in-sample", reviewers will assume this is an OOS result and
immediately reject it as implausibly high.

**Fix — add "in-sample" qualifier every time a % value appears in analysis:**
```latex
The improvement from Global RF (70.3\% in-sample F1) to
RC-MoE (95.5\% in-sample F1)---driven solely by the three-regime
partitioning---demonstrates that per-regime expert specialisation,
rather than architectural expressiveness, is the dominant source of
\emph{structural separability} gains under regime-isolated training conditions.
```

---

### 🔴 ISSUE A3: `fig:ablation` Caption — Missing "In-Sample" Qualifier

**Current caption:**
```
\caption{Ablation study F1 scores. Each bar represents a successive
         addition of the RC-MoE components.}
```

**Why wrong:** Without "in-sample", the figure caption implies these are OOS
performance numbers. 95.5% will look fraudulent to any reviewer.

**Fix:**
```latex
\caption{Ablation study: in-sample structural separability F1 scores under
         regime-isolated training conditions. Each bar represents a successive
         addition of RC-MoE components. These results constitute an upper
         bound on regime-conditional factor-space discriminability and are
         \emph{not} out-of-sample performance claims.}
```

---

### 🔴 ISSUE A4: Anomaly Transformer F1 Score — Missing from Analysis

**Location:** §V.3 Ablation Analysis

**Problem:** The ablation design lists 6 variants (SD Rule, Z-Score, ConvVAE,
Anomaly Transformer, Global RF, RC-MoE). The analysis only explicitly mentions
F1 scores for **ConvVAE (64.0%)**, **Global RF (70.3%)**, and **RC-MoE (95.5%)**. 

**Missing:** What is the Anomaly Transformer's F1 score? What are SD Rule and
Z-Score scores? If `fig:ablation` shows all bars, the reader can see them, but
the text never states the Anomaly Transformer result, which is listed as a key
baseline.

**Fix — add exact number to analysis observation 3:**
```latex
Deep architectures (ConvVAE: 64.0\%; Anomaly Transformer: 58.0\%)
underperform despite possessing orders of magnitude more parameters
than the shallow RC-MoE experts.
```
> Replace 58.0% with the actual number from your results.

---

### 🟡 ISSUE A5: Ablation Uses Different Feature Set Than Main Evaluation

**Location:** Ablation Design, item 5 + Ablation Analysis

**Problem:** The Global RF ablation baseline is described as using a
`24-dimensional` (should be 48) factor set, but the full RC-MoE uses the same
48-dimensional set. A reviewer will ask: "Is the Global RF being compared fairly,
with the same 48 features as the MoE experts?"

**Fix — clarify in item 5:**
```latex
\item \textbf{Global Random Forest:} A single Random Forest
      classifier~\cite{breiman2001random} trained on the full
      48-dimensional factor representation (identical features used by
      the MoE experts) \emph{without} regime separation, serving as the
      primary ML baseline and isolating the contribution of regime
      conditioning alone.
```

---

### 🟡 ISSUE A6: Ablation Results Has No Numbers Table — Only a Figure

**Location:** §V.2 Results

**Current:** Only `fig:ablation` (a bar chart). No table with exact values.

**Problem:** For reproducibility and peer review, exact F1 values should be stated.
A bar chart alone does not give sufficient precision (readers cannot read off exact
numbers). Q1 journals typically expect a table for ablation results.

**Fix — add a small inline table after the figure:**

```latex
\begin{table}[H]
\caption{Ablation Study: In-Sample Structural Separability F1 Scores}
\label{tab:ablation_results}
\centering
\footnotesize
\begin{tabular}{lcc}
\toprule
\textbf{Variant} & \textbf{In-Sample F1 (\%)} & \textbf{$\Delta$ vs Previous} \\
\midrule
Standard Deviation Rule & XX.X & --- \\
Z-Score                 & XX.X & $+$X.X \\
ConvVAE                 & 64.0 & $+$X.X \\
Anomaly Transformer     & XX.X & $+$X.X \\
Global RF (48-dim)      & 70.3 & $+$X.X \\
\textbf{RC-MoE (Proposed)} & \textbf{95.5} & $\mathbf{+25.2}$ \\
\bottomrule
\multicolumn{3}{l}{\footnotesize $^\dagger$ All F1 values are
in-sample structural separability diagnostics, not OOS claims.}
\end{tabular}
\end{table}
```
> Replace XX.X with actual values from your code run.

---

### ✅ THINGS THAT ARE FINE — §V

| Element | Status |
|---------|--------|
| Ablation design structure (stepwise component removal) | ✅ Correct methodology |
| Observation 1: factor engineering > DL end-to-end | ✅ Valid conclusion |
| Observation 3: model capacity ≠ problem solution | ✅ Valid conclusion |
| Citations for ConvVAE, Anomaly Transformer, RF | ✅ Appropriate |
| Six-variant design covering full spectrum | ✅ Well structured |

---

## COMPLETE CHECKLIST — §IV + §V

```
§IV EXPERIMENTAL SETUP
[x] E1 — Add 5-fold temporal CV as the PRIMARY OOS protocol
[x] E2 — Add circular block bootstrap CIs to Statistical Considerations
[ ] E3 — Decide: keep or remove walk-forward protocol description

§V ABLATION STUDY
[x] A1 — Fix "24-dimensional" → "48-dimensional" (3 occurrences)
[x] A1 — Fix "1,280 to 24" → "640 to 48"
[x] A2 — Add "in-sample" qualifier to all F1 % values in analysis text
[x] A3 — Update fig:ablation caption with "in-sample structural separability"
[x] A4 — Add Anomaly Transformer F1 score to analysis observation 3
[x] A5 — Clarify Global RF uses same 48-dim features as MoE experts
[ ] A6 — Add inline ablation table with exact numbers (recommended)
```

**§IV mandatory changes: 2**
**§V mandatory changes: 5 (A1 counts as 4 text fixes)**
**After confirming done → §VI Multi-Model Comparison + Economic Evaluation**
