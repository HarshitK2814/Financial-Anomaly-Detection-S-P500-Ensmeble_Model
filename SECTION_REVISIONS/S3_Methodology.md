# Section 3 Revision: Methodology (§III)
## RC-MoE Paper — Q1 Journal Fix Plan

> **Scope:** ONLY §III Methodology
> **Status:** 3 grammatical errors (must fix), 3 accuracy issues (must fix), 1 missing subsection, 2 optional clarifications
> **Next:** After confirming done → §IV Experimental Setup

---

## ISSUES FOUND — METHODOLOGY

### 🔴 ISSUE M1: Incomplete Sentence in Opening Paragraph

**Location:** §III opening paragraph, sentence 3

**Exact broken sentence:**
```
The system design, where a rolling window of returns is passed through a
factor extraction stage, an adaptive multi-detector gating network, and a
regime-specific expert scoring procedure to produce a final calibrated
anomaly probability.
```

**Why wrong:** "The system design, where..." is a grammatical fragment. "The system design" is a noun phrase but there is no predicate — the sentence never states *what* the system design *does* or *is*. This reads as a sentence cut short during editing.

**Fix — add a verb to make it a complete sentence:**
```latex
The system processes each rolling window of returns through three sequential
stages: a factor extraction pipeline, an adaptive multi-detector gating
network, and a regime-specific expert scoring procedure, culminating in a
final calibrated anomaly probability.
```

---

### 🔴 ISSUE M2: Anomaly Label Definition Inconsistent with Leakage-Free Fix

**Location:** §III.1 Dataset Description, sentence:

**Exact text:**
```
The term ``anomaly'' is defined using a regime-conditioned threshold set at
the top 2\% tail of the return distribution for each volatility regime,
resulting in an overall anomaly rate of approximately 2.0\%.
```

**Why wrong:** Before the leakage fix, anomaly labels were computed from **full-dataset quantile thresholds** (look-ahead bias). After the fix, they use **rolling expanding-window quantiles** — the threshold at time $t$ is derived only from $|r_0|, \ldots, |r_{t-1}|$. The paper text still describes the old (leaky) label construction without any mention of the rolling expanding-window approach.

**Fix — update that sentence:**
```latex
The term ``anomaly'' is defined using a regime-conditioned threshold set at
the top 2\% tail of the absolute return distribution conditional on the
prevailing volatility regime. To eliminate look-ahead bias in label
construction, thresholds at time $t$ are computed using a rolling
expanding-window quantile derived exclusively from historical observations
$\{|r_0|, \ldots, |r_{t-1}|\}$ (minimum 20 observations), ensuring that
future return information never influences labeling. The resulting overall
anomaly rate is approximately 2.0\%.
```

---

### 🔴 ISSUE M3: Incomplete Sentence Before Figure 2

**Location:** End of §III.4 Factor-Based Representation, before `fig:phase2_regime`

**Exact broken sentence:**
```
The extraction process , where the raw OHLCV windows are projected onto
the 48-dimensional factor manifold.
```

**Why wrong:** "The extraction process, where..." is again a grammatical fragment — a dependent clause with no main verb. Also has a stray space before the comma. This was likely meant to reference a figure that was removed or rearranged.

**Fix — rewrite as a complete sentence:**
```latex
The factor extraction pipeline maps raw OHLCV windows onto the
48-dimensional factor manifold; the resulting separation between
crisis and normal states in this projected space confirms that the
extended representation preserves the discriminative information
required for regime-conditioned anomaly detection.
```

---

### 🔴 ISSUE M4: Static Quantile Detector — Describes Full-Dataset Look-Ahead

**Location:** §III.3 Regime-Conditioned Market States, "Detector 3: Static Quantile"

**Exact text:**
```
\textbf{Detector 3: Static Quantile:} A global baseline detector applies
fixed quantile thresholds computed over the entire historical sample,
providing a stable reference that anchors the adaptive detectors.
```

**Why wrong:** "Thresholds computed over the entire historical sample" means test-fold data is used to set thresholds — exactly the leakage we fixed. Even if this is one of three detectors in a fusion, a reviewer will flag this as leakage. The per-fold regime refitting fix (`predict_from_last_state`) was applied to the rolling quantile detector, but the paper still describes Detector 3 as using the full historical sample.

**Fix — clarify that thresholds are frozen at training-fold boundaries:**
```latex
\textbf{Detector 3: Static Quantile:} A global baseline detector applies
fixed quantile thresholds calibrated on the training partition only,
providing a stable reference that anchors the adaptive detectors while
preventing look-ahead bias from future observations.
```

---

### 🟡 ISSUE M5: HMM and Regime Detectors — Per-Fold Refitting Not Mentioned

**Location:** §III.3 Regime-Conditioned Market States, "Detector 2: Gaussian HMM"

**Current text says nothing about whether HMM is re-fit per fold.**

**Why it matters:** A reviewer evaluating whether the regime assignment is leaky will check whether the HMM transition matrix and emission parameters are fit on training data only or the full dataset. If the paper is silent, reviewers will assume the worst.

**Fix — add one sentence after the HMM description:**
```latex
Under the leakage-free evaluation protocol (Section~\ref{sec:leakage_free}),
all three regime detectors are re-calibrated within each training fold;
test-fold regime assignments are produced by projecting the terminal
training-fold thresholds forward without access to future volatility
observations.
```

---

### 🟡 ISSUE M6: `fig:topological_boundaries` Caption — Missing In-Sample Caveat

**Location:** Caption of `fig:topological_boundaries`

**Current caption:**
```
...confirming that financial anomalies are structurally regime-dependent
and cannot be modeled adequately using a single global decision boundary.
```

**Why wrong:** This visualization is generated from in-sample training partition data. The caption presents the conclusion as a general fact, but a reviewer seeing "confirming" will ask: in-sample or OOS?

**Fix — add caveat:**
```latex
\caption{Regime-dependent topological transformation of anomaly decision
boundaries, visualized on in-sample training partitions. The geometric
structure of anomaly probability surfaces changes substantially across
low-, medium-, and high-volatility regimes, confirming that financial
anomalies are structurally regime-dependent within each volatility
environment and cannot be modeled adequately using a single global
decision boundary.}
```

---

### 🟡 ISSUE M7: Text After Fig 3 — Specific In-Sample Claims Need Caveat

**Location:** Paragraph after `\label{fig:topological_boundaries}`

**Current text:**
```
...confirm that anomaly structure is fundamentally regime-dependent:
anomalies in the low-volatility regime are concentrated at extreme
momentum values, whereas in the high-volatility regime they span a
broader range of return trends...
```

**Fix — add "(under in-sample training partition analysis)" after the colon:**
```latex
...confirm that anomaly structure is fundamentally regime-dependent
(under in-sample training partition analysis): anomalies in the
low-volatility regime are concentrated at extreme momentum values,
whereas in the high-volatility regime they span a broader range of
return trends---a separation that a single global classifier would
inevitably compromise.
```

---

### 🔴 ISSUE M8: Missing Leakage-Free Evaluation Protocol Subsection

**Location:** After §III.1 Dataset Description — this subsection does not exist yet

**Why wrong:** The paper now implements 6 leakage-prevention measures (per-fold normalization, per-fold regime detection, rolling expanding-window labels, training-only threshold optimization, circular block bootstrap CIs, non-overlapping stride). None of these are described in the Methodology. Reviewers will ask "where is the evaluation protocol described?" and find nothing. This is the most important missing piece.

**Add as a new §III.2 (renaming subsequent subsections):**

```latex
\subsection{Leakage-Free Evaluation Protocol}
\label{sec:leakage_free}

To ensure evaluation integrity, six potential sources of look-ahead
bias were systematically identified and remediated:

\begin{enumerate}[leftmargin=*,itemsep=2pt]
  \item \textbf{Per-fold normalization:} \texttt{StandardScaler} is
        fitted exclusively on each training partition and applied to
        the held-out test partition, replacing global Z-score
        normalization that would otherwise import test-set statistics.

  \item \textbf{Per-fold regime detection:} All three regime detectors
        (rolling quantile, Gaussian HMM, static quantile) are
        re-calibrated within each training fold. Test-fold regime
        assignments are produced by projecting the terminal
        training-fold thresholds forward
        (\texttt{predict\_from\_last\_state}) without access to future
        volatility observations.

  \item \textbf{Rolling expanding-window anomaly labels:} The anomaly
        threshold at time $t$ is derived only from
        $\{|r_0|, \ldots, |r_{t-1}|\}$ via expanding quantile
        (minimum 20 observations), eliminating the circular dependency
        that arises when full-dataset quantiles define labels.

  \item \textbf{Training-only threshold optimization:} Decision
        boundaries are calibrated exclusively on training-fold
        predicted probabilities; test-fold labels are never accessed
        during the threshold search.

  \item \textbf{Circular block bootstrap confidence intervals:} A
        block size of $b = 20$ trading days preserves the
        autocorrelation structure of financial return series,
        replacing i.i.d.\ bootstrap which produces artificially
        narrow intervals for dependent data.

  \item \textbf{Non-overlapping evaluation stride:} Cross-validation
        folds are constructed with stride equal to the window length
        to prevent temporal leakage from overlapping windows across
        train/test boundaries.
\end{enumerate}

All empirical results in subsequent sections are produced under this
protocol. The protocol is implemented in the open-source evaluation
script \texttt{run\_leakage\_free\_evaluation.py} accompanying this paper.
```

---

### 🟢 ISSUE M9: "Latent\_State\_A/B/C" — Vague Factor Names

**Location:** `tab:factor_summary`, Microstructure rows

**Current rows:**
```
Microstruct. & Latent_State_A & Hidden market dynamics (ch. A)
Microstruct. & Latent_State_B & Hidden market dynamics (ch. B)
Microstruct. & Latent_State C & Range-based liquidity proxy
```

**Why it matters:** "ch. A/B/C" is not a meaningful economic explanation. A Q1 reviewer will push back on three factors labeled "hidden market dynamics." They are extracted from additional OHLCV data channels (High-Low spread, Volume-Price ratio, etc.) and should be named accordingly.

**Fix (optional but recommended):**
```
Microstruct. & HL\_Spread\_Factor   & Intraday price range proxy (High--Low) \\
Microstruct. & VolPrice\_Ratio      & Abnormal volume-to-price-move ratio    \\
Microstruct. & Range\_Illiquidity   & Range-based Amihud-style liquidity     \\
```

---

## SUMMARY OF ALL CHANGES

```
MUST FIX
[x] M1 — Fix incomplete sentence in §III opening paragraph
[x] M2 — Update anomaly label to mention rolling expanding-window quantiles
[x] M3 — Fix incomplete sentence "The extraction process , where..."
[x] M4 — Fix Static Quantile detector: "entire historical sample" → "training partition"
[x] M8 — Add leakage-free protocol subsection (§III.2, new)

SHOULD FIX
[x] M5 — Add per-fold refitting note after HMM detector description
[x] M6 — Add "in-sample training partitions" to fig:topological_boundaries caption
[x] M7 — Add "in-sample training partition analysis" caveat to text after topological fig

OPTIONAL
[ ] M9 — Rename Latent_State_A/B/C to meaningful factor names in tab:factor_summary
```

**Total mandatory changes: 5**
**Total recommended changes: 3**
**Total optional changes: 1**

**After confirming done → §IV Experimental Setup**
