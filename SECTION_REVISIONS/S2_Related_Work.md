# Section 2 Revision: Related Work (§II)
## RC-MoE Paper — Q1 Journal Fix Plan

> **Scope:** ONLY §II Related Work
> **Status after review:** Mostly clean — only 2 real fixes needed, 1 optional improvement
> **Next:** After confirming done → §III Methodology

---

## HONEST ASSESSMENT FIRST

This section is **well-written and well-structured**. The four subsections (Statistical, Regime-Aware, Deep Learning, Interpretable ML) provide solid positioning. The literature table (`tab:lit_review`) is comprehensive. The taxonomy table (`tab:taxonomy`) and comparison table (`tab:comparison`) are academically sound. Do NOT over-edit this section — it reads well as-is.

---

## ISSUES FOUND

### 🔴 ISSUE R1: Typo — Stray "ork." at End of §II.4

**Location:** Last sentence of `\subsection{Interpretable Machine Learning in Finance}`

**Exact text (end of that subsection):**
```
...while shallow ensemble experts provide transparent decision rules at
the model level.ork.
```

**Why wrong:** The word `ork.` is a stray fragment — almost certainly a remnant of the word "framework." split across an edit. This is an embarrassing typo that a reviewer or copyeditor will flag immediately.

**Fix — delete `.ork.` so the sentence ends at "model level.":**
```
...while shallow ensemble experts provide transparent decision rules at
the model level.
```

---

### 🟡 ISSUE R2: `tab:taxonomy` — "Evaluation Focus" Row Undersells OOS Work

**Location:** Table `tab:taxonomy`, row "Evaluation Focus"

**Current row:**
```
Evaluation Focus & OOS performance & In-sample regime isolation
```

**Why it needs attention:** The taxonomy table is meant to contrast the predictive paradigm vs. this work's structural paradigm. However, now that the paper includes a proper leakage-free OOS evaluation (5-fold temporal CV), saying our evaluation focus is *only* "In-sample regime isolation" is no longer fully accurate — and could make a reviewer think we never do OOS testing. The claim that OOS is only for the "predictive paradigm" column is overstated.

**Fix — update only the right-hand cell:**
```
Evaluation Focus & OOS performance & In-sample separability + leakage-free OOS validation
```

This one-line change makes the taxonomy honest without changing the overall argument.

---

### ✅ THINGS THAT ARE FINE — DO NOT CHANGE

| Element | Status | Why |
|---------|--------|-----|
| §II.1 Statistical/Econometric subsection | ✅ | Accurate, well-cited, proper limitations |
| §II.2 Regime-Aware/MoE subsection | ✅ | Correct positioning, accurate claims about prior work |
| §II.3 Deep Learning subsection | ✅ | Accurate characterization of DL limitations |
| §II.4 Interpretable ML subsection | ✅ (except typo) | Sound argument, good citations |
| `tab:lit_review` — all 15 rows | ✅ | Accurate method/dataset/paradigm/limitation entries |
| `tab:comparison` — 3-row framework table | ✅ | Correct and concise |
| Positioning subsection | ✅ | Well-argued, defensible |
| "No prior work combines deterministic regime gating..." claim | ✅ | Defensible given the table evidence |
| All citations in this section | ✅ | Verify bib file has all entries (cannot check here) |

---

## CORRECTED LaTeX (Only the 2 Changed Locations)

### Fix 1: Remove stray "ork." — 1 word deletion

**FIND (end of §II.4 Interpretable ML subsection):**
```latex
...while shallow ensemble experts provide transparent decision rules at the model level.ork.
```

**REPLACE WITH:**
```latex
...while shallow ensemble experts provide transparent decision rules at the model level.
```

---

### Fix 2: Update taxonomy table evaluation focus row

**FIND:**
```latex
Evaluation Focus & OOS performance & In-sample regime isolation \\
```

**REPLACE WITH:**
```latex
Evaluation Focus & OOS performance & In-sample separability $+$ leakage-free OOS validation \\
```

---

## OPTIONAL IMPROVEMENT (NOT REQUIRED FOR ACCEPTANCE)

### 🟢 OPTIONAL R3: Add RC-MoE Row to `tab:lit_review`

Some Q1 journals expect the literature table to include the proposed method as the final row for direct comparison. This makes it easy for reviewers to see at a glance how the proposed work differs.

**Add as the last row before `\hline` in `tab:lit_review`:**
```latex
\textbf{RC-MoE (This Work)} & \textbf{2025} & \textbf{MoE + Regime Gating} &
\textbf{S\&P~500} & \textbf{Regime Tail} & \textbf{Long-Horizon OOS} &
\textbf{Yes} & \textbf{High} & \textbf{---} \\
```
> Only add this if the journal style allows self-citation in tables. IEEE usually does.

---

## SUMMARY CHECKLIST — RELATED WORK

```
MUST FIX
[x] R1 — Delete stray "ork." at end of §II.4 last sentence
[x] R2 — Update taxonomy table: "In-sample regime isolation"
          → "In-sample separability + leakage-free OOS validation"

OPTIONAL
[ ] R3 — Add RC-MoE row to tab:lit_review as last entry
```

**Total mandatory changes: 2 (both minor)**
**After confirming done → §III Methodology**
