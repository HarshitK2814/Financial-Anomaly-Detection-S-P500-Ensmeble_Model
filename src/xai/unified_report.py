"""
📋 UNIFIED XAI REPORT GENERATOR
==================================
Aggregates explanations from ALL pipeline stages into a single
cohesive report for the paper and dashboard.

Stages covered:
  1. Preprocessing (MI, KS, correlation)
  2. Regime detection (transition analysis, boundary)
  3. Tree models (SHAP)
  4. Deep learning (attention, integrated gradients)
  5. Meta-ensemble (expert contributions)
  6. Cross-method consistency (ECI)
"""
import numpy as np
import pandas as pd
from typing import Dict, Optional
import json
import os


class UnifiedXAIReport:
    """Generates comprehensive end-to-end XAI report.

    This is the module that synthesizes ALL explainability results
    into a format suitable for the paper's Results section.
    """

    def __init__(self, output_dir: str = 'artifacts/xai_reports'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.sections = {}

    def add_preprocessing_report(self, report: dict):
        """Add preprocessing explainability results."""
        self.sections['preprocessing'] = {
            'top_factors_per_regime': report.get('summary', {}),
            'mi_table': report['mi_table'].to_dict() if 'mi_table' in report else {},
        }

    def add_regime_report(self, transition_analysis: dict,
                           regime_distribution: pd.DataFrame = None,
                           boundary_analysis: dict = None):
        """Add regime-level explainability."""
        self.sections['regime'] = {
            'transition_analysis': transition_analysis,
            'distribution': regime_distribution.to_dict() if regime_distribution is not None else {},
            'boundary': boundary_analysis or {},
        }

    def add_model_shap_profiles(self, profiles: Dict[str, pd.DataFrame]):
        """Add per-regime SHAP importance profiles."""
        serialized = {}
        for regime, df in profiles.items():
            serialized[regime] = df.head(10).to_dict('records')
        self.sections['shap_profiles'] = serialized

    def add_attention_analysis(self, analysis: dict):
        """Add attention-based explanations from DL models."""
        self.sections['attention'] = analysis

    def add_expert_contributions(self, contributions: dict):
        """Add meta-ensemble expert contribution breakdown."""
        self.sections['expert_contributions'] = contributions

    def add_eci_results(self, eci_summary: dict):
        """Add Explainability Consistency Index results."""
        self.sections['eci'] = eci_summary

    def generate_report(self) -> str:
        """Generate formatted text report."""
        lines = [
            "=" * 70,
            "END-TO-END EXPLAINABILITY REPORT",
            "Regime-Conditioned Explainable Multi-Model Ensemble (RC-EME)",
            "=" * 70,
        ]

        # 1. Preprocessing
        if 'preprocessing' in self.sections:
            lines.extend([
                "\n" + "─" * 50,
                "§1 PREPROCESSING EXPLAINABILITY",
                "─" * 50,
                "\nTop discriminative factors per regime:",
            ])
            for regime, factors in self.sections['preprocessing'].get('top_factors_per_regime', {}).items():
                lines.append(f"  {regime} Vol: {', '.join(factors)}")

        # 2. Regime
        if 'regime' in self.sections:
            lines.extend([
                "\n" + "─" * 50,
                "§2 REGIME DETECTION EXPLAINABILITY",
                "─" * 50,
            ])
            ta = self.sections['regime'].get('transition_analysis', {})
            if ta:
                lines.append(f"\n  Insight: {ta.get('insight', 'N/A')}")
                lines.append(f"  Anomaly pre-transition rate: {ta.get('anomalies_preceded_by_transition', 0):.1%}")

        # 3. SHAP
        if 'shap_profiles' in self.sections:
            lines.extend([
                "\n" + "─" * 50,
                "§3 MODEL-LEVEL SHAP EXPLANATION PROFILES",
                "─" * 50,
            ])
            for regime, top_features in self.sections['shap_profiles'].items():
                lines.append(f"\n  {regime} Vol — Top-3 drivers:")
                for i, feat in enumerate(top_features[:3], 1):
                    lines.append(f"    {i}. {feat.get('feature', 'N/A')} "
                                f"(|SHAP|={feat.get('mean_abs_shap', 0):.4f})")

        # 4. Attention
        if 'attention' in self.sections:
            lines.extend([
                "\n" + "─" * 50,
                "§4 TEMPORAL ATTENTION ANALYSIS",
                "─" * 50,
            ])
            attn = self.sections['attention']
            if 'mean_entropy' in attn:
                lines.append(f"  Mean attention entropy: {attn['mean_entropy']:.4f}")

        # 5. Expert contributions
        if 'expert_contributions' in self.sections:
            lines.extend([
                "\n" + "─" * 50,
                "§5 META-ENSEMBLE EXPERT CONTRIBUTIONS",
                "─" * 50,
            ])
            for regime, contribs in self.sections['expert_contributions'].items():
                lines.append(f"\n  {regime} Vol:")
                for expert, weight in contribs.items():
                    bar = "█" * int(weight * 20)
                    lines.append(f"    {expert:<15} {weight:.1%} {bar}")

        # 6. ECI
        if 'eci' in self.sections:
            lines.extend([
                "\n" + "─" * 50,
                "§6 EXPLAINABILITY CONSISTENCY INDEX (ECI)",
                "─" * 50,
            ])
            eci = self.sections['eci']
            global_eci = eci.get('global', {})
            lines.append(f"\n  Global Mean ECI: {global_eci.get('mean', 0):.4f}")
            lines.append(f"  High consistency (>0.7): {global_eci.get('pct_high_consistency', 0):.1%}")
            lines.append(f"  Low consistency (<0.3): {global_eci.get('pct_low_consistency', 0):.1%}")

            if 'per_regime' in eci:
                for regime, stats in eci['per_regime'].items():
                    lines.append(f"  {regime} Vol: ECI={stats['mean']:.4f} ± {stats['std']:.4f}")

        lines.append("\n" + "=" * 70)
        report_text = "\n".join(lines)

        # Save
        report_path = os.path.join(self.output_dir, 'xai_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)

        json_path = os.path.join(self.output_dir, 'xai_report.json')
        with open(json_path, 'w') as f:
            json.dump(self.sections, f, indent=2, default=str)

        print(f"💾 XAI Report saved to {report_path}")
        print(f"💾 XAI Data saved to {json_path}")
        return report_text

    def get_paper_table(self) -> pd.DataFrame:
        """Generate a table suitable for the paper's Results section.

        Columns: Regime, Top Factor, SHAP Value, ECI Score, Expert Weight
        """
        rows = []
        regimes = ['Low', 'Med', 'High']

        for regime in regimes:
            row = {'Regime': f'{regime} Vol'}

            # Top SHAP factor
            if 'shap_profiles' in self.sections:
                profile = self.sections['shap_profiles'].get(regime, [])
                if profile:
                    row['Top Factor'] = profile[0].get('feature', 'N/A')
                    row['SHAP Value'] = profile[0].get('mean_abs_shap', 0)

            # ECI
            if 'eci' in self.sections:
                per_regime = self.sections['eci'].get('per_regime', {})
                if regime in per_regime:
                    row['Mean ECI'] = per_regime[regime].get('mean', 0)

            rows.append(row)

        return pd.DataFrame(rows)
