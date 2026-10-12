"""
Journal upload set: every figure of the revised paper as a numbered TIFF (LZW, RGB, white
background) at its native resolution (300 dpi; the hedging-efficiency chart is 400 dpi).
Vector PDFs (phase-1/2/3) are rasterised at 300 dpi. The three new revision figures are
copied from Revision_Files/new_figures/tiff.

Numbering follows the float order of the revised paper (new figures: 8, 15, 20).
Run from model root:  python src/utils/export_submission_tiffs.py
"""
from pathlib import Path

import fitz
from PIL import Image

OUT = Path("../Revision_Files/all_figures_tiff")
OUT.mkdir(parents=True, exist_ok=True)
NEW = Path("../Revision_Files/new_figures/tiff")
CF, KS, FF = Path("correct figures"), Path("kronos_style_figures"), Path("final_figures")

ORDER = [
    ("phase-1", FF / "phase-1.pdf"),
    ("phase-2", FF / "phase-2.pdf"),
    ("phase-3", FF / "phase-3.pdf"),
    ("fig_m3_topological_boundaries", CF / "fig_m3_topological_boundaries.png"),
    ("fig3_regime_tsne_kde", KS / "fig3_regime_tsne_kde.png"),
    ("fig6_regime_patterns", KS / "fig6_regime_patterns.png"),
    ("ablation_chart_paper", CF / "ablation_chart_paper.png"),
    ("fig_sensitivity_oos", NEW / "fig_sensitivity_oos.tif"),            # new (R1-Q4)
    ("fig1_radar_overview", KS / "fig1_radar_overview.png"),
    ("oos_performance_chart", CF / "oos_performance_chart.png"),
    ("fig4_anomaly_detection_zoom", KS / "fig4_anomaly_detection_zoom.png"),
    ("fig5_crisis_focus_ohlc", KS / "fig5_crisis_focus_ohlc.png"),
    ("fig8_detection_panels", KS / "fig8_detection_panels.png"),
    ("per_regime_heatmap", CF / "per_regime_heatmap.png"),
    ("fig_execution_lag", NEW / "fig_execution_lag.tif"),                # new (R1-Q6)
    ("fig2_cumulative_returns", KS / "fig2_cumulative_returns.png"),
    ("hedging_efficiency_chart", CF / "hedging_efficiency_chart.png"),
    ("fig_factor_beta", CF / "fig_factor_beta.png"),
    ("fig_evt_tail", CF / "fig_evt_tail.png"),
    ("fig_var_coverage", NEW / "fig_var_coverage.tif"),                  # new (R1-Q5)
    ("fig_basel_capital", CF / "fig_basel_capital.png"),
    ("fig_multimarket", CF / "fig_multimarket.png"),
    ("fig7_multimarket_distributions", KS / "fig7_multimarket_distributions.png"),
    ("var_boundaries_chart", CF / "var_boundaries_chart.png"),
    ("eci_chart", CF / "eci_chart.png"),
    ("explainability_attribution", CF / "explainability_attribution.png"),
]


def to_rgb(im):
    if im.mode in ("RGBA", "LA", "P"):
        im = im.convert("RGBA")
        bg = Image.new("RGB", im.size, "white")
        bg.paste(im, mask=im.split()[-1])
        return bg
    return im.convert("RGB")


for i, (name, src) in enumerate(ORDER, 1):
    if src.suffix == ".pdf":
        pix = fitz.open(src)[0].get_pixmap(dpi=300, alpha=False)
        im, dpi = Image.frombytes("RGB", (pix.width, pix.height), pix.samples), (300, 300)
    else:
        im = Image.open(src)
        dpi = tuple(round(x) for x in im.info.get("dpi", (300, 300)))
        im = to_rgb(im)
    dst = OUT / f"Figure{i:02d}_{name}.tif"
    im.save(dst, compression="tiff_lzw", dpi=dpi)
    print(f"{dst.name:48s} {im.size[0]}x{im.size[1]} px  {dpi[0]} dpi  "
          f"{im.size[0] / dpi[0] * 2.54:.1f} cm wide")
