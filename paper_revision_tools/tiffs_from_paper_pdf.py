"""Build the journal TIFF set from the compiled paper itself, so every TIFF is exactly the
figure the paper shows. Raster figures are extracted at their embedded (native) resolution and
flattened onto white; the two vector figures (phase-1, phase-2) are rendered at 300 dpi.
Figures are numbered by their position in the PDF (page, then top-to-bottom), which is the
order LaTeX numbers them."""
import re
import sys
from pathlib import Path

import fitz
from PIL import Image

PDF = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
NAMES = ["phase-1", "phase-2", "phase-3", "topological_boundaries", "regime_tsne_kde", "regime_patterns",
         "ablation", "sensitivity_oos", "radar_overview", "oos_performance", "anomaly_detection_zoom",
         "crisis_focus_ohlc", "detection_panels", "per_regime_heatmap", "execution_lag",
         "cumulative_returns", "hedging_efficiency", "factor_beta", "evt_tail", "var_coverage",
         "basel_capital", "multimarket", "multimarket_distributions", "var_boundaries", "eci_chart",
         "shap_attribution"]

doc = fitz.open(PDF)
items = []          # (page, y, kind, payload)
seen = set()
for pno, page in enumerate(doc):
    for img in page.get_images(full=True):
        xref, smask = img[0], img[1]
        if xref in seen:
            continue
        rects = page.get_image_rects(xref)
        if not rects:
            continue
        pix = fitz.Pixmap(doc, xref)
        if pix.width < 300:
            continue
        seen.add(xref)
        items.append((pno, rects[0].y0, "raster", (xref, smask)))
    # vector figures: drawings with a "Figure N." caption but no raster image on the page area
    text = page.get_text("dict")
    for b in text["blocks"]:
        for line in b.get("lines", []):
            s = "".join(sp["text"] for sp in line["spans"])
            m = re.match(r"Figure (\d+)\.", s.strip())
            if m and int(m.group(1)) in (1, 2):
                items.append((pno, -1, "vector", (int(m.group(1)), b["bbox"])))
items.sort(key=lambda t: (t[0], t[1]))
vectors = [t for t in items if t[2] == "vector"]
rasters = [t for t in items if t[2] == "raster"]
print("vector figures:", len(vectors), "raster figures:", len(rasters))
assert len(vectors) == 2 and len(rasters) == 24, "unexpected figure count"

out = []
for pno, _, _, (num, cap_bbox) in vectors:
    page = doc[pno]
    # the figure occupies the area above its caption on that page
    clip = fitz.Rect(page.rect.x0 + 30, 40, page.rect.x1 - 30, cap_bbox[1] - 2)
    drawings = [d["rect"] for d in page.get_drawings() if d["rect"].y1 <= cap_bbox[1] and d["rect"].width < page.rect.width]
    if drawings:
        r = drawings[0]
        for d in drawings[1:]:
            r |= d
        clip = r & clip
    pix = page.get_pixmap(dpi=300, clip=clip, alpha=False)
    out.append((num, Image.frombytes("RGB", (pix.width, pix.height), pix.samples)))
for i, (pno, _, _, (xref, smask)) in enumerate(rasters):
    pix = fitz.Pixmap(doc, xref)
    if pix.n - pix.alpha >= 4:
        pix = fitz.Pixmap(fitz.csRGB, pix)
    if smask:
        pix = fitz.Pixmap(pix, fitz.Pixmap(doc, smask))
    mode = "RGBA" if pix.alpha else ("RGB" if pix.n == 3 else "L")
    im = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if im.mode == "RGBA":
        bg = Image.new("RGB", im.size, "white"); bg.paste(im, mask=im.split()[-1]); im = bg
    out.append((i + 3, im.convert("RGB")))
out.sort(key=lambda t: t[0])
for num, im in out:
    dst = OUT / f"Figure{num:02d}_{NAMES[num - 1]}.tif"
    im.save(dst, compression="tiff_lzw", dpi=(300, 300))
    print(f"{dst.name:44s} {im.size[0]}x{im.size[1]} px")
