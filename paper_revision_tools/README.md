# Paper revision tools

Helper scripts used to prepare the revised manuscript (`final.tex`) for Applied Artificial Intelligence.

| File | Purpose |
|---|---|
| `make_marked_copy.py` | Builds the "changes highlighted" copy of the manuscript with latexdiff: original text black, new or changed text highlighted in yellow, deleted text hidden; tables and figures are compared separately so that only changed cells and caption words are highlighted. |
| `highlight_preamble.tex` | latexdiff preamble used by `make_marked_copy.py` (soul `\hl` highlighting). |
| `tiffs_from_paper_pdf.py` | Exports every figure of the compiled paper as a 300-dpi TIFF for journal submission. |
| `chk.py` | Checks brace balance and column counts of every table in a LaTeX file. |
| `table_lines.py` | Prints the line ranges of selected tables in `final.tex`. |

The scripts were run from the authors' local project folder, so the input and output paths at the top of each script (the original submitted manuscript, the latexdiff installation and the output folder) need to be adjusted before use elsewhere.
