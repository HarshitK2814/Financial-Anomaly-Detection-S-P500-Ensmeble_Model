r"""Build the 'changes highlighted' copy of the revised manuscript.

Style follows the journal's example (DR_detection_springer_marked.pdf): original text black,
new or changed text with a yellow highlight, deleted text not shown.
Running text: word-level latexdiff with highlight_preamble.tex.
Floats (tables and figures): latexdiff breaks complex tabular code (\cmidrule, \multicolumn,
\tbl{}{} ...), so every float is replaced by a placeholder before diffing and put back
afterwards. Images are left untouched; only the changed words of captions and table notes and
the changed cells of tables are highlighted (moved but unchanged rows stay black).
"""
import difflib
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
WORK = ROOT / "Revision_Files" / "latexdiff"
SP = Path(r"C:\Temp\claude\C--Users-Harshit-Kumar-Downloads-Financial-Anomaly-Detection-Journal-Paper-Coding-Part\b2f34e9c-cc78-42ff-ad1b-c4000a2ace4e\scratchpad")
OLD = WORK / "original_submitted.tex"
NEW = ROOT / "Financial-Anomaly-Detection-S-P500-Ensmeble_Model" / "final.tex"
OUT = WORK / "revised_marked_changes.tex"

FLOAT = re.compile(r"\\begin\{(table\*?|figure\*?)\}.*?\\end\{\1\}", re.S)


def key_of(block, i):
    m = re.search(r"\\label\{([^}]+)\}", block)
    return m.group(1) if m else f"nolabel{i}"


def extract(tex):
    floats = {}

    def rep(m):
        k = key_of(m.group(0), len(floats))
        floats[k] = m.group(0)
        return "\n\\RCMFLOAT{" + k + "}\n"
    return FLOAT.sub(rep, tex), floats


def norm(s):
    return re.sub(r"\s+", " ", s).strip()


# ---- float marking (as in the journal's example: images untouched, only changed words
# ---- of captions/notes and changed table cells are highlighted; rules stay black)
TOKEN = re.compile(r"""
    (?P<math>\$[^$]*\$)
  | (?P<cmd>\\[A-Za-z]+\*?(?:\[[^\]]*\])?(?:\{(?:[^{}]|\{(?:[^{}]|\{[^{}]*\})*\})*\})*)
  | (?P<word>(?:[^\s\\${}%]|\\['`^"~=.][A-Za-z]|\\['`^"~=.]\{\\?[A-Za-z]\})+)
  | (?P<esc>\\.)
  | (?P<grp>\{(?:[^{}]|\{[^{}]*\})*\})
  | (?P<com>%[^\n]*)
""", re.X)
NO_HL = re.compile(r"\\(hspace|vspace|newline|label|centering|raggedright|footnotesize|scriptsize|small|"
                   r"normalsize|arraybackslash|includegraphics|resizebox|setlength|renewcommand|par)\b")


def tokens(s):
    """[(token, kind, trailing whitespace)], leading whitespace returned separately"""
    lead = re.match(r"\s*", s).group(0)
    out, pos = [], len(lead)
    while pos < len(s):
        m = TOKEN.match(s, pos)
        if not m:                      # stray character (e.g. lone brace): keep it as an unhighlightable token
            m = re.compile(r"\S").match(s, pos)
            kind = "raw"
        else:
            kind = m.lastgroup
        ws = re.compile(r"\s*").match(s, m.end())
        out.append((m.group(0), kind, ws.group(0)))
        pos = ws.end()
    return lead, out


def hl_able(tok, kind):
    if kind in ("com", "raw", "math"):
        return False
    if kind == "grp":
        return "\\" not in tok and tok != "{}"
    if kind == "esc":
        return tok not in ("\\\\",)
    if kind == "cmd":
        return not NO_HL.match(tok)
    return True


FONT_CMD = re.compile(r"\\(emph|textbf|textit|textsc|texttt|textrm|textsf|textup|underline|rev|textsuperscript|textsubscript)\{(.*)\}$", re.S)
BOX_CMD = re.compile(r"\\(cite[a-zA-Z]*|ref|eqref|autoref|pageref|url|href)\b")


def hl_wrap(s):
    r"""highlight a stretch of new LaTeX text.
    soul's \hl draws one bar for a whole command argument, which runs past the margin when the
    argument (a citation, an \emph{title}) wraps; so the highlight goes inside font commands and
    citations/references/URLs/math get their own tight yellow box (\RCMbox)."""
    lead, toks = tokens(s)
    out, buf = [lead], []

    def flush():
        if buf:
            body = "".join(t + w for t, k, w in buf[:-1]) + buf[-1][0]
            out.append("\\hl{" + body + "}" + buf[-1][2])
            buf.clear()
    for t, k, w in toks:
        if k == "cmd":
            f = FONT_CMD.match(t)
            if f and get_cmd_arg(t, f.group(1)) == (len(f.group(1)) + 2, len(t) - 1):
                flush()
                out.append("\\" + f.group(1) + "{" + hl_wrap(f.group(2)) + "}" + w)
                continue
            if BOX_CMD.match(t) or "$" in t:
                flush()
                out.append("\\RCMbox{" + t + "}" + w)
                continue
            if "{" in t and not NO_HL.match(t):     # unknown command with arguments: leave as is
                flush()
                out.append(t + w)
                continue
        if k == "math":
            flush()
            out.append("\\RCMbox{" + t + "}" + w)
        elif hl_able(t, k):
            buf.append(("~" if t == "\\ " else t, k, w))     # soul cannot rebuild a control space
        else:
            flush()
            out.append(t + w)
    flush()
    return "".join(out)


def mark_run(run):
    """run = list of tokens that are new"""
    return hl_wrap("".join(t + w for t, k, w in run))


def word_diff(old, new):
    """new text with the words that are not in old highlighted"""
    lead, nt = tokens(new)
    if old is None:
        return lead + mark_run(nt)
    _, ot = tokens(old)
    sm = difflib.SequenceMatcher(a=[t for t, k, w in ot], b=[t for t, k, w in nt], autojunk=False)
    out = [lead]
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg = nt[j1:j2]
        out.append("".join(t + w for t, k, w in seg) if op == "equal" else mark_run(seg))
    return "".join(out)


def find_arg(s, i):
    """s[i] == '{' -> index just after the matching '}'"""
    depth = 0
    for j in range(i, len(s)):
        c = s[j]
        if c == "\\":
            continue
        if c == "{" and (j == 0 or s[j - 1] != "\\"):
            depth += 1
        elif c == "}" and s[j - 1] != "\\":
            depth -= 1
            if depth == 0:
                return j + 1
    raise ValueError("unbalanced braces")


def get_cmd_arg(s, cmd, nth=1):
    """(start, end) of the nth brace argument of the first \\cmd in s, or None"""
    m = re.search(r"\\" + cmd + r"\*?(?:\[[^\]]*\])?\s*\{", s)
    if not m:
        return None
    i = m.end() - 1
    for _ in range(nth - 1):
        i = find_arg(s, i)
        while s[i].isspace():
            i += 1
    j = find_arg(s, i)
    return i + 1, j - 1


def arg_text(s, cmd, nth=1):
    r = get_cmd_arg(s, cmd, nth) if s else None
    return s[r[0]:r[1]] if r else None


def replace_arg(s, cmd, old_s, nth=1):
    r = get_cmd_arg(s, cmd, nth)
    if not r:
        return s
    return s[:r[0]] + word_diff(arg_text(old_s, cmd, nth), s[r[0]:r[1]]) + s[r[1]:]


ROW_END = re.compile(r"\\\\(?:\[[^\]]*\])?")
RULES = re.compile(r"\s*(?:\\(?:toprule|midrule|bottomrule|hline|addlinespace)(?:\[[^\]]*\])?"
                   r"|\\cmidrule(?:\([^)]*\))?\{[^}]*\}|%[^\n]*\n)")


def split_cells(row):
    cells, depth, start = [], 0, 0
    for j, c in enumerate(row):
        if c in "{}" and (j == 0 or row[j - 1] != "\\"):
            depth += 1 if c == "{" else -1
        elif c == "&" and depth == 0 and (j == 0 or row[j - 1] != "\\"):
            cells.append(row[start:j])
            start = j + 1
    cells.append(row[start:])
    return cells


def rows_of(body):
    """[(rule prefix, [cells], row terminator)]"""
    parts = ROW_END.split(body)
    ends = ROW_END.findall(body) + [""]
    rows = []
    for p, e in zip(parts, ends):
        pre = ""
        while True:
            m = RULES.match(p)
            if not m or not m.group(0).strip():
                break
            pre += m.group(0)
            p = p[m.end():]
        rows.append((pre, split_cells(p), e))
    return rows


def mark_cell(old, new):
    if old is not None and norm(old) == norm(new):
        return new
    inner = re.match(r"(\s*\\multi(?:column|row)\{[^}]*\}\{[^}]*\}\{)(.*)(\}\s*)$", new, re.S)
    if inner:
        o = re.match(r"\s*\\multi(?:column|row)\{[^}]*\}\{[^}]*\}\{(.*)\}\s*$", old or "", re.S)
        return inner.group(1) + word_diff(o.group(1) if o else None, inner.group(2)) + inner.group(3)
    return word_diff(old, new)


def mark_tabular(old_body, new_body):
    old_rows = rows_of(old_body) if old_body is not None else []
    old_keys = [norm("&".join(c)) for _, c, _ in old_rows]
    used = set()
    out = []
    for pre, cells, end in rows_of(new_body):
        key = norm("&".join(cells))
        if not key:
            out.append(pre + "&".join(cells) + end)
            continue
        if key in old_keys and old_keys.index(key) not in used:     # unchanged row (possibly moved)
            used.add(old_keys.index(key))
            out.append(pre + "&".join(cells) + end)
            continue
        # pair with an edited old row only if it is clearly the same row (same first cell, or
        # nearly identical); otherwise the row is new and highlighted as a whole
        best, score = None, 0.0
        first = norm(cells[0])
        for i, ok in enumerate(old_keys):
            if i in used or not ok:
                continue
            r = difflib.SequenceMatcher(a=ok, b=key, autojunk=False).ratio()
            need = 0.6 if norm(old_rows[i][1][0]) == first else 0.85
            if r > need and r > score:
                best, score = i, r
        ocells = old_rows[best][1] if best is not None else []
        if best is not None:
            used.add(best)
        marked = [mark_cell(ocells[i] if i < len(ocells) else None, c) for i, c in enumerate(cells)]
        out.append(pre + "&".join(marked) + end)
    return "".join(out)


def tabular_body(s):
    """(start, end) of the rows of the first tabular in s (after the column spec), or None"""
    m = re.search(r"\\begin\{(tabular\*?)\}", s)
    if not m:
        return None
    i = m.end()
    for _ in range(2 if m.group(1) == "tabular*" else 1):   # [width] + column spec, brace-matched
        while s[i].isspace():
            i += 1
        i = find_arg(s, i)
    return i, s.index("\\end{" + m.group(1) + "}", i)


def balance(s):
    s = re.sub(r"\\[{}]", "", re.sub(r"(?<!\\)%[^\n]*", "", s))
    return s.count("{") - s.count("}")


def colour(new, old=None):
    s = _colour(new, old)
    if balance(s) != balance(new):
        sys.exit("brace imbalance after marking float:\n" + new[:300])
    return s


def _colour(new, old=None):
    s = new
    if s.lstrip().startswith("\\begin{figure"):
        return replace_arg(s, "caption", old)
    s = replace_arg(s, "tbl", old)
    s = replace_arg(s, "tabnote", old)
    ob = tabular_body(old) if old else None
    nb = tabular_body(s)
    if nb:
        s = s[:nb[0]] + mark_tabular(old[ob[0]:ob[1]] if ob else None, s[nb[0]:nb[1]]) + s[nb[1]:]
    elif "\\begin{tabular" in s:
        sys.exit("tabular not found in float:\n" + s[:300])
    return s


old_tex = OLD.read_text(encoding="utf8")
new_tex = NEW.read_text(encoding="utf8")
old_body, old_f = extract(old_tex)
new_body, new_f = extract(new_tex)
(WORK / "_old_ph.tex").write_text(old_body, encoding="utf8")
(WORK / "_new_ph.tex").write_text(new_body, encoding="utf8")

cmd = ["perl", f"-I{SP / 'perllib'}", str(SP / "latexdiff"), f"--preamble={WORK / 'highlight_preamble.tex'}",
       "--math-markup=coarse",
       "--graphics-markup=none", "--append-safecmd=rev", "--exclude-safecmd=tbl,RCMFLOAT",
       str(WORK / "_old_ph.tex"), str(WORK / "_new_ph.tex")]
diff = subprocess.run(cmd, capture_output=True, text=True, encoding="utf8").stdout
if "\\begin{document}" not in diff:
    sys.exit("latexdiff failed")

changed, unchanged, added = [], [], []


def put_back(m):
    k = m.group(1)
    blk = new_f[k]
    if k not in old_f:
        added.append(k)
        return colour(blk)
    if norm(old_f[k]) != norm(blk):
        changed.append(k)
        return colour(blk, old_f[k])
    unchanged.append(k)
    return blk


# active placeholders only (deleted ones are inside %DIFDELCMD comments and stay commented)
out_lines = []
for line in diff.split("\n"):
    m = re.search(r"(?<!\\)%", line)   # first unescaped % starts a comment
    code, comment = (line[:m.start()], line[m.start():]) if m else (line, "")
    out_lines.append(re.sub(r"\\RCMFLOAT\{([^}]+)\}", put_back, code) + comment)
out = "\n".join(out_lines)


def tidy_adds(s):
    # a paragraph break left at the end of \DIFadd{...} makes \hl add an empty line after
    # new paragraphs / references; move that whitespace outside the highlight
    res, pos = [], 0
    for m in re.finditer(r"\\DIFadd\{", s):
        if m.start() < pos:
            continue
        j = find_arg(s, m.end() - 1)
        arg = s[m.end():j - 1]
        core = arg.rstrip()
        res.append(s[pos:m.end()] + hl_wrap(core) + "}" + arg[len(core):])
        pos = j
    res.append(s[pos:])
    return "".join(res)


out = tidy_adds(out)
# deleted text in the original pointed to a table that never existed; show it as plain text
out = out.replace("Table~\\ref{tab:factor_summary}", "Table~[factor summary]")
removed = [k for k in old_f if k not in new_f]
OUT.write_text(out, encoding="utf8")
for p in ("_old_ph.tex", "_new_ph.tex"):
    (WORK / p).unlink()
print("floats: new", len(added), added)
print("        changed", len(changed))
print("        unchanged", len(unchanged), unchanged)
print("        removed in revision", removed)
print("left-over placeholders:", len(re.findall(r"^[^%]*\\RCMFLOAT", out, re.M)))
print("saved", OUT)
