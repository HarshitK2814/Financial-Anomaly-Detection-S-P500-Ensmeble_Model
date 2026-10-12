import sys
BS = chr(92)
s = open(sys.argv[1], encoding="utf8").read()
body = s[s.index(BS + "begin{document}"):]
for p in body.split(BS + "begin{table}")[1:]:
    t = p.split(BS + "end{table}")[0]
    t2 = t.replace(BS + "{", "").replace(BS + "}", "")
    lab = t.split(BS + "label{")[1].split("}")[0]
    rows = [r for r in t.split(BS + BS) if "&" in r]
    amp = sorted({r.count("&") - r.count(BS + "&") for r in rows})
    print(f"{lab:28s} braces={t2.count('{') - t2.count('}')} amps/row={amp}")
