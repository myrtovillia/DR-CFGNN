import os, glob, csv

PRETTY = {
    "ba_2motifs": "BA-2Motifs",
    "ba_2motifs_3class": "BA-2Motifs-3Classes",
    "ba_3motifs": "BA-3Motifs",
    "ba_4motifs": "BA-4Motifs",
    "bbbp": "BBBP",
    "twitter": "Twitter"
}

ROW_ORDER = ["ba_2motifs","ba_2motifs_3class","ba_3motifs","ba_4motifs","bbbp", "twitter"]

TOTALS = {"ba_2motifs":98,"ba_2motifs_3class":82,"ba_3motifs":94,"ba_4motifs":94,"bbbp":168, "twitter": 489}


COLS = ["random","global","cf2","dr4explainer","rc","gist","(-DEN,-OH)","(+DEN,-OH)","(-DEN,+OH)","(+DEN,+OH)"]

TAB_SPEC = "p{3cm}|" + "|".join(["c"]*10)
HEADER = (
    " & \\textbf{random} & global & cf2 & dr4explainer & rc & gist"
    " & \\textbf{(-DEN,-OH)} & \\textbf{(+DEN,-OH)} & \\textbf{(-DEN,+OH)} & \\textbf{(+DEN,+OH)} \\\\\n"
)

here = os.path.dirname(os.path.abspath(__file__))

def short2(top, bottom):
    return f"\\shortstack{{\\scriptsize {top}\\\\\\scriptsize {bottom}}}"

def pn_pct(ds, pn):
    return f"{float(pn)/TOTALS[ds]:.3f}\\%"

def load_prints(ds):
    path = os.path.join(here, f"prints_{ds}.csv")
    with open(path, newline="") as f:
        return list(csv.DictReader(f))

def pick(rows, den_label, oh_label):
    for r in rows:
        if r["DEN"] == den_label and r["OH"] == oh_label:
            return r
    return None 
        
def row_for_metric(ds, metric):
    rows = load_prints(ds)

    mapping = {
        "random": ("RANDOM","RANDOM"),
        "(-DEN,-OH)": ("-DEN","-OH"),
        "(+DEN,-OH)": ("+DEN","-OH"),
        "(-DEN,+OH)": ("-DEN","+OH"),
        "(+DEN,+OH)": ("+DEN","+OH"),
    }

    out = []
    for c in COLS:
        if c in ["global","cf2","dr4explainer","rc","gist"]:
            out.append("")  
            continue

        den, oh = mapping[c]
        r = pick(rows, den, oh)

        if metric == "pn":
            top = f"{pn_pct(ds, r['PN'])}, {r['fidelity']}"
            bottom = f"{r['exp_size']} / {r['first_exp_size']}"
            out.append(short2(top, bottom))

        elif metric == "min":
        	out.append(f"\\scriptsize {r['minimality']}")


        elif metric == "motif":
            out.append(short2(r["motif_prox"], r["best_motif_prox"]))

    return out

def write_table(fname, caption, label, metric):
    out_tex = os.path.join(here, fname)
    with open(out_tex, "w") as f:
        f.write("\\begin{table}[t]\n")
        f.write(f"\\caption{{{caption}}}\\label{{{label}}}\n")
        f.write("\\centering\n")
        f.write("\\resizebox{\\linewidth}{!}{%\n")
        f.write(f"\\begin{{tabular}}{{{TAB_SPEC}}}\n")
        f.write("\\hline\n")
        f.write(HEADER)
        f.write("\\hline\n")

        for ds in ROW_ORDER:
            name = PRETTY.get(ds, ds)
            f.write(" & ".join([name] + row_for_metric(ds, metric)) + " \\\\\n")
            f.write("\\hline\n")

        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")

if __name__ == "__main__":
    write_table(
        "pn_table.tex",
        "PN/Fidelity (top), explanation size / first explanation size (bottom).",
        "tab:pn_sizes",
        "pn",
    )
    write_table(
        "minimality_table.tex",
        "Minimality: avg\\_comb\\_score.",
        "tab:minimality",
        "min",
    )
    write_table(
        "motif_table.tex",
        "Motif proximity: avg / best.",
        "tab:motif",
        "motif",
    )

