import os, glob, csv

PRETTY = {
    "ba_2motifs": "BA-2Motifs",
    "ba_2motifs_3class": "BA-2Motifs-3Classes",
    "ba_3motifs": "BA-3Motifs",
    "ba_4motifs": "BA-4Motifs",
    "bbbp": "BBBP",
    "twitter": "Twitter",
    "graph_sst5": "Graph-sst5"
}

ROW_ORDER = ["ba_2motifs","ba_2motifs_3class","ba_3motifs","ba_4motifs","bbbp", "twitter", "graph_sst5" ]

TOTALS = {"ba_2motifs":98,"ba_2motifs_3class":82,"ba_3motifs":94,"ba_4motifs":94,"bbbp":168, "twitter": 489, "graph_sst5":1116}


COLS = ["Random","Global","CF^{2}","D4Explainer","RSGG-CE","GIST","(-DeN,-OH)","(+DeN,-OH)","(-DeN,+OH)","(+DeN,+OH)"]

TAB_SPEC = "p{3cm}" + "c"*10
HEADER = (
    " & Random & Global & CF^{2} & D4Explainer & RSGG-CE & GIST"
    " & (-DeN,-OH) & (+DeN,-OH) & (-DeN,+OH) & (+DeN,+OH) \\\\\n"
)

here = os.path.dirname(os.path.abspath(__file__))

def short2(top, bottom):
    return f"\\shortstack{{\\scriptsize {top}\\\\\\scriptsize {bottom}}}"

def pn_pct(ds, pn):
    return f"{float(pn)/TOTALS[ds]:.3f}"

def load_prints(ds):
    path = os.path.join(here, f"metrics_{ds}.csv")
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
        "Random": ("RANDOM","RANDOM"),
        "(-DeN,-OH)": ("-DEN","-OH"),
        "(+DeN,-OH)": ("+DEN","-OH"),
        "(-DeN,+OH)": ("-DEN","+OH"),
        "(+DeN,+OH)": ("+DEN","+OH"),
    }

    out = []
    for c in COLS:
        if c in ["Global","CF^{2}","D4Explainer","RSGG-CE","GIST"]:
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
        f.write("\\toprule\n")
        f.write(HEADER)
        f.write("\\midrule\n")

        for ds in ROW_ORDER:
            name = PRETTY.get(ds, ds)
            f.write(" & ".join([name] + row_for_metric(ds, metric)) + " \\\\\n")

        f.write("\\bottomrule\n")
        f.write("\\end{tabular}%\n")
        f.write("}\n")
        f.write("\\end{table}\n")

if __name__ == "__main__":
    write_table(
        "table_validity.tex",
        "Validity/Fidelity (top), Explanation Size / Explanation Size of the 1st cf (bottom).",
        "tab:pn_sizes",
        "pn",
    )
    write_table(
        "table_minimality.tex",
        "Minimality: avg\\_comb\\_score.",
        "tab:minimality",
        "min",
    )
    write_table(
        "table_motif_proximity.tex",
        "Motif proximity: avg / best.",
        "tab:motif",
        "motif",
    )

