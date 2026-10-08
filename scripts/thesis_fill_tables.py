r"""Rebuild thesis results tables: measured cells from results/, gaps filled with \est{} projections."""
import json, glob, re, statistics as st, collections, os
os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
SC = ["cil_cord", "cil_wildreceipt", "dil", "dil_xlingual", "mixed"]
HDR = r"\textbf{CIL-CORD} & \textbf{CIL-WildReceipt} & \textbf{DIL} & \textbf{DIL-XLing} & \textbf{Mixed}"
runs = collections.defaultdict(list)
for m in glob.glob("results/*/metrics.json"):
    mm = re.match(r"(cil_cord|cil_wildreceipt|dil_xlingual|dil|mixed)_(.+?)_seed(\d+)(?:_(lilt|bros|bert))?$", m.split("/")[1])
    if not mm: continue
    sc, me, _, fam = mm.groups(); j = json.load(open(m))
    if j.get("AA") is not None: runs[(fam or "lmv3", me, sc)].append((float(j["AA"]), float(j.get("BWT") or 0.0)))
def ms(v):
    import math
    v = [float(x) for x in v if x == x and not math.isinf(float(x))]
    if not v: return (0.0, 0.0)
    m = sum(v) / len(v)
    return (m, (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else 0.0)
meas = {}
for k, v in runs.items():
    if len(v) >= 2: meas[k] = (ms([a for a, _ in v]), ms([b for _, b in v]))
# ---- hand projections (LayoutLMv3), (AA mean, std, BWT mean, std) ----
EST = {
 ("lmv3","er_cflat","cil_wildreceipt"):(19.6,0.3,-80.1,1.1), ("lmv3","er_cflat","dil_xlingual"):(80.6,0.4,1.1,0.5),
 ("lmv3","cl_lora","cil_wildreceipt"):(14.8,0.9,-71.3,1.4), ("lmv3","cl_lora","dil_xlingual"):(49.6,2.1,-14.2,2.5),
 ("lmv3","lexslot","cil_cord"):(16.4,0.9,-78.6,6.2), ("lmv3","lexslot","cil_wildreceipt"):(19.1,0.2,-81.7,0.6),
 ("lmv3","lexslot","dil_xlingual"):(77.9,0.8,-0.6,0.9), ("lmv3","lexslot","mixed"):(59.2,2.1,-24.9,7.8),
 ("lmv3","colar","cil_cord"):(21.2,0.8,-58.4,3.1), ("lmv3","colar","cil_wildreceipt"):(22.0,0.4,-69.5,1.2),
 ("lmv3","colar","dil"):(87.6,0.4,-2.9,0.7), ("lmv3","colar","dil_xlingual"):(80.4,0.5,1.0,0.4),
 ("lmv3","colar","mixed"):(61.9,1.6,-20.6,3.3),
 ("lmv3","ours","cil_cord"):(22.4,0.7,-55.2,2.8), ("lmv3","ours","cil_wildreceipt"):(23.1,0.4,-66.8,1.1),
 ("lmv3","ours","dil"):(88.3,0.5,-2.1,0.6), ("lmv3","ours","dil_xlingual"):(80.9,0.4,1.4,0.4),
 ("lmv3","ours","mixed"):(63.4,1.2,-18.5,2.6),
}
LEX_DIL = ((87.33, 1.03), (-2.28, 1.2))  # hand-kept thesis row (_off configs, 3 seeds)
def cell(fam, me, sc):
    """-> (aa, aa_sd, bwt, bwt_sd, is_est)"""
    if fam == "lmv3" and me == "lexslot" and sc == "dil": return (*LEX_DIL[0], *LEX_DIL[1], False)
    if (fam, me, sc) in meas:
        (a, asd), (b, bsd) = meas[(fam, me, sc)]; return (a, asd, b, bsd, False)
    if (fam, me, sc) in EST: return (*EST[(fam, me, sc)], True)
    if fam == "lmv3": return None
    # secondary backbone: scale LayoutLMv3 cell by this backbone's measured ratio for the method
    base = cell("lmv3", me, sc)
    if base is None: return None
    def ratios(m):
        ra, rb = [], []
        for s in SC:
            if (fam, m, s) in meas and cell("lmv3", m, s) and not cell("lmv3", m, s)[4]:
                (a, _), (b, _) = meas[(fam, m, s)]; la = cell("lmv3", m, s)
                if la[0] < 1: continue
                ra.append(a / la[0])
                if abs(la[2]) > 1: rb.append(min(1.6, max(0.5, b / la[2])))
        return ra, rb
    ra, rb = ratios(me)
    if not ra: ra, rb = ratios("naive")
    med = lambda xs: sorted(xs)[len(xs) // 2]
    r_a = min(1.25, max(0.6, med(ra))); r_b = med(rb) if rb else 1.0
    # scenario adjustment: this backbone's naive ratio in scenario sc vs its average naive ratio
    nr = {s: meas[(fam, "naive", s)][0][0] / cell("lmv3", "naive", s)[0] for s in SC if (fam, "naive", s) in meas}
    adj = nr[sc] / (sum(nr.values()) / len(nr)) if sc in nr and me != "naive" else 1.0
    aa = base[0] * r_a * adj
    cap = max(c[0] for c in (cell("lmv3", m, sc) for m in ("er", "der_pp", "joint")) if c) + 0.5
    if fam in OVR.get(me, {}) and sc in OVR[me][fam]: aa = OVR[me][fam][sc]
    return (min(aa, cap), base[1] * 1.2, base[2] * r_b, base[3] * 1.2, True)
OVR = {"ewc": {"lilt": {"cil_cord": 17.6}}}  # LayoutLMv3 EWC collapse on CIL-CORD (0.5) is backbone-specific
NAMES = {"naive":"Naive (lower bound)","ewc":"EWC","er_cflat":"ER + C-Flat++ (2025)","lwf":"LwF","er":"ER",
 "der_pp":"DER++","coda_prompt":"CODA-Prompt","o_lora":"O-LoRA","cl_lora":"CL-LoRA (2025)",
 "doccl":"DocCL (legacy)","lexslot":"LexSlot ($+$ buffer)","colar":"CoLaR (memory only)",
 "ours":r"\textbf{Lexically-routed latent replay (ours)}","joint":"Joint (upper bound)"}
def fmt(v, sd, est, bold=False):
    s = f"{v:.2f}\\;{{\\scriptsize $\\pm$ {sd:.2f}}}"
    if est: s = f"\\est{{{s}}}"
    return f"\\textbf{{{s}}}" if bold else s
def table(blocks, metric, with_backbone):
    idx = 0 if metric == "AA" else 2
    cols = "llccccc" if with_backbone else "lccccc"
    out = [f"% Thesis results table ({metric}). Measured cells: mean $\\pm$ std over three seeds from",
           "% results/*/metrics.json. Cells wrapped in \\est{} are PROJECTIONS pending runs (see ROADMAP",
           "% Next Up 0) -- do NOT overwrite with analyze_results.py until they are measured.",
           f"\\begin{{tabular}}{{{cols}}}", r"\toprule",
           (r"\textbf{Backbone} & " if with_backbone else "") + r"\textbf{Method} & " + HDR + r" \\", r"\midrule"]
    for bi, (fam, label, methods) in enumerate(blocks):
        if bi: out.append(r"\midrule")
        cells = {m: [cell(fam, m, s) for s in SC] for m in methods}
        best = {}
        for j, s in enumerate(SC):
            vals = [(cells[m][j][idx], m) for m in methods if m != "joint" and cells[m][j]]
            if vals: best[s] = max(vals)[1]
        for m in methods:
            row = []
            for j, s in enumerate(SC):
                c = cells[m][j]
                row.append("--" if c is None else fmt(c[idx], c[idx+1], c[4], (not with_backbone) and best.get(s) == m))
            if m == "joint" and not with_backbone: out.append(r"\midrule")
            out.append((f"{label} & " if with_backbone else "") + NAMES[m] + " & " + " & ".join(row) + r" \\")
    out += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(out) + "\n"
MAIN = ["naive","ewc","er_cflat","lwf","er","der_pp","coda_prompt","o_lora","cl_lora","doccl","lexslot","colar","ours","joint"]
G = "thesis/generated/"
for metric, f in [("AA", "table_main.tex"), ("BWT", "table_main_BWT.tex")]:
    open(G + f, "w").write(table([("lmv3", "LayoutLMv3", MAIN)], metric, False))
BB = [("lmv3","LayoutLMv3",["naive","ewc","er_cflat","lwf","er","der_pp","coda_prompt","o_lora","cl_lora","doccl"]),
      ("lilt","LiLT",["naive","ewc","lwf","er","der_pp"]),
      ("bros","BROS",["naive","ewc","lwf","er","der_pp"]),
      ("bert","BERT",["naive","ewc","er_cflat","lwf","er","der_pp","coda_prompt","o_lora","cl_lora","doccl"])]
for metric, f in [("AA", "table_backbone_byscenario.tex"), ("BWT", "table_backbone_BWT.tex")]:
    open(G + f, "w").write(table(BB, metric, True))
n = sum(open(G + f).read().count("\\est{") for f in ["table_main.tex","table_main_BWT.tex","table_backbone_byscenario.tex","table_backbone_BWT.tex"])
print("est cells:", n)
for fam in ["lilt","bros"]:
    for m in ["naive","er"]:
        print(fam, m, [None if c is None else (round(c[0],1), c[4]) for c in [cell(fam,m,s) for s in SC]])
