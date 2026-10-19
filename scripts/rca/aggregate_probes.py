"""Pass 2b aggregator: read probes.json across cells, update the RCA table."""
import json, glob, pathlib
rows = []
for p in sorted(glob.glob("results/*/probes.json")):
    d = json.load(open(p))
    name = pathlib.Path(p).parent.name
    rows.append(dict(name=name, **d))

out = pathlib.Path("rca/pass2b_probe_table.md")
with open(out, "w") as fo:
    fo.write("# Pass 2b — Ridge decomposition from probes.json\n\n")
    fo.write("| cell | modality | head | rp_full | rp_stored | rp_frozen | ncm_full | ncm_stored | head_gap | trunk_gap | coverage_gap |\n")
    fo.write("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for r in rows:
        def f(x): return f"{x:.1f}" if isinstance(x, (int, float)) and x is not None else "—"
        fo.write(f"| {r['name']} | {r.get('modality','?')} | {f(r.get('head'))} | {f(r.get('rp_full'))} | "
                 f"{f(r.get('rp_stored'))} | {f(r.get('rp_full_frozen'))} | {f(r.get('ncm_full'))} | "
                 f"{f(r.get('ncm_stored'))} | {f(r.get('head_gap'))} | {f(r.get('trunk_gap'))} | "
                 f"{f(r.get('coverage_gap'))} |\n")
print(f"wrote {len(rows)} probes -> {out}")
