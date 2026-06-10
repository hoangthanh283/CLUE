#!/usr/bin/env bash
# Export every draw.io figure source to a single-page cropped PDF in thesis/figures/.
#
#   usage:  bash figures_src/build_drawio.sh      (from thesis/)
#           bash build_drawio.sh                  (from thesis/figures_src/)
#
# Requires draw.io desktop (app.diagrams.net) installed + poppler's pdfseparate.
# draw.io's --crop can emit a trailing blank page, so we keep page 1 (the content)
# with pdfseparate, which preserves vector quality. Committed figures/*.pdf are what
# the thesis (and Overleaf) include via \graphicspath{{figures/}}.
#
# IMPORTANT: the draw.io binary is an Electron app. If ELECTRON_RUN_AS_NODE=1 is set
# in the environment (some shells/IDEs export it), the binary runs as plain Node and
# silently rejects --export ("bad option"), leaving stale PDFs. We strip that var
# with `env -u` and pass an ABSOLUTE input path (Electron's cwd is not the shell's).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
SRC="$HERE/drawio"
OUT="$HERE/../figures"
DRAWIO="${DRAWIO:-/Applications/draw.io.app/Contents/MacOS/draw.io}"
[ -x "$DRAWIO" ] || { echo "draw.io CLI not found at $DRAWIO (set \$DRAWIO)"; exit 1; }
command -v pdfseparate >/dev/null || { echo "pdfseparate (poppler) not found"; exit 1; }

drawio_export() {  # $1 = absolute .drawio in, $2 = out pdf
  env -u ELECTRON_RUN_AS_NODE "$DRAWIO" --export --format pdf --crop --border 8 -o "$2" "$1"
}

shopt -s nullglob
# Figures exported by the loop below (alphabetical):
#   candidate_a_lapp_hlora   candidate_b_c          cl_taxonomy
#   diagnostic_conditions    diagnostic_pipeline    doc_pipeline
#   forgetting_matrix_schematic  layoutlmv3_components  method_selection_tree
#   scenario_cil_cord        scenario_dil           scenario_mixed
#   stability_plasticity     system_architecture
n=0
for f in "$SRC"/*.drawio; do
  name="$(basename "${f%.drawio}")"
  echo "==> ${name}"
  # must be an absolute path for Electron; emit a clear error instead of a silent stale PDF
  if ! drawio_export "$f" "/tmp/${name}_raw.pdf" 2>&1 | grep -q -- '->'; then
    echo "   ERROR: export failed for ${name} (is ELECTRON_RUN_AS_NODE set? is draw.io importable?)"; exit 1
  fi
  pdfseparate -f 1 -l 1 "/tmp/${name}_raw.pdf" "$OUT/${name}.pdf"
  rm -f "/tmp/${name}_raw.pdf"
  n=$((n+1))
done
echo "Exported ${n} draw.io figures into thesis/figures/"
