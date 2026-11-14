#!/usr/bin/env bash
# Split the merged build (main.pdf, appendix after the references) into the two CVPR uploads,
# the kit's way: compile together, then split. Needs main.aux from the same build.
set -euo pipefail
cd "$(dirname "$0")"
pg() { sed -n "s/^\\\\newlabel{$1}{{[^}]*}{\([0-9]*\)}.*/\1/p" main.aux | head -1; }
end=$(pg mainend); app=$(pg app:setup); total=$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')
[ -n "$end" ] && [ -n "$app" ] || { echo "FAIL: labels mainend/app:setup missing in main.aux" >&2; exit 1; }
(( end <= 8 )) || { echo "FAIL: main text ends on page $end (> 8)" >&2; exit 1; }
mkdir -p submission
# gs stderr dropped: it only warns about outline links into the other half (gs still fails on errors)
split() { gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -dFirstPage="$1" -dLastPage="$2" -sOutputFile="$3" main.pdf 2>/dev/null; }
split 1 $((app - 1)) submission/main_paper.pdf
split "$app" "$total" submission/supplementary.pdf
pdftotext submission/main_paper.pdf - | grep -c "LLM Usage Statement" >/dev/null \
  || { echo "FAIL: main_paper.pdf has no LLM Usage Statement" >&2; exit 1; }
echo "main text ends on p$end; appendix starts on p$app of $total"
for f in submission/main_paper.pdf submission/supplementary.pdf; do
  echo "$f: $(pdfinfo "$f" | awk '/^Pages:/{print $2}') pages, $(du -h "$f" | cut -f1)"
done
