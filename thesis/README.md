# DocCL — Master's Thesis (LaTeX)

LaTeX sources for the Master's thesis *"Where Does a Document Encoder Forget? A
Per-Component Diagnosis of Catastrophic Forgetting in LayoutLMv3 and a
Mechanism-Targeted Remedy"* (Thanh Hoang, HUST).

Built on the **official HUST thesis class** `hust-thesis` (Ngoc Bui, CC BY 4.0).

## Compiler

**XeLaTeX** (the class uses `fontspec`/`mathspec` and bundled OTF fonts) + **BibTeX**.
On Overleaf set *Menu → Compiler → XeLaTeX*. The build verified locally with:

```bash
latexmk -xelatex -interaction=nonstopmode main.tex
```

→ 48-page PDF, 0 undefined references.

> Note on fonts: the class sets `\setmainfont{Times New Roman}` and loads Arno Pro
> from `fonts/`. Overleaf and TeX Live resolve these; if a future Overleaf image
> lacks Times New Roman, swap it for a bundled serif (e.g. `\setmainfont{TeX Gyre
> Termes}`) in `hust-thesis.cls`.

## Loading into Overleaf

- **Option A (zip upload):** zip the contents of this `thesis/` folder (including
  `hust-thesis.cls`, `packages/`, `fonts/`) and upload via *New Project → Upload
  Project*. Set the compiler to XeLaTeX and `main.tex` as the main document.
- **Option B (GitHub sync):** point Overleaf at this repository and set the
  document root to `thesis/main.tex`.

## Structure

```
main.tex              entry point: metadata + \input of all parts (XeLaTeX)
refs.bib              BibTeX references (plainnat / natbib authoryear)
frontmatter/          coverpage, thanks, abstract, glossaries (acronyms)
chapters/             chapter1..chapter7
appendices/           appendixA
figures/              figure assets (copy pilot figures here for the build)
hust-thesis.cls       official HUST class (exported from Overleaf)
packages/, fonts/     class dependencies (do not remove)
Thesis_template_..../ pristine official template, kept for reference
```

## Drafting status

| Part | Status |
|------|--------|
| Ch.1 Introduction | drafted (achievement numbers are `TODO`) |
| Ch.2 Background & Literature Review | drafted from the CL4IE wiki |
| Ch.3 Methodology | framework drafted; **selected method + Algorithm 2 are `TODO` until the Week-4 pilot decision** |
| Ch.4 Implementation | drafted |
| Ch.5 Experimental Setup | drafted |
| Ch.6 Results & Discussion | scaffold only — table/figure shells in place, **numbers `TODO`** (need experiments) |
| Ch.7 Conclusions | drafted (headline finding paragraph `TODO`) |
| Abstract | drafted; quantitative blanks `TODO` |
| Appendix A | hyperparameter/metric stubs; extended result tables `TODO` |

Search the sources for `TODO` to find every gap. No fabricated results are present
— experiment-dependent values are explicitly left as `TODO`.

## Filling results

Pilot/grid figures are produced by the analysis scripts under
`../results/.../figures/`. Copy the final PDFs into `figures/` and replace the
placeholder boxes in `chapters/chapter6.tex`; fill the `TODO` table cells from the
aggregated results.
