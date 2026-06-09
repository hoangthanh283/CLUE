# DocCL — Master's Thesis (LaTeX)

LaTeX sources for the Master's thesis *"Where Does a Document Encoder Forget? A
Per-Component Diagnosis of Catastrophic Forgetting in LayoutLMv3 and a
Mechanism-Targeted Remedy"* (Thanh Hoang, HUST).

Built on the **official HUST thesis class** `hust-thesis` (Ngoc Bui, CC BY 4.0).

## Compiler

**XeLaTeX** (the class uses `fontspec`/`mathspec`, which require XeTeX, plus
bundled OTF fonts) + **BibTeX**. A `latexmkrc` (`$pdf_mode = 5`) forces XeLaTeX, so
the project builds correctly on Overleaf and locally without any manual setting.

### Compile locally (no Overleaf)

You need a TeX distribution with XeLaTeX (e.g. MacTeX / TeX Live). Times New Roman
must be available as a system font (it is on macOS by default).

```bash
cd thesis
latexmk -interaction=nonstopmode main.tex   # latexmkrc selects xelatex + bibtex + glossaries
# output: thesis/main.pdf      (clean aux files with:  latexmk -c)
```

### Compile via GitHub CI (no Overleaf, no local TeX)

`.github/workflows/build-thesis.yml` builds the PDF on every push that touches
`thesis/**` (and on manual *Run workflow*). It installs TeX Live + Times New Roman
in the runner and uploads the result. Download it from the Actions run page →
**Artifacts → `thesis-pdf`**. No timeout, free for public repos.

> On Overleaf, if you still see an `\RequireXeTeX` / `mathspec` "Emergency stop",
> the project is being built with pdfLaTeX — set *Menu → Compiler → XeLaTeX* and
> recompile (the bundled `latexmkrc` should make this unnecessary).

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
