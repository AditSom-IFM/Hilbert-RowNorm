# Toward a First-Principles Update Geometry for the Language-Model Head

The editable LaTeX source is [`main.tex`](main.tex), with bibliography entries
in [`sample.bib`](sample.bib).

## Build on GitHub

Every push or pull request that changes this directory runs the
[`Manuscript PDF`](../.github/workflows/manuscript-pdf.yml) workflow. The
compiled PDF is available from that workflow run as the
directly downloadable `main.pdf` artifact. The workflow can also be started
manually from **Actions → Manuscript PDF → Run workflow**.

## Build locally

With `latexmk` and a TeX Live installation:

```bash
cd manuscript
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Generated LaTeX files are ignored by Git. Commit only the source files unless
there is an explicit reason to publish a fixed PDF snapshot.
