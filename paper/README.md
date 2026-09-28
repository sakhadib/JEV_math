# Paper source

LaTeX source for the paper, built on the official ACL style files
(https://github.com/acl-org/acl-style-files).

For arXiv submission instructions and form metadata, see `ARXIV.md`.


LaTeX source for the JEV_math paper, built on the official ACL style files
(https://github.com/acl-org/acl-style-files).

## Build

```bash
pdflatex acl_latex
bibtex   acl_latex
pdflatex acl_latex
pdflatex acl_latex
```

Requires `texlive-fonts-extra` for `inconsolata.sty`.

## Files

| File | Purpose |
|---|---|
| `acl_latex.tex` | the paper |
| `custom.bib` | 9 related-work references plus 3 dataset citations |
| `acl.sty`, `acl_natbib.bst` | unmodified from acl-style-files |
| `acl_latex.bbl` | pre-built bibliography (needed for arXiv upload) |
| `make_figures.py` | regenerates every figure from the results repo |
| `figures/` | six vector PDFs (the teaser is TikZ) |
| `tables/` | aggregated CSVs from the phase-1 analysis |

## Figures

All figures are vector PDF and carry no embedded titles and no in-plot
legends: the captions do both jobs. Series are identified in the caption with
inline markers (`\mdot`, `\msq`, `\mdia`, `\mstr`, `\mbar`, `\mline`),
which are built from rules and symbols rather than TikZ, because a
`tikzpicture` inside a `\caption` breaks on TikZ's catcode changes.

Figures 1 and 2 are TikZ, drawn inline in `acl_latex.tex` rather than being
separate files. Figure 1, the teaser, plots the mean probability vector after
sorting each item's distribution descending: the numbers in it are
`[0.9455, 0.0407, 0.0100, 0.0038]`, `[0.9412, 0.0472, 0.0115]` and
`[0.5817, 0.2031, 0.1119, 0.0662, 0.0369]`, and bar heights are 10mm at
p = 1.0.

Its geometry is hand-tuned to the column: the label column runs to x=30mm,
the bar group occupies 30--48mm and the statistics start at x=51mm, which
leaves about 1.3mm inside the right text edge. Lengthening a benchmark name
or a statistic line will push it into the margin, so re-measure after any
edit rather than trusting that it still fits.

The teaser is a stepped 3D histogram: every bin is a flat quad joined by
vertical walls, which is what a histogram is, rather than a smooth surface
interpolated between bin centres. There are no wall panes and no gridlines;
the contour base is the only reference plane. `teaser()` asserts that the
histogram counts every item, because bin edges built by accumulating a
floating-point step can terminate just below 1.0 and silently drop every
item whose top probability is exactly 1.0.

```bash
python3 make_figures.py [path/to/jev_math_repo]
```

The default path is `../sakhadib/jev_math`. The script reads `derived.csv`
(phase 1), `phase_2/results_jev.jsonl`, `phase_3/results_jev.jsonl` and
`phase_3/stats.json`. None of those are bundled here because they embed
benchmark content; point the script at your own checkout.

Predictions are re-derived as an argmax over the stored distribution with
alphabetical tie-breaking, matching the analysis scripts. This matters: exact
ties rise from 8 items on phase 1 to 71 on HARP, and using the API's own
tie-break instead shifts the HARP count by 2.

## Palette

One palette across figures and table highlighting. The hex values are defined
twice, at the top of `make_figures.py` and in the `acl_latex.tex` preamble;
change both together.

| Role | Hex | Used for |
|---|---|---|
| cyan | `#00B8D4` | `raw.confidence`, correct answers |
| electric cyan | `#22F0FF` | colormap midtone |
| baby pink | `#FFAEDC` | margin signal, option spread |
| magenta | `#FF2BAF` | top-1 probability, errors, headline cells |
| yellow | `#FFD84D` | entropy signal, gold rank, cost cells |

Table cells use pale tints of the same hues (`cellmagenta`, `cellcyan`,
`cellyellow`, `cellpink`) so black body text stays legible. The four macros
are `\hi`, `\hic`, `\hiy`, `\hip`.

## Page budget

Body through Conclusion ends on page 8, which is the limit. Limitations and
Future Work follow it on page 8 and References begin on page 9. Adding more
than a few lines to Sections 1 through 7 will push the Conclusion onto page 9,
so add to the appendices instead.

## Mode switch

Line 4 sets the output mode:

- `[preprint]` (current) gives page numbers, named author, no line numbers
- `[review]` gives anonymised, line-numbered output for submission
- `[final]` gives the camera-ready version

## Before submitting

- Fill in the author block (currently "Anonymous Submission").
- Verify the BibTeX entries in `custom.bib` against the publisher pages. They
  were written from the reference list rather than copied from each venue, so
  venue, page numbers, and author lists should be confirmed.
- The failure taxonomy (Table 5, Appendix H) rests on one annotator over 80 of
  229 errors. A second annotator would strengthen it, particularly for the
  "ambiguous or mis-keyed" category.
