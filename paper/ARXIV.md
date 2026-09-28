# arXiv submission

Upload `arxiv_submission.tar.gz`. It contains `acl_latex.tex`, `acl.sty`,
`acl_natbib.bst`, `acl_latex.bbl`, `custom.bib` and the six figure PDFs.

It has been test-compiled from a clean directory with pdfLaTeX alone and no
BibTeX run, which is how arXiv processes a submission that ships its own
`.bbl`: 11 pages, no undefined references, no unresolved citations, all fonts
Type 1 and embedded.

## Form fields

**Title**

    Certainty Collapses, Calibration Holds: Typed Confidence in a
    Non-Reasoning Decision Model Across Three Mathematics Benchmarks

**Authors**

    Adib Sakhawat

**Affiliation** (not a form field on arXiv, but it is on the title page)

    Department of Computer Science and Engineering,
    Islamic University of Technology, Gazipur, Bangladesh

**Categories**

| Slot | Category | Why |
|---|---|---|
| Primary | `cs.CL` | Language-model evaluation on natural-language mathematics items |
| Cross-list | `cs.LG` | Calibration and selective prediction are the core methods |
| Cross-list | `cs.AI` | Optional; include only if you want the wider audience |

**Comments**

    11 pages, 7 figures, 6 tables. Code and aggregated results:
    https://github.com/sakhadib/JEV_math

**License.** CC BY 4.0 is the usual choice for a preprint you want cited and
reused. arXiv's non-exclusive licence is the more conservative alternative.

**ACM class** (optional): `I.2.7; I.2.6`

## Abstract, as plain text

Paste this into the abstract box. arXiv renders a limited subset of TeX in
abstracts, so the mathematics has been flattened to words and the citation
commands removed.

```
A System One model returns a typed decision with a probability distribution in a single forward pass: no chain of thought, no sampling, no text to parse. We evaluate Jev-1.13, the first such model, on 13,110 multiple-choice mathematics items drawn from three benchmarks of increasing difficulty, against three chat models run on identical items under a matched zero-shot direct-choice protocol. Accuracy alone tells a discouraging story: Jev scores 96.94%, 94.89% and 54.70%, is statistically tied with a free stealth model on the two easier benchmarks, and is beaten by 26.6 points on the hardest. Its confidence behaves very differently. As the benchmarks harden, the fraction of items on which Jev commits fully collapses from 53.4% to 5.2%, while accuracy on those committed items stays at 100.00%, 99.79% and 99.06%; across all three it answered at maximum confidence 5,155 times for 4 errors, of which blind adjudication found 3 to be benchmark key errors rather than model failures. Selective prediction inherits this: deferring the least-confident 10% of items lifts accuracy to 99.70% on the easiest benchmark but only to 58.04% on the hardest, because on hard data there is little confidence to threshold on. We also find that positional bias is not a fixed model property but emerges under uncertainty: Jev's predicted-option marginals match the answer key on both easier benchmarks and diverge sharply on the hardest (p = 2.3 x 10^-15), with the divergence concentrated in low-confidence items and absent in high-confidence ones. We argue that a confidence signal which degrades in step with capability, rather than one that stays high while accuracy falls, is the property worth building on, and that evaluating calibration on benchmarks where every strong system exceeds 94% cannot detect it.
```

## Before you click submit

- The name is set as "Adib Sakhawat", so BibTeX will treat Sakhawat as the
  family name. If that is backwards, fix it in `acl_latex.tex` and in the
  `docs/index.html` citation block, which use the same ordering.
- arXiv holds the first submission from a new account for moderation, which
  usually clears within a business day or two.
- The repository is public and the paper's footnote points at it, so make sure
  it is in the state you want before the paper is announced.
