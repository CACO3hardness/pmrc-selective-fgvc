# Manuscript package validation

## 2026-10-27 revision status

- The source now uses the publication-style structure: fixed-accuracy
  selective reliability gap -> monotone-calibration counterexample -> PMRC ->
  Q1/Q2/Q3 evidence -> bounded Discussion. It includes matched three-seed CE,
  ELR, and GCE transfer, three strict CUB open-set classifiers, and three
  Stanford Dogs downstream dataset/split classifiers.
- The completed research set now contains 53 run directories.
- Static checks pass: braces balance, all LaTeX environments match, all
  citation keys resolve in `references.bib`, and all internal references have
  labels.
- The Chinese complete manuscript was rebuilt as a new 12-page DOCX and
  rendered to a 2026-10-27 preview PDF. Every page was visually inspected after
  the final wording and pagination changes.
- The Codex sandbox cannot execute the installed MiKTeX binaries. Therefore
  `main.pdf` and `supplementary.pdf` are previous 2026-10-26 outputs and must
  be rebuilt from the current sources before submission.

## Post-compression static audit

The source was checked after moving detailed analyses to SI and after the
publication-style rewrite. Both TeX files
have balanced braces and matched table/figure environments; all internal
  labels resolve, citation keys match `references.bib`, and all four main-text
  figure files exist. The main source now contains four figures, four tables,
  and no in-manuscript appendix.

## 2026-10-27 publication-rewrite audit

- English abstract: 181 words after concentrating the abstract on the mainline
  evidence and removing secondary numerical details.
- Citation set: 21 used keys and 21 bibliography entries; no missing key.
- Main LaTeX: brace balance zero, no unmatched environments, 15 labels, seven
  references, and no unresolved internal reference.
- Supplementary LaTeX: brace balance zero, no unmatched environments, and 19
  labels.
- The defensive-writing audit is stored in
  `PMRC_defensive_writing_audit_2026-10-27.md`.
- The Chinese complete manuscript is
  `PMRC_中文重排版_2026-10-27_v2.docx`.
- Unit test suite: 18 tests passed.

The 2026-10-26 source and PDFs were compiled after the three-seed ResNet-18
extension and the updated Figure 1. All pages were then rendered and visually
inspected. Those PDF checks do not apply to the later 2026-10-27 source edits.

## Static LaTeX checks

- Main manuscript: balanced braces, matched environments, 21 citation keys,
  no missing labels or references, and all four referenced PDF figures present.
- Supplementary manuscript: balanced braces, matched environments, and its
  referenced PDF figure present.
- Bibliography: no duplicate keys and no missing keys used by `main.tex`.

The last compilation used MiKTeX's native `pdflatex` and `bibtex`. It contained
no unresolved references, missing citations, overfull boxes, underfull boxes,
missing glyphs, or BibTeX warnings.

## Figure checks

| Figure | Minimum rendered font | Alignment | Collision |
|---|---:|---|---|
| Figure 1, method overview | 6.2 pt source | PASS | PASS |
| Figure 2, controlled selective effect | 6.0 pt source | PASS | PASS |
| Figure 3, baseline and prototype mechanism | 6.0 pt source | PASS | PASS |
| Figure 4, WebFG gate | 6.0 pt source | PASS | PASS |
| Supplementary Figure S1, stability | 5.7 pt | PASS | PASS |
| Supplementary Figure S2, method overview | 6.2 pt | PASS | PASS |

## Compiled PDF checks

- `main.pdf`: 13 A4 pages; minimum embedded font size 5.25 pt.
- `supplementary.pdf`: 7 A4 pages; minimum embedded font size 7.97 pt.
- Every page of both PDFs was rendered and visually reviewed.
- Both logs contain no unresolved references, missing citations, overfull
  boxes, underfull boxes, ignored box errors, or missing glyphs.

These compiled-PDF checks refer to the earlier 2026-10-26 sources. The current
2026-10-27 English sources still require a user-side MiKTeX compile. The Codex
environment has neither `pdflatex` on PATH nor LibreOffice/Word automation; the
Chinese DOCX was therefore checked with the project's reportlab/PyMuPDF preview
pipeline. The final Word rendering should still be reopened once by the author.

## Image substitution check

The figures in `paper/figures` are copies of the audited source figures in
`outputs/figures`. The manuscript references only these copied files.

## 2026-10-27 final packages

- Manuscript package:
  `pmrc_manuscript_publication_rewrite_2026-10-27.zip`
- AI-review package:
  `AI_review_bundle_publication_2026-10-27.zip`
- Package checksums:
  `pmrc_publication_packages_2026-10-27.sha256`
- Full research snapshot:
  `fg_semantic_noise_snapshot_2026-10-27.zip`

## Pattern Recognition version

- `main_pr.tex` uses `elsarticle` review mode with Times font, single-column
  double spacing, line numbers, page numbers, and the supplied local
  `elsarticle.cls`.
- The abstract is 161 words, below the journal limit of 250, and the contribution
  sentence appears once.
- The PR bibliography contains 35 entries, and all 35 are cited.
- All four referenced figure PDFs exist.
- All five highlights are between 62 and 71 characters.
- The local Codex sandbox does not permit execution of the installed MiKTeX
  binary. Run `compile_pr.ps1` from the author's normal PowerShell session to
  confirm the required 20-35 page count and inspect the final PDF.
- A user-side compile produced 28 pages, within the 20-35 page range. After
  compacting the table layouts, the main manuscript log contains no overfull
  boxes, oversized floats, or unresolved citations.
- The final user-side compile produced 28 main pages and 10 supplementary
  pages with no overfull boxes, oversized floats, or unresolved citations.
  The four keywords are visible in full on the title page.
- A later humanizer pass tightened the abstract and removed repeated
  evaluative closers without changing numerical results or claim boundaries.
- The method overview was regenerated from Python via the installed
  `diagram-design` skill. Its HTML passes the packaged offline self-check, and
  the HTML, SVG, and PDF are retained in both the figures directory and the
  manuscript package.
