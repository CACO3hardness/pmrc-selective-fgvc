# Reference verification

Verification date: 2026-10-27

Scope: all 21 references used by the current main manuscript. The check covered
the author list and order, title, venue, year, volume/article number, pages,
and DOI or arXiv identifier where applicable. Sources were Crossref DOI records,
official NeurIPS proceedings, PMLR, arXiv, OpenReview, CaltechAUTHORS, and
OpenAlex. Topic relevance was checked separately from bibliographic accuracy.

## Result

- Critical metadata errors found and corrected: 4
- Formal metadata or DOI/pages added or corrected: 7
- Weak source replaced: 1
- Verified with an online/issue-year distinction: 1
- Original workshop record not directly reopened, but dataset identity and
  authors confirmed by the authors' frozen Stanford Dogs provenance: 1

## Entry-by-entry check

| # | Key | Status after correction | Finding |
|---:|---|---|---|
| 1 | `wei2022cifar100n` | Corrected | The correct title is "Learning with Noisy Labels Revisited: A Study Using Real-World Human Annotations", not "A Study on Deep Neural Networks". ICLR 2022 is correct. |
| 2 | `sun2021webfg` | Verified | ICCV 2021, pp. 10582-10591, DOI `10.1109/ICCV48922.2021.01043`; authors and WebFG-496 scope match. |
| 3 | `liu2020elr` | Verified | NeurIPS 2020, volume 33, pp. 20331-20342; official proceedings author list matches. |
| 4 | `geifman2017selective` | Verified | NeurIPS 2017, volume 30; authors and title match the official proceedings. |
| 5 | `rabanser2025selectivegap` | Verified | NeurIPS 2025, pp. 87431-87468, DOI `10.52202/085713-2635`; arXiv `2510.20242` was confirmed directly through the arXiv API. |
| 6 | `hendrycks2017msp` | Verified | ICLR 2017; authors and title match arXiv `1610.02136` and the ICLR publication note. |
| 7 | `guo2017calibration` | Verified | ICML 2017, PMLR 70:1321-1330; author list and venue match. |
| 8 | `cattelan2024fix` | Verified and completed | UAI 2024, PMLR 244:547-584; pages, volume, publisher, and URL were added. |
| 9 | `jiang2018trust` | Corrected | The fourth author is Maya Gupta, not Manik Gupta. NeurIPS 2018, volume 31 is correct. |
| 10 | `snell2017prototypical` | Verified | NeurIPS 2017, volume 30; authors and title match. Used as background for class prototypes, not as the direct PMRC precedent. |
| 11 | `hong2026recalmatch` | Verified and upgraded | arXiv `2609.29678` states acceptance at BMVC 2026. The entry now uses BMVC as the venue and retains the arXiv identifier. |
| 12 | `guo2027dgcl` | Verified with year distinction | Neural Networks 205:109571. Crossref gives an issue date of January 2027, while the DOI prefix and online record contain 2026. Retaining 2027 is consistent with the published volume. |
| 13 | `gilal2025cyflod` | Verified and completed | CVPR Workshops 2025, pp. 2059-2069, DOI `10.1109/CVPRW67362.2025.00194`. |
| 14 | `sensoy2018evidential` | Replaced and verified | The master's thesis was removed as the representative evidential-learning citation and replaced with Sensoy, Kaplan, and Kandemir, NeurIPS 2018. |
| 15 | `wah2011cub` | Verified | CaltechAUTHORS record `cvm3y-5hh21` confirms the authors, title, technical-report type, and July 2011 date. |
| 16 | `maji2013aircraft` | Corrected | Authors are Subhransu Maji, Esa Rahtu, Juho Kannala, Matthew B. Blaschko, and Andrea Vedaldi. The previous Jonghyun Choi entry was wrong. |
| 17 | `nilsback2008flowers` | Verified and completed | ICVGIP 2008, pp. 722-729, DOI `10.1109/ICVGIP.2008.47`. |
| 18 | `angelopoulos2025ltt` | Corrected | The citable journal version is Annals of Applied Statistics 19(2):1641-1662, 2025, DOI `10.1214/24-AOAS1998`. |
| 19 | `angelopoulos2024conformal` | Corrected | OpenReview identifies the publication as ICLR 2024; the previous ICLR 2023 year was wrong. |
| 20 | `zhang2018gce` | Verified | NeurIPS 2018, volume 31; title, authors, and method identity match the official proceedings. |
| 21 | `khosla2011dogs` | Verified for use; workshop metadata caveat | The dataset is consistently identified as Khosla et al., FGVC Workshop at CVPR 2011, including the authors' frozen dataset provenance. The original workshop page was not reopened during this audit. |

## Relevance judgment

The direct argumentative citations are appropriate:

- selective classification and its gap: [4], [5];
- post-hoc selective calibration: [7], [8];
- confidence and feature-geometry baselines: [6], [7], [9];
- noisy-label training: [1], [3], [11], [12], [13];
- prototypes: [10];
- web-supervised fine-grained data: [2];
- evidential uncertainty background: [14];
- risk control: [18], [19].

The datasets [15], [16], [17], and [21] are cited for data provenance rather
than as competitors. Their role is appropriate.

## Remaining caveats

- DGCL should not be described as “published 2026”; the volume is dated 2027.
  The DOI-prefix year alone is not a publication year.
- The Stanford Dogs workshop citation remains a non-DOI legacy record. Its use
  is supported, but a target journal may require the workshop proceedings URL
  if one becomes available.
- The main manuscript no longer relies on a master's thesis to represent the
  evidential-learning literature.
