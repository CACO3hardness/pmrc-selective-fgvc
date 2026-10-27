from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
HTML_PATH = FIGURE_ROOT / "figureS2_method_overview.html"

PAPER = "#f7f7f5"
INK = "#202124"
MUTED = "#5f6368"
SOFT = "#73777d"
RULE = "rgba(32,33,36,0.14)"
ACCENT = "#b33b1e"
ACCENT_TINT = "rgba(179,59,30,0.075)"
WHITE = "#ffffff"

SANS = "Arial, Helvetica, sans-serif"
MONO = "Consolas, 'Courier New', monospace"


def box(
    x: int,
    y: int,
    width: int,
    height: int,
    tag: str,
    title: str,
    sublabel: str,
    focal: bool = False,
) -> str:
    fill = ACCENT_TINT if focal else WHITE
    stroke = ACCENT if focal else INK
    tag_stroke = "rgba(179,59,30,0.55)" if focal else "rgba(32,33,36,0.42)"
    tag_fill = ACCENT if focal else MUTED
    cx = x + width / 2
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{PAPER}"/>
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6"
            fill="{fill}" stroke="{stroke}" stroke-width="1"/>
      <rect x="{x + 8}" y="{y + 8}" width="48" height="12" rx="2"
            fill="none" stroke="{tag_stroke}" stroke-width="0.8"/>
      <text x="{x + 32}" y="{y + 17}" fill="{tag_fill}" font-size="7"
            font-family="{MONO}" text-anchor="middle"
            letter-spacing="0.08em">{tag}</text>
      <text x="{cx:.0f}" y="{y + 48}" fill="{INK}" font-size="12"
            font-weight="600" font-family="{SANS}" text-anchor="middle">{title}</text>
      <text x="{cx:.0f}" y="{y + 68}" fill="{MUTED}" font-size="9"
            font-family="{MONO}" text-anchor="middle">{sublabel}</text>
    </g>"""


def label_mask(
    x: int,
    y: int,
    width: int,
    text: str,
    color: str = MUTED,
) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="12" rx="2" fill="{PAPER}"/>'
        f'<text x="{x + width / 2:.0f}" y="{y + 10}" fill="{color}" font-size="8" '
        f'font-family="{MONO}" text-anchor="middle" letter-spacing="0.06em">{text}</text>'
    )


def build_html() -> str:
    nodes = "".join(
        [
            box(80, 120, 240, 96, "INPUT", "Noisy training data", "images + observed labels"),
            box(392, 120, 240, 96, "MODEL", "Frozen classifier", "features + logits"),
            box(704, 120, 240, 96, "ARTIFACT", "Class prototypes", "means under observed labels"),
            box(80, 344, 240, 96, "INPUT", "Clean calibration data", "images + clean labels"),
            box(392, 344, 240, 96, "FEATURES", "Reliability vector", "confidence + views + margin"),
            box(704, 344, 240, 96, "MODEL", "Correctness ranker", "L2 logistic regression"),
            box(
                1008,
                344,
                200,
                96,
                "GATE",
                "Validation-only gate",
                "clean labels + prototype AUROC",
                focal=True,
            ),
            box(
                704,
                576,
                504,
                96,
                "OUTPUT",
                "Selective acceptance",
                "PMRC or MSP order; argmax predictions fixed",
                focal=True,
            ),
        ]
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>PMRC method overview</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: {PAPER};
      color: {INK};
      font-family: {SANS};
      min-height: 100vh;
      padding: 24px;
    }}
    .frame {{ width: 100%; max-width: 1280px; margin: 0 auto; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    svg {{ width: 100%; min-width: 1280px; display: block; }}
    @page {{ size: 13.333in 7.5in; margin: 0; }}
    @media print {{
      body {{ padding: 0; }}
      .diagram-container {{ overflow-x: visible; }}
      svg {{ width: 100%; min-width: 0; }}
    }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="diagram-container">
      <svg viewBox="0 0 1280 720" xmlns="http://www.w3.org/2000/svg"
           role="img" aria-labelledby="pmrc-method-title pmrc-method-desc">
        <title id="pmrc-method-title">PMRC method overview</title>
        <desc id="pmrc-method-desc">PMRC constructs prototypes from observed labels, builds a sample-level reliability vector from a frozen classifier, fits a clean-calibration correctness ranker, and uses a validation-only gate to choose PMRC or MSP ordering.</desc>
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/>
          </marker>
          <marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/>
          </marker>
        </defs>

        <rect width="1280" height="720" fill="{PAPER}"/>

        <!-- Connectors are drawn before nodes. -->
        <line x1="320" y1="168" x2="392" y2="168" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="632" y1="168" x2="704" y2="168" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <path d="M 200,120 V 88 H 824 V 120" fill="none" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="200" y1="216" x2="200" y2="344" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="512" y1="216" x2="512" y2="344" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <path d="M 824,216 V 296 Q 824,304 816,304 H 568 Q 560,304 560,312 V 344"
              fill="none" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="320" y1="392" x2="392" y2="392" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="632" y1="392" x2="704" y2="392" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <path d="M 160,440 V 488 H 752 Q 760,488 760,480 V 440"
              fill="none" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="820" y1="440" x2="820" y2="576" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="1108" y1="440" x2="1108" y2="576" stroke="{ACCENT}" stroke-width="1.4" marker-end="url(#arrow-accent)"/>

        <!-- Connector labels. Every mask leaves a visible gap from its line. -->
        {label_mask(332, 146, 48, "TRAIN")}
        {label_mask(636, 146, 64, "FEATURES")}
        {label_mask(476, 70, 56, "LABELS")}
        {label_mask(210, 268, 64, "OBS LABEL")}
        {label_mask(522, 268, 48, "LOGITS")}
        {label_mask(766, 268, 48, "MARGIN")}
        {label_mask(328, 374, 48, "CALIB")}
        {label_mask(636, 374, 64, "EVIDENCE")}
        {label_mask(420, 470, 32, "FIT")}
        {label_mask(832, 536, 40, "ORDER")}
        {label_mask(1118, 500, 64, "FALLBACK", ACCENT)}

        {nodes}
      </svg>
    </div>
  </div>
</body>
</html>
"""


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    HTML_PATH.write_text(build_html(), encoding="utf-8")
    print(HTML_PATH)


if __name__ == "__main__":
    main()
