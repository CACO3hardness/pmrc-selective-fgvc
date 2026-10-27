from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT.parent.parent / "outputs"
FIGURE_ROOT = OUTPUT_ROOT / "figures"
HTML_PATH = FIGURE_ROOT / "figureS2_method_overview.html"

PAPER = "#f7f7f5"
INK = "#202124"
MUTED = "#5f6368"
SOFT = "#777b80"
ACCENT = "#b33b1e"
ACCENT_TINT = "rgba(179,59,30,0.075)"
WHITE = "#ffffff"
SANS = "Arial, Helvetica, sans-serif"
MONO = "Consolas, 'Courier New', monospace"


def node(
    x: int,
    y: int,
    width: int,
    height: int,
    tag: str,
    lines: tuple[str, ...],
    sublabel: str,
    focal: bool = False,
) -> str:
    center_x = x + width / 2
    fill = ACCENT_TINT if focal else WHITE
    stroke = ACCENT if focal else INK
    tag_stroke = "rgba(179,59,30,0.55)" if focal else "rgba(32,33,36,0.42)"
    tag_fill = ACCENT if focal else MUTED
    title_start = y + (44 if len(lines) == 1 else 36)
    title_markup = "".join(
        f'<text x="{center_x:.0f}" y="{title_start + index * 18}" '
        f'fill="{INK}" font-size="18" font-weight="600" '
        f'font-family="{SANS}" text-anchor="middle">{line}</text>'
        for index, line in enumerate(lines)
    )
    sub_y = title_start + (len(lines) - 1) * 18 + 20
    return f"""
    <g>
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6" fill="{PAPER}"/>
      <rect x="{x}" y="{y}" width="{width}" height="{height}" rx="6"
            fill="{fill}" stroke="{stroke}" stroke-width="1"/>
      <rect x="{x + 8}" y="{y + 8}" width="48" height="12" rx="2"
            fill="none" stroke="{tag_stroke}" stroke-width="0.8"/>
      <text x="{x + 32}" y="{y + 17}" fill="{tag_fill}" font-size="8"
            font-family="{MONO}" text-anchor="middle"
            letter-spacing="0.08em">{tag}</text>
      {title_markup}
      <text x="{center_x:.0f}" y="{sub_y}" fill="{MUTED}" font-size="12"
            font-family="{MONO}" text-anchor="middle">{sublabel}</text>
    </g>"""


def label(x: int, y: int, width: int, text: str, color: str = MUTED) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{width}" height="12" rx="2" fill="{PAPER}"/>'
        f'<text x="{x + width / 2:.0f}" y="{y + 10}" fill="{color}" '
        f'font-size="8" font-family="{MONO}" text-anchor="middle" '
        f'letter-spacing="0.06em">{text}</text>'
    )


def build_html() -> str:
    nodes = "".join(
        [
            node(24, 72, 168, 88, "INPUT", ("Noisy data",), "images + noisy labels"),
            node(232, 72, 168, 88, "MODEL", ("Frozen model",), "features + logits"),
            node(440, 72, 176, 88, "ARTIFACT", ("Prototypes",), "observed-label means"),
            node(24, 240, 168, 104, "INPUT", ("Clean", "calibration"), "images + clean labels"),
            node(
                232,
                240,
                168,
                104,
                "MODEL",
                ("Reliability", "scorer"),
                "confidence + margin",
            ),
            node(
                440,
                240,
                176,
                104,
                "GATE",
                ("Gate + ordering",),
                "PMRC or MSP; argmax fixed",
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
    body {{ background: {PAPER}; color: {INK}; font-family: {SANS}; padding: 24px; }}
    .frame {{ width: 100%; max-width: 640px; margin: 0 auto; }}
    .diagram-container {{ width: 100%; overflow-x: auto; }}
    svg {{ width: 100%; min-width: 640px; display: block; }}
    @page {{ size: 5in 3.75in; margin: 0; }}
    @media print {{
      body {{ padding: 0; }}
      .diagram-container {{ overflow-x: visible; }}
      svg {{ min-width: 0; }}
    }}
  </style>
</head>
<body>
  <div class="frame">
    <div class="diagram-container">
      <svg viewBox="0 0 640 480" xmlns="http://www.w3.org/2000/svg"
           role="img" aria-labelledby="pmrc-method-title pmrc-method-desc">
        <title id="pmrc-method-title">PMRC method overview</title>
        <desc id="pmrc-method-desc">PMRC builds observed-label prototypes from a frozen classifier, combines confidence and prototype margin into a sample-level reliability score fitted on clean calibration data, and uses a validation-only gate to choose PMRC or MSP ordering while leaving predictions unchanged.</desc>
        <defs>
          <marker id="arrow" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{MUTED}"/>
          </marker>
          <marker id="arrow-accent" markerWidth="8" markerHeight="6" refX="7" refY="3" orient="auto">
            <polygon points="0 0, 8 3, 0 6" fill="{ACCENT}"/>
          </marker>
        </defs>

        <rect width="640" height="480" fill="{PAPER}"/>

        <!-- Connectors are drawn before boxes. -->
        <path d="M 108,72 V 40 H 528 V 72" fill="none" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="192" y1="116" x2="232" y2="116" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="400" y1="116" x2="440" y2="116" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="316" y1="160" x2="316" y2="240" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <path d="M 528,160 V 204 Q 528,212 520,212 H 388 Q 380,212 380,220 V 240"
              fill="none" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="192" y1="292" x2="232" y2="292" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <path d="M 108,344 V 384 H 528 V 344"
              fill="none" stroke="{MUTED}" stroke-width="1.2" marker-end="url(#arrow)"/>
        <line x1="400" y1="292" x2="440" y2="292" stroke="{ACCENT}" stroke-width="1.4" marker-end="url(#arrow-accent)"/>

        {label(292, 20, 56, "LABELS")}
        {label(194, 94, 36, "TRAIN")}
        {label(406, 94, 28, "DATA")}
        {label(326, 184, 48, "LOGITS")}
        {label(446, 184, 48, "MARGIN")}
        {label(196, 270, 32, "FIT")}
        {label(276, 366, 56, "CALIB")}

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
