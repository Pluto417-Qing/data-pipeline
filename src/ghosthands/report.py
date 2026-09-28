from __future__ import annotations

import html
import json
from pathlib import Path


def build_report(dataset_root: Path) -> Path:
    cards: list[str] = []
    for quality_path in sorted((dataset_root / "samples").glob("*/quality.json")):
        sample_dir = quality_path.parent
        quality = json.loads(quality_path.read_text(encoding="utf-8"))
        metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
        title = html.escape(sample_dir.name)
        flags = ", ".join(quality["flags"]) or "none"
        cards.append(f'''<article class="{quality["status"]}"><h2>{title}</h2><p><b>{quality["status"]}</b> · {quality["frame_count"]} frames · hand coverage {quality["hand_frame_ratio"]:.1%}</p><p>backend: {html.escape(metadata["backend"])}<br>flags: {html.escape(flags)}</p><div><video controls preload="metadata" src="samples/{title}/source.mp4"></video><video controls preload="metadata" src="samples/{title}/target.mp4"></video></div></article>''')
    page = f'''<!doctype html><html><head><meta charset="utf-8"><title>GhostHands quality report</title><style>body{{font:14px system-ui;background:#111;color:#eee;margin:24px}}article{{padding:16px;margin:16px 0;border-left:6px solid #42b7ff;background:#1b1b1b}}article.needs_review{{border-color:#ffab3d}}h2{{margin-top:0}}video{{width:min(48%,560px);margin-right:1%;background:#000}}</style></head><body><h1>GhostHands quality report</h1>{''.join(cards) or '<p>No processed samples found.</p>'}</body></html>'''
    report_path = dataset_root / "report.html"
    report_path.write_text(page, encoding="utf-8")
    return report_path
