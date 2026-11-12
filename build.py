"""Assemble src/ en un seul docs/index.html autonome (CSS, JS et modèle intégrés).

Aucun fichier externe à charger : la page s'affiche correctement quelle que soit
l'URL (avec ou sans « / » final), sur GitHub Pages, Vercel ou en local (file://).
"""
from pathlib import Path

ROOT = Path(__file__).parent
SRC, OUT = ROOT / "src", ROOT / "docs" / "index.html"

html = (SRC / "index.html").read_text(encoding="utf-8")
css = (SRC / "style.css").read_text(encoding="utf-8")
js = (SRC / "app.js").read_text(encoding="utf-8")
model = (SRC / "model.json").read_text(encoding="utf-8")

html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>")
html = html.replace('<script src="app.js"></script>',
                    f"<script>window.MODEL = {model};</script>\n  <script>\n{js}</script>")
assert "style.css" not in html and 'src="app.js"' not in html

OUT.parent.mkdir(exist_ok=True)
OUT.write_text(html, encoding="utf-8")
print(f"Construit -> {OUT} ({len(html) // 1024} Ko)")
