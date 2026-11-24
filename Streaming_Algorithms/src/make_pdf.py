"""reports/report.md → HTML → PDF (Chrome/Chromium headless print)."""
import shutil
import subprocess
from pathlib import Path

import markdown

from .data import ROOT

REPORT_MD = ROOT / "reports" / "report.md"
REPORT_HTML = ROOT / "reports" / "report.html"
REPORT_PDF = ROOT / "reports" / "report.pdf"

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "google-chrome", "chromium", "chromium-browser", "chrome",
]

CSS = """
@page { size: A4; margin: 16mm 14mm; }
body { font-family: 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', sans-serif;
       font-size: 10.5pt; line-height: 1.55; color: #1a1a1a; }
h1 { font-size: 20pt; border-bottom: 3px solid #333; padding-bottom: 6px; }
h2 { font-size: 15pt; border-bottom: 1px solid #999; padding-bottom: 3px; margin-top: 26px;
     page-break-before: always; }
h2:first-of-type { page-break-before: avoid; }
h3 { font-size: 12pt; margin-top: 18px; }
table { border-collapse: collapse; margin: 8px 0 12px; font-size: 8.8pt; width: 100%; }
th, td { border: 1px solid #bbb; padding: 3px 6px; text-align: center; }
th { background: #eef1f5; }
td:first-child { text-align: left; }
img { max-width: 100%; max-height: 105mm; width: auto; display: block; margin: 8px auto; page-break-inside: avoid; }
code { background: #f3f3f3; padding: 1px 4px; border-radius: 3px; font-size: 9pt; }
pre { background: #f6f8fa; padding: 8px; border-radius: 4px; font-size: 8.5pt; }
blockquote { border-left: 4px solid #4c72b0; margin: 8px 0; padding: 4px 12px; background: #f5f8fc; }
.best { font-weight: bold; }
.qa { border: 2px solid #2a6fdb; border-left-width: 6px; border-radius: 6px; background: #f3f7fe;
      padding: 4px 14px 8px; margin: 14px 0; page-break-inside: avoid; }
.qa h3 { color: #1d4fa3; margin-top: 8px; }
.qa table { background: #fff; }
"""


def _chrome() -> str:
    for c in CHROME_CANDIDATES:
        if Path(c).exists() or shutil.which(c):
            return c
    raise RuntimeError("Chrome/Chromium을 찾을 수 없습니다. reports/report.html을 브라우저에서 PDF로 인쇄하세요.")


def run() -> None:
    body = markdown.markdown(REPORT_MD.read_text(encoding="utf-8"),
                             extensions=["tables", "fenced_code", "sane_lists", "md_in_html"])
    html = f"<!doctype html><html lang='ko'><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"
    REPORT_HTML.write_text(html, encoding="utf-8")
    subprocess.run([
        _chrome(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
        f"--print-to-pdf={REPORT_PDF}", REPORT_HTML.as_uri(),
    ], check=True, capture_output=True)
    print("PDF 생성:", REPORT_PDF)


if __name__ == "__main__":
    run()
