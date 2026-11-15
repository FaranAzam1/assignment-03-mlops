"""Build report/report.md (evidence embedded) and report/report.pdf from report/report_template.md.

    python scripts/build_report.py

`{{evidence:file|max=N}}` is replaced by the file's text in a code block (first N lines). The PDF is
printed from HTML by a locally installed Edge or Chrome in headless mode (no extra Python dependency
besides `markdown`, installed with `pip install -e ".[report]"`).
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import markdown

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "report"
EVID = REPORT / "evidence"
BROWSERS = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "/usr/bin/google-chrome",
    "/usr/bin/chromium",
]
CSS = """
body{font:13px/1.5 'Segoe UI',Arial,sans-serif;max-width:900px;margin:24px auto;color:#1f2328}
h1{font-size:24px;border-bottom:2px solid #1f3a5f;padding-bottom:6px;color:#1f3a5f}
h2{font-size:18px;color:#1f3a5f;margin-top:28px;border-bottom:1px solid #d0d7de}
h3{font-size:15px;color:#1f3a5f}
pre{background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:10px;white-space:pre-wrap;
word-break:break-word;font:11px/1.4 Consolas,monospace}
code{font:11.5px Consolas,monospace;background:#f6f8fa;padding:1px 4px;border-radius:4px}
pre code{background:none;padding:0}
table{border-collapse:collapse;width:100%;margin:10px 0;font-size:12px}
th,td{border:1px solid #d0d7de;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#eef2f7}
img{max-width:100%;border:1px solid #d0d7de;margin:6px 0}
"""


def embed(match: re.Match) -> str:
    name, _, opt = match.group(1).partition("|max=")
    lines = (EVID / name).read_text(encoding="utf-8").rstrip().splitlines()
    if opt and len(lines) > int(opt):
        lines = lines[: int(opt)] + [f"... ({len(lines) - int(opt)} more lines in report/evidence/{name})"]
    return "```text\n" + "\n".join(lines) + "\n```"


def main() -> None:
    text = (REPORT / "report_template.md").read_text(encoding="utf-8")
    text = re.sub(r"\{\{evidence:([^}]+)\}\}", embed, text)
    (REPORT / "report.md").write_text(text, encoding="utf-8")
    body = markdown.markdown(text, extensions=["tables", "fenced_code"])
    html_path = REPORT / "report.html"
    html_path.write_text(f"<!doctype html><meta charset=utf-8><style>{CSS}</style>{body}", encoding="utf-8")
    browser = next((b for b in BROWSERS if Path(b).exists()), None)
    if browser is None:
        sys.exit("no Edge/Chrome found; open report/report.html and print to PDF manually")
    pdf = REPORT / "report.pdf"
    pdf.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as profile:  # fresh profile: works while another Edge/Chrome is open
        subprocess.run(
            [browser, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", f"--user-data-dir={profile}",
             f"--print-to-pdf={pdf}", html_path.as_uri()],
            check=True, capture_output=True, timeout=180,
        )
    for _ in range(60):  # the browser may flush the file just after it returns
        if pdf.exists() and pdf.stat().st_size > 0:
            break
        time.sleep(0.5)
    html_path.unlink()
    if not pdf.exists():
        sys.exit("browser exited without writing report.pdf")
    print("wrote report/report.md and report/report.pdf")


if __name__ == "__main__":
    main()
