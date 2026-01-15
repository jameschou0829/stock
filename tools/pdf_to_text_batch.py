"""
tools/pdf_to_text_batch.py

把 docs/pdfs/*.pdf 批次轉成文字：
- 若 pdftotext 抽得到字（chars >= threshold），用 pdftotext 產出
- 否則走 OCR：pdftoppm 轉 PNG，再用 tesseract(chi_tra+eng) 產出

輸出：
- docs/pdfs_txt/            （pdftotext 版本）
- docs/pdfs_ocr_txt/        （OCR 版本）
- docs/pdfs_text_index.json （每份PDF的抽取方式/字數/頁數）
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
from dataclasses import dataclass, asdict
from typing import Optional

from pypdf import PdfReader


@dataclass
class PdfTextIndexRow:
    pdf: str
    pages: int
    method: str  # "pdftotext" or "ocr" or "skip"
    chars: int
    out_txt: Optional[str]
    note: Optional[str] = None


def _run(cmd: list[str]) -> None:
    subprocess.check_call(cmd)


def _run_capture(cmd: list[str]) -> str:
    p = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return p.stdout or ""


def _ensure_tool(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"missing tool: {name}")


def extract_with_pdftotext(pdf_path: str, out_txt: str) -> int:
    _ensure_tool("pdftotext")
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    _run(["pdftotext", "-layout", pdf_path, out_txt])
    with open(out_txt, "r", encoding="utf-8", errors="ignore") as f:
        return len(f.read())


def extract_with_ocr(pdf_path: str, out_txt: str, work_dir: str, lang: str = "chi_tra+eng") -> int:
    _ensure_tool("pdftoppm")
    _ensure_tool("tesseract")
    os.makedirs(os.path.dirname(out_txt), exist_ok=True)
    os.makedirs(work_dir, exist_ok=True)

    # 1) PDF -> PNG pages
    out_prefix = os.path.join(work_dir, "page")
    _run(["pdftoppm", "-png", "-r", "300", pdf_path, out_prefix])

    # 2) OCR each page
    imgs = sorted(glob.glob(os.path.join(work_dir, "page-*.png")))
    parts: list[str] = []
    for img in imgs:
        txt = _run_capture(["tesseract", img, "stdout", "-l", lang, "--psm", "6"])
        parts.append(f"\n\n===== {os.path.basename(img)} =====\n\n{txt}")

    all_txt = "".join(parts)
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(all_txt)
    return len(all_txt)


def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pdf_dir = os.path.join(root, "docs", "pdfs")
    out_pdftotext_dir = os.path.join(root, "docs", "pdfs_txt")
    out_ocr_dir = os.path.join(root, "docs", "pdfs_ocr_txt")
    index_path = os.path.join(root, "docs", "pdfs_text_index.json")
    work_root = os.path.join(root, "docs", "_pdf_ocr_work_all")

    os.makedirs(out_pdftotext_dir, exist_ok=True)
    os.makedirs(out_ocr_dir, exist_ok=True)
    os.makedirs(work_root, exist_ok=True)

    pdfs = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    rows: list[PdfTextIndexRow] = []

    # 若 pdftotext 抽出來字數太少，視為圖片型 PDF，改走 OCR
    threshold_chars = 200

    for pdf_path in pdfs:
        base = os.path.splitext(os.path.basename(pdf_path))[0]
        try:
            reader = PdfReader(pdf_path)
            pages = len(reader.pages)
        except Exception:
            pages = 0

        out_pdftotext = os.path.join(out_pdftotext_dir, base + ".txt")
        out_ocr = os.path.join(out_ocr_dir, base + ".txt")
        work_dir = os.path.join(work_root, base)

        # 先試 pdftotext
        chars = 0
        try:
            chars = extract_with_pdftotext(pdf_path, out_pdftotext)
        except Exception as e:
            rows.append(PdfTextIndexRow(pdf=os.path.basename(pdf_path), pages=pages, method="skip", chars=0, out_txt=None, note=f"pdftotext error: {e}"))
            continue

        if chars >= threshold_chars:
            rows.append(PdfTextIndexRow(pdf=os.path.basename(pdf_path), pages=pages, method="pdftotext", chars=chars, out_txt=os.path.relpath(out_pdftotext, root)))
            continue

        # 再走 OCR（覆蓋輸出）
        try:
            chars2 = extract_with_ocr(pdf_path, out_ocr, work_dir=work_dir, lang="chi_tra+eng")
            rows.append(PdfTextIndexRow(pdf=os.path.basename(pdf_path), pages=pages, method="ocr", chars=chars2, out_txt=os.path.relpath(out_ocr, root), note=f"pdftotext chars={chars}"))
        except Exception as e:
            rows.append(PdfTextIndexRow(pdf=os.path.basename(pdf_path), pages=pages, method="skip", chars=0, out_txt=None, note=f"ocr error: {e}"))

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)

    print(f"[OK] processed={len(rows)} index={index_path}")
    ocr_cnt = sum(1 for r in rows if r.method == "ocr")
    text_cnt = sum(1 for r in rows if r.method == "pdftotext")
    print(f"[OK] pdftotext={text_cnt} ocr={ocr_cnt} skip={len(rows)-ocr_cnt-text_cnt}")


if __name__ == "__main__":
    main()

