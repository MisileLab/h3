from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import httpx
from bs4 import BeautifulSoup

# Allow importing from project src
import sys
sys.path.append(str(Path(__file__).parent.parent))
from src.app import PDFProcessor


BASE_URL: str = "https://assembly.dongjak.go.kr"
LIST_PATH: str = "/kr/costBBS.do"
DOWNLOAD_DIR: Path = Path("downloads") / "cost_bbs"


@dataclass(frozen=True)
class Post:
  title: str
  file_url: str | None


def build_absolute_url(path_or_url: str | None) -> str | None:
  if not path_or_url:
    return None
  if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
    return path_or_url
  if not path_or_url.startswith("/"):
    path_or_url = "/" + path_or_url
  return f"{BASE_URL}{path_or_url}"


def parse_posts(html: str) -> list[Post]:
  soup = BeautifulSoup(html, "lxml")

  posts: list[Post] = []

  rows = soup.select("table tbody tr")
  for row in rows:
    title_el = row.select_one("td.subject a, td.tit a, td a")
    title_text = title_el.get_text(strip=True) if title_el else None
    file_el = row.select_one('a[href*="download"], a[href*="file"], a[href*="atchFile"], a[href*="atchfile"], a[href*="atchFileId"], a[download]')
    if not file_el:
      file_el = title_el
    if not title_text:
      alt_title_el = row.select_one("td.subject, td.tit, td")
      title_text = alt_title_el.get_text(strip=True) if alt_title_el else None
    if not title_text:
      continue
    file_href = str(file_el.get("href")) if file_el and file_el.get("href") else None
    posts.append(Post(title=title_text, file_url=build_absolute_url(file_href)))

  if not posts:
    items = soup.select(".board_list li, ul.list li, .list li")
    for item in items:
      title_el = item.select_one("a")
      title_text = title_el.get_text(strip=True) if title_el else None
      file_el = item.select_one('a[href*="download"], a[download]') or title_el
      if not title_text:
        continue
      file_href = str(file_el.get("href")) if file_el and file_el.get("href") else None
      posts.append(Post(title=title_text, file_url=build_absolute_url(file_href)))

  return posts


def fetch_listing() -> str:
  url = f"{BASE_URL}{LIST_PATH}"
  with httpx.Client(timeout=30) as client:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.text


def to_json_records(posts: Iterable[Post]) -> str:
  records = [
    {"title": p.title, "file": p.file_url}
    for p in posts
  ]
  return json.dumps(records, ensure_ascii=False, indent=2)


def sanitize_filename(name: str) -> str:
  name = name.strip().replace(" ", "_")
  name = re.sub(r"[^A-Za-z0-9._-]", "", name)
  return name[:200] if len(name) > 200 else name


def derive_filename(url: str, title: str, headers: Mapping[str, str]) -> str:
  cd = headers.get("content-disposition") or headers.get("Content-Disposition")
  if cd and "filename=" in cd:
    fname = cd.split("filename=")[-1].strip('"')
    return sanitize_filename(fname)
  from urllib.parse import urlparse
  path = urlparse(url).path
  last = Path(path).name or f"{sanitize_filename(title)}.pdf"
  return sanitize_filename(last)


def is_pdf_filename(filename: str) -> bool:
  return filename.lower().endswith(".pdf")


def download_file(url: str, title: str) -> Path | None:
  DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
  with httpx.Client(timeout=60, follow_redirects=True) as client:
    resp = client.get(url)
    resp.raise_for_status()
    filename = derive_filename(str(resp.url), title, resp.headers)  # after redirects
    if not is_pdf_filename(filename):
      return None
    target = DOWNLOAD_DIR / filename
    with open(target, "wb") as f:
      f.write(resp.content)
    return target


def extract_pdf(pdf_path: Path, use_ocr: bool, fmt: str, output_root: Path | None) -> None:
  processor = PDFProcessor(use_ocr=use_ocr)
  output_dir = str((output_root or pdf_path.parent) / f"{pdf_path.stem}_extract")
  output_cfg = {
    "format": fmt,
    "output_dir": output_dir,
    "include_tables": False,
    "show_preview": False,
  }
  processor.process_pdf(str(pdf_path), output_cfg)


def main() -> None:
  parser = argparse.ArgumentParser(description="Scrape costBBS, download PDFs, and extract text")
  parser.add_argument("--ocr", action="store_true", help="Use OCR extraction instead of direct text")
  parser.add_argument("--format", "-f", default="txt", choices=["txt", "json", "csv", "all"], help="Output format")
  parser.add_argument("--limit", type=int, default=0, help="Limit number of files to download/process (0 = no limit)")
  parser.add_argument("--skip-existing", action="store_true", help="Skip files that already exist in downloads")
  parser.add_argument("--output-root", type=Path, default=None, help="Root directory for extracted outputs (defaults next to downloads)")
  args = parser.parse_args()

  html = fetch_listing()
  posts = parse_posts(html)

  processed = 0
  for post in posts:
    if args.limit and processed >= args.limit:
      break
    if not post.file_url:
      continue
    try:
      # Probe filename first via HEAD to decide PDF or not
      with httpx.Client(timeout=30, follow_redirects=True) as client:
        head = client.head(post.file_url)
        fname_probe = derive_filename(str(head.url), post.title, head.headers)
      if not is_pdf_filename(fname_probe):
        continue

      target = DOWNLOAD_DIR / fname_probe
      if args.skip_existing and target.exists():
        extract_pdf(target, args.ocr, args.format, args.output_root)
        processed += 1
        continue

      file_path = download_file(post.file_url, post.title)
      if not file_path:
        continue
      extract_pdf(file_path, args.ocr, args.format, args.output_root)
      processed += 1
    except Exception as e:
      # Fail fast for each item but continue overall
      print(f"Skip '{post.title}': {e}")

  # Also print JSON summary of scraped posts
  print(to_json_records(posts))


if __name__ == "__main__":
  main()


