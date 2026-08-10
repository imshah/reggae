"""Parse pdf/docx/txt/md into text blocks and extracted diagram images.

Each parser returns:
  - blocks:  list[TextBlock]   (text with a section heading + page)
  - images:  list[ImageRef]    (embedded raster images = candidate diagrams)

Images are written under data/images/<doc_id>/ and referenced by path so the
vision step and later cleanup can find them.
"""
from __future__ import annotations

import zipfile
from dataclasses import dataclass, field
from pathlib import Path

from docmind.config import IMAGES_DIR, ensure_dirs

SUPPORTED = {".pdf", ".docx", ".txt", ".md", ".markdown"}


@dataclass
class TextBlock:
    text: str
    section: str
    page: int


@dataclass
class ImageRef:
    path: str
    page: int
    section: str


@dataclass
class ParsedDoc:
    title: str
    blocks: list[TextBlock] = field(default_factory=list)
    images: list[ImageRef] = field(default_factory=list)


def _img_dir(doc_id: str) -> Path:
    ensure_dirs()
    d = IMAGES_DIR / doc_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def parse(path: Path, doc_id: str) -> ParsedDoc:
    ext = path.suffix.lower()
    if ext == ".pdf":
        return _parse_pdf(path, doc_id)
    if ext == ".docx":
        return _parse_docx(path, doc_id)
    if ext in (".txt", ".md", ".markdown"):
        return _parse_text(path)
    raise ValueError(f"Unsupported file type: {ext}")


# --- PDF -------------------------------------------------------------------


def _parse_pdf(path: Path, doc_id: str) -> ParsedDoc:
    import pymupdf as fitz  # PyMuPDF (modern import; `import fitz` is deprecated)

    doc = ParsedDoc(title=path.stem)
    img_dir = _img_dir(doc_id)
    with fitz.open(path) as pdf:
        for pno, page in enumerate(pdf, start=1):
            text = page.get_text("text").strip()
            if text:
                doc.blocks.append(TextBlock(text=text, section=f"Page {pno}", page=pno))
            for i, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(pdf, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK etc → RGB
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    if pix.width < 64 or pix.height < 64:
                        continue  # skip icons/bullets
                    out = img_dir / f"p{pno}_{i}.png"
                    pix.save(str(out))
                    doc.images.append(ImageRef(path=str(out), page=pno, section=f"Page {pno}"))
                except Exception:
                    continue
    return doc


# --- DOCX ------------------------------------------------------------------


def _parse_docx(path: Path, doc_id: str) -> ParsedDoc:
    import docx  # python-docx

    doc = ParsedDoc(title=path.stem)
    d = docx.Document(str(path))

    current_section = "Introduction"
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                doc.blocks.append(TextBlock(text=joined, section=current_section, page=0))
            buffer.clear()

    for para in d.paragraphs:
        style = (para.style.name or "").lower() if para.style else ""
        txt = para.text.strip()
        if not txt:
            continue
        if style.startswith("heading") or style == "title":
            flush()
            current_section = txt
        else:
            buffer.append(txt)
    flush()

    # extract embedded images from the docx zip
    img_dir = _img_dir(doc_id)
    try:
        with zipfile.ZipFile(path) as z:
            for name in z.namelist():
                if name.startswith("word/media/") and name.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".gif", ".bmp", ".emf", ".wmf")
                ):
                    data = z.read(name)
                    out = img_dir / Path(name).name
                    out.write_bytes(data)
                    if out.suffix.lower() in (".emf", ".wmf"):
                        continue  # vision models can't read vector metafiles
                    doc.images.append(ImageRef(path=str(out), page=0, section="Document"))
    except zipfile.BadZipFile:
        pass
    return doc


# --- plain text / markdown -------------------------------------------------


def _parse_text(path: Path) -> ParsedDoc:
    doc = ParsedDoc(title=path.stem)
    text = path.read_text(errors="replace")
    is_md = path.suffix.lower() in (".md", ".markdown")

    if not is_md:
        doc.blocks.append(TextBlock(text=text.strip(), section=path.stem, page=0))
        return doc

    # Markdown: split on ATX headings, keep each section together
    current_section = path.stem
    buffer: list[str] = []

    def flush() -> None:
        if buffer:
            joined = "\n".join(buffer).strip()
            if joined:
                doc.blocks.append(TextBlock(text=joined, section=current_section, page=0))
            buffer.clear()

    for line in text.splitlines():
        if line.lstrip().startswith("#"):
            flush()
            current_section = line.lstrip("#").strip() or current_section
        else:
            buffer.append(line)
    flush()
    return doc
