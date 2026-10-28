from pathlib import Path

from docling.document_converter import DocumentConverter


SUPPORTED_EXTENSIONS = {
    ".docx",
    ".html",
    ".md",
    ".pdf",
    ".pptx",
    ".txt",
    ".xlsx",
}


def parse_document(
    source: Path, converter: DocumentConverter | None = None
) -> str:
    """Convert a supported local file to Docling's Markdown representation."""
    active_converter = converter or DocumentConverter()
    result = active_converter.convert(str(source))
    markdown = result.document.export_to_markdown().strip()
    if not markdown:
        raise ValueError(f"Docling extracted no text from {source}")
    return markdown