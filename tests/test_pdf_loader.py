from pathlib import Path

from RAG_PROJECT.document_loader.pdf import extract_pdf_text


def test_extract_pdf_text_returns_resume_content():
    pdf_path = Path(__file__).resolve().parents[1] / "RAG_PROJECT" / "document_loader" / "PRASHANTH A_1BY22EC073_2.pdf"
    text = extract_pdf_text(str(pdf_path))

    assert "PRASHANTH A" in text
    assert "EXPERIENCE" in text
