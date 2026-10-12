from docx import Document

from ink_ai.documents import read_document


def test_reads_utf8_text_document(tmp_path):
    file_path = tmp_path / "örnek.py"
    file_path.write_text("print('merhaba')\n", encoding="utf-8")

    assert read_document(file_path) == "print('merhaba')\n"


def test_rejects_document_text_that_would_exceed_model_context(tmp_path):
    file_path = tmp_path / "large.txt"
    file_path.write_text("x" * 6_001, encoding="utf-8")

    try:
        read_document(file_path)
    except ValueError as error:
        assert "6.000 karakter" in str(error)
    else:
        raise AssertionError("Oversized documents must be rejected, not truncated.")


def test_rejects_unsupported_file_type(tmp_path):
    file_path = tmp_path / "image.png"
    file_path.write_bytes(b"not an image")

    try:
        read_document(file_path)
    except ValueError as error:
        assert "Desteklenmeyen" in str(error)
    else:
        raise AssertionError("Unsupported files must be rejected.")


def test_reads_word_document(tmp_path):
    file_path = tmp_path / "brief.docx"
    document = Document()
    document.add_paragraph("Proje gereksinimleri")
    document.save(file_path)

    assert read_document(file_path) == "Proje gereksinimleri"
