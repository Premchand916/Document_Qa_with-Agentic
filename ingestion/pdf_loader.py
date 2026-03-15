import pdfplumber


def load_pdf(file):
    """Load text from a PDF file.

    Supports file-like objects (e.g., Streamlit `UploadedFile`) and file paths.
    """

    # pdfplumber supports file-like objects, including BytesIO from Streamlit.
    with pdfplumber.open(file) as pdf:
        pages = [page.extract_text() or "" for page in pdf.pages]

    return "\n\n".join(pages)
