from langchain_core.documents import Document
import pdfplumber


def load_pdf(file):

    documents = []

    with pdfplumber.open(file) as pdf:

        for page_num, page in enumerate(pdf.pages):

            text = page.extract_text()

            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": file.name,
                            "page": page_num + 1
                        }
                    )
                )

    return documents