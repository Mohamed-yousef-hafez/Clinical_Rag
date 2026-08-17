"""
PDF Loader

Loads one or multiple PDF documents
while preserving page numbers and
document names.
"""

import fitz
from pathlib import Path

from src.config.settings import settings


class PDFLoader:

    def __init__(self, pdf_path=None):

        self.pdf_path = pdf_path

    def load(self):

        pages = []

        # -----------------------------------------
        # Single PDF
        # -----------------------------------------

        if self.pdf_path:

            pdf_files = [Path(self.pdf_path)]

        # -----------------------------------------
        # Load all PDFs inside data/docs
        # -----------------------------------------

        else:

            pdf_files = sorted(
                Path(settings.DATA_FOLDER).glob("*.pdf")
            )

        if not pdf_files:

            raise FileNotFoundError(
                "No PDF files found in data/docs"
            )

        # -----------------------------------------
        # Read every PDF
        # -----------------------------------------

        for pdf_file in pdf_files:

            document = fitz.open(str(pdf_file))

            document_name = pdf_file.stem

            for page_number, page in enumerate(
                document,
                start=1
            ):

                text = page.get_text("text")

                pages.append(
                    {
                        "document": document_name,
                        "page": page_number,
                        "text": text
                    }
                )

            document.close()

        return pages