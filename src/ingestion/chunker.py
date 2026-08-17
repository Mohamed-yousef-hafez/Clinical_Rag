from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.config.settings import settings


class TextChunker:

    def __init__(self):

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.CHUNK_SIZE,
            chunk_overlap=settings.CHUNK_OVERLAP,
            separators=[
                "\n\n",
                "\n",
                ". ",
                " ",
                ""
            ]
        )

    def chunk(self, pages):

        chunks = []

        for page in pages:

            text = page.get(
                "text",
                ""
            ).strip()

            if not text:
                continue

            page_chunks = self.splitter.split_text(
                text
            )

            for chunk in page_chunks:

                chunks.append(
                    {
                        "document": page.get(
                            "document",
                            "Clinical Guideline"
                        ),
                        "page": page.get(
                            "page",
                            "Unknown"
                        ),
                        "text": chunk
                    }
                )

        return chunks