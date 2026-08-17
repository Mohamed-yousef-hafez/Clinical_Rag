"""
Text Cleaner

Cleans extracted PDF text while
preserving page information.
"""

import re


class TextCleaner:

    def clean(self, pages):

        cleaned_pages = []

        for page in pages:

            text = page.get(
                "text",
                ""
            )

            # -----------------------------------------
            # Normalize whitespace
            # -----------------------------------------

            text = re.sub(
                r"\s+",
                " ",
                text
            )

            text = text.strip()

            # -----------------------------------------
            # Keep metadata
            # -----------------------------------------

            cleaned_pages.append(
                {
                    "document": page.get(
                        "document",
                        "Clinical Guideline"
                    ),
                    "page": page.get(
                        "page",
                        "Unknown"
                    ),
                    "text": text
                }
            )

        return cleaned_pages