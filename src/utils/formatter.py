class ResultFormatter:

    def format(self, results):

        formatted = []

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, distance in zip(
            documents,
            metadatas,
            distances
        ):

            formatted.append(
                {
                    "text": doc,

                    "page": meta.get(
                        "page",
                        "Unknown"
                    ),

                    "document": meta.get(
                        "document",
                        "Clinical Guideline"
                    ),

                    "section": meta.get(
                        "section",
                        "Unknown"
                    ),

                    "distance": round(
                        distance,
                        3
                    )
                }
            )

        return formatted