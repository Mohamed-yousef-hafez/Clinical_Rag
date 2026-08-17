"""
Safe Refusal Messages
"""


class SafeRefusal:

    def respond(self, risk):

        messages = {

            "prompt_injection": (
                "❌ Request blocked.\n\n"
                "The system detected a prompt injection attempt."
            ),

            "out_of_scope": (
                "I couldn't find sufficient evidence "
                "inside the uploaded hypertension guideline."
            ),

            "unsafe": (
                "❌ Request blocked.\n\n"
                "The request could not be processed safely."
            )

        }

        return messages.get(
            risk,
            "❌ Request cannot be processed safely."
        )