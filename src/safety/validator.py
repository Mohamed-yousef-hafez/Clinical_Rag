"""
Module:
Output Validation
"""


class OutputValidator:

    def validate(self, answer):

        if not answer:

            return False

        if not isinstance(answer, str):

            return False

        if not answer.strip():

            return False

        return True