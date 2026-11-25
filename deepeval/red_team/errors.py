from typing import Optional


class ModelRefusalError(Exception):

    def __init__(self, entity: Optional[str] = None):
        self.message = "Simulator model refused to simulate harmful attacks, this could be because of your model's guardrails, please use a different model."

        if entity:
            self.message += f" for '{entity}'"

        super().__init__(self.message)
