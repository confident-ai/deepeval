"""Client for Twilix Evals
"""
import os
from .api import Api

# GET localhost:3000/api/v1/pipeline/eval
# curl --location --request GET 'localhost:3000/api/v1/pipeline/eval' \
# --header 'Authorization: Bearer ysmtQIeNVDATpGlxoLWpWvb6+U4cOEG6PKUZOcBTBBw=' \
# --header 'Content-Type: application/json' \
# --data '{
#   "query": "What is the meaning of life?",
#   "expected_resposne": "42",
#   "tags": ["misc"]
# }'


class Evaluator(Api):
    def __init__(self, api_key: str = None, **kwargs):
        if api_key is None:
            api_key = os.environ["TWILIX_API_KEY"]
        self.api_key = api_key
        super().__init__(api_key=api_key, **kwargs)

    def add_ground_truth(self, query: str, expected_response: str, tags: list = None):
        """Add ground truth"""
        return self.post_request(
            endpoint="/v1/pipeline/eval",
            data={"query": query, "expected_response": expected_response, "tags": tags},
        )

    def build_ground_truth(self):
        pass
