"""Client for Twilix Evals
"""
import os
import getpass
from .api import Api
from .get_api_key import _get_api_key
from typing import Optional, List, Dict


class Client(Api):
    """API with convenience functions"""

    _implementation_id: Dict[str, str] = {}

    def __init__(self, api_key: str = None, local_mode: bool = False, **kwargs):
        if api_key is None or api_key == "":
            api_key = _get_api_key()
        self.api_key = api_key
        self.local_mode = local_mode
        if self.local_mode:
            self.data = []
        super().__init__(api_key=api_key, **kwargs)

    def add_ground_truth(self, query: str, expected_response: str, tags: list = None):
        """Add ground truth"""
        if self.local_mode:
            self.data.append(
                {"query": query, "expected_response": expected_response, "tags": tags}
            )
        return self.post_request(
            endpoint="v1/pipeline/eval",
            body={"query": query, "expected_response": expected_response, "tags": tags},
        )

    async def add_ground_truth_async(
        self, query: str, expected_response: str, tags: list = None
    ):
        """Add ground truth"""
        return self.post_request(
            endpoint="v1/pipeline/eval",
            body={"query": query, "expected_response": expected_response, "tags": tags},
        )

    def _get_ground_truths(self, take: int = 50, skip: int = 0):
        return self.get_request(
            endpoint="v1/pipeline/eval", params={"take": take, "skip": skip}
        )

    def _post_evaluation_result(
        self, evaluation_score: float, pipeline_id: str, test_id: str = None
    ):
        return self.post_request(
            endpoint="v1/pipeline/eval/run",
            body={
                "testId": test_id,
                "evaluationScore": evaluation_score,
                "pipelineGroupName": pipeline_id,
            },
        )

    def get_implementation_id_by_name(
        self, name: str, description: Optional[str] = None
    ):
        """Gets implementation. If none exists, it creates one."""
        if name in self._implementation_id:
            return self._implementation_id[name]
        implementations: List[Dict] = self.list_implementations()
        imp_id = None
        for imp in implementations:
            if imp["name"] == name:
                imp_id = imp["id"]
        if imp_id is None:
            created_imp = self.create_implementation(name=name, description=description)
            imp_id = created_imp["id"]
        # to avoid hammering the server
        self._implementation_id[name] = imp_id
        return imp_id
