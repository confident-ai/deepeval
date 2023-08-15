"""Client for Twilix Evals
"""
import os
import asyncio
import getpass
from tqdm.auto import tqdm
from .pipeline import Pipeline
from .api import Api
from .metrics.metric import Metric
from .query_generator import BEIRQueryGenerator, QueryGenerator
from typing import Optional, List


class Evaluator(Api):
    def __init__(self, api_key: str = None, local_mode: bool = False, **kwargs):
        if api_key is None:
            if "TWILIX_API_KEY" not in os.environ:
                api_key = getpass.getpass(
                    "Grab your API key from https://app.twilix.io"
                )
            else:
                api_key = os.environ["TWILIX_API_KEY"]
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

    def evaluate(self, pipeline: Pipeline, metric: Metric, test_id: str = None):
        #  {
        #     "created_at": "2023-08-10T09:28:44.782Z",
        #     "query": "Example",
        #     "expected_response": "Customer success response is here",
        #     "Tags": [
        #         "sample"
        #     ],
        #     "EvalRuns": []
        # },
        truths = self._get_ground_truths()
        for t in tqdm(truths["runs"]):
            result = pipeline.result_function(t["query"])
            expected_result = t["expected_response"]
            score = metric.measure(result, expected_result)
            result = self._post_evaluation_result(
                evaluation_score=score,
                pipeline_id=pipeline.pipeilne_id,
                test_id=t["id"],
            )
            print(result)

    async def generate_queries_async(
        self,
        texts: List,
        tags: Optional[List] = None,
        num_queries: int = 3,
        generator: QueryGenerator = None,
        max_workers: int = 3,  # Maximum number of threads and blocks until it's free
    ):
        """Generate these queries"""
        if generator is None:
            generator = BEIRQueryGenerator()

        semaphore = asyncio.Semaphore(max_workers)

        for i, t in enumerate(tqdm(texts)):
            async with semaphore:
                results = generator.generate_queries(
                    texts=[t], tags=tags, num_queries=num_queries
                )
                for j, r in enumerate(results):
                    asyncio.create_task(
                        self.add_ground_truth_async(
                            query=r, expected_response=t, tags=[tags[i]]
                        )
                    )
        assert len(results) > 0, "Provide some results."

    def generate_queries(
        self,
        texts: List,
        tags: Optional[List] = None,
        num_queries: int = 3,
        generator: QueryGenerator = None,
        max_workers: int = 3,
    ):
        asyncio.run(
            self.generate_queries_async(
                texts=texts,
                tags=tags,
                num_queries=num_queries,
                generator=generator,
                max_workers=max_workers,
            )
        )
