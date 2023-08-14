from typing import Optional, List


class GroundTruth:
    query: str
    expected_response: str
    tags: Optional[List] = None
