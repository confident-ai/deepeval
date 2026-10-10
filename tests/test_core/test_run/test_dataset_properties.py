from deepeval.test_case import LLMTestCase
from deepeval.test_run.test_run import TestRun


def _test_case(dataset_id=None, dataset_version=None) -> LLMTestCase:
    test_case = LLMTestCase(input="What is 2 + 2?")
    test_case._dataset_id = dataset_id
    test_case._dataset_version = dataset_version
    return test_case


def test_dataset_version_is_sent_as_dataset_version():
    test_run = TestRun()

    test_run.set_dataset_properties(_test_case("dataset-id", "00.00.02"))

    body = test_run.model_dump(by_alias=True, exclude_none=True)
    assert body["datasetId"] == "dataset-id"
    assert body["datasetVersion"] == "00.00.02"


def test_dataset_version_is_omitted_when_unknown():
    test_run = TestRun()

    test_run.set_dataset_properties(_test_case("dataset-id"))

    body = test_run.model_dump(by_alias=True, exclude_none=True)
    assert "datasetVersion" not in body


def test_dataset_version_comes_from_the_same_test_case_as_dataset_id():
    test_run = TestRun()

    test_run.set_dataset_properties(_test_case("first-dataset"))
    test_run.set_dataset_properties(_test_case("second-dataset", "00.00.02"))

    assert test_run.dataset_id == "first-dataset"
    assert test_run.dataset_version is None
