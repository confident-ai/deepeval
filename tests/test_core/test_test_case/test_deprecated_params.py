import warnings


def test_llm_test_case_params_alias_is_single_turn_params():
    from deepeval.test_case import SingleTurnParams

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from deepeval.test_case import LLMTestCaseParams

    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "expected DeprecationWarning when importing LLMTestCaseParams"
    assert LLMTestCaseParams is SingleTurnParams
    assert LLMTestCaseParams.METADATA is SingleTurnParams.METADATA


def test_turn_params_alias_is_multi_turn_params():
    from deepeval.test_case import MultiTurnParams

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from deepeval.test_case import TurnParams

    assert any(
        issubclass(w.category, DeprecationWarning) for w in caught
    ), "expected DeprecationWarning when importing TurnParams"
    assert TurnParams is MultiTurnParams
    assert TurnParams.METADATA is MultiTurnParams.METADATA


def test_llm_test_case_params_alias_from_submodule():
    from deepeval.test_case.llm_test_case import SingleTurnParams

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from deepeval.test_case.llm_test_case import LLMTestCaseParams

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert LLMTestCaseParams is SingleTurnParams


def test_turn_params_alias_from_submodule():
    from deepeval.test_case.conversational_test_case import MultiTurnParams

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        from deepeval.test_case.conversational_test_case import TurnParams

    assert any(issubclass(w.category, DeprecationWarning) for w in caught)
    assert TurnParams is MultiTurnParams
