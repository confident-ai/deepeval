from deepeval.openai.extractors import (
    extract_input_parameters_from_completion,
)


def test_extract_input_parameters_stringifies_multimodal_user_content_list():
    # simulate OpenAI chat payload where the user content is a list of parts
    kwargs = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What fruit is shown?"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/banana.jpg"},
                    },
                ],
            },
        ],
    }

    params = extract_input_parameters_from_completion(kwargs)

    # ensure the params input is a string and not a list of parts
    # or a validation error will result
    # but check that content from the list is retained in the string
    assert isinstance(params.input, str)
    assert "What fruit is shown?" in params.input
    assert "banana.jpg" in params.input or "image" in params.input
