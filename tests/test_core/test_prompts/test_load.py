import pytest
import os
import tempfile
from deepeval.prompt.prompt import Prompt


class TestPromptLoad:

    def test_load_plain_text_file(self):
        prompt = Prompt()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write("You are a helpful assistant.")
            temp_file_path = temp_file.name

        try:
            prompt.load(temp_file_path)
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
            assert prompt.text_template == "You are a helpful assistant."
        finally:
            os.unlink(temp_file_path)

    def test_load_json_list_format(self):
        prompt = Prompt()
        json_content = """[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hello, how are you?"
  }
]"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file_path = temp_file.name

        try:
            prompt.load(temp_file_path)
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
            assert prompt.messages_template is not None
            assert len(prompt.messages_template) == 2
            assert prompt.messages_template[0].role == "system"
            assert (
                prompt.messages_template[0].content
                == "You are a helpful assistant."
            )
            assert prompt.messages_template[1].role == "user"
            assert prompt.messages_template[1].content == "Hello, how are you?"
        finally:
            os.unlink(temp_file_path)

    def test_load_json_list_format_txt_extension(self):
        prompt = Prompt()
        json_content = """[
  {
    "role": "system",
    "content": "You are a helpful assistant."
  },
  {
    "role": "user",
    "content": "Hello, how are you?"
  }
]"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file_path = temp_file.name

        try:
            prompt.load(temp_file_path)
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
            assert prompt.messages_template is not None
            assert len(prompt.messages_template) == 2
        finally:
            os.unlink(temp_file_path)

    def test_load_json_dict_format_with_correct_key(self):
        prompt = Prompt()
        json_content = """{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    },
    {
      "role": "user",
      "content": "Hello, how are you?"
    }
  ]
}"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file_path = temp_file.name

        try:
            prompt.load(temp_file_path, messages_key="messages")
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
            assert prompt.messages_template is not None
            assert len(prompt.messages_template) == 2
            assert prompt.messages_template[0].role == "system"
            assert prompt.messages_template[1].role == "user"
        finally:
            os.unlink(temp_file_path)

    def test_load_json_dict_format_without_messages_key_raises_error(self):
        prompt = Prompt()
        json_content = """{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    }
  ]
}"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file_path = temp_file.name

        try:
            with pytest.raises(
                ValueError,
                match="messages `key` must be provided if file is a dictionary",
            ):
                prompt.load(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_load_json_dict_format_with_wrong_key(self):
        prompt = Prompt()
        json_content = """{
  "messages": [
    {
      "role": "system",
      "content": "You are a helpful assistant."
    }
  ]
}"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file_path = temp_file.name

        try:
            with pytest.raises(KeyError):
                prompt.load(temp_file_path, messages_key="wrong_key")
        finally:
            os.unlink(temp_file_path)

    def test_load_unsupported_file_extension(self):
        prompt = Prompt()
        with tempfile.NamedTemporaryFile(
            suffix=".py", delete=False
        ) as temp_file:
            temp_file.write(b"print('hello')")
            temp_file_path = temp_file.name
        try:
            with pytest.raises(
                ValueError, match="Only .json and .txt files are supported"
            ):
                prompt.load(temp_file_path)
        finally:
            os.unlink(temp_file_path)

    def test_load_invalid_json_falls_back_to_text(self):
        prompt = Prompt()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write("This is not valid JSON content")
            temp_file_path = temp_file.name
        try:
            prompt.load(temp_file_path)
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
            assert prompt.text_template == "This is not valid JSON content"
        finally:
            os.unlink(temp_file_path)

    def test_load_malformed_messages_falls_back_to_text(self):
        prompt = Prompt()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write('[{"invalid": "structure"}]')
            temp_file_path = temp_file.name
        try:
            prompt.load(temp_file_path)
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
            assert prompt.text_template == '[{"invalid": "structure"}]'
        finally:
            os.unlink(temp_file_path)

    def test_load_sets_correct_alias_from_filename(self):
        prompt = Prompt()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False
        ) as temp_file:
            temp_file.write("You are a helpful assistant.")
            temp_file_path = temp_file.name

        try:
            prompt.load(temp_file_path)
            assert (
                prompt.alias == os.path.basename(temp_file_path).split(".")[0]
            )
        finally:
            os.unlink(temp_file_path)

    def test_load_dict_with_custom_messages_key(self):
        prompt = Prompt()
        json_content = (
            '{"custom_messages": [{"role": "system", "content": "Test"}]}'
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as temp_file:
            temp_file.write(json_content)
            temp_file_path = temp_file.name

        try:
            prompt.load(temp_file_path, messages_key="custom_messages")
            assert hasattr(prompt, "messages_template")
            assert len(prompt.messages_template) == 1
            assert prompt.messages_template[0].role == "system"
            assert prompt.messages_template[0].content == "Test"
        finally:
            os.unlink(temp_file_path)
