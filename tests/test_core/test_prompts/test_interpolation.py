"""
Comprehensive tests for all prompt interpolation methods.
Tests edge cases including JSON, special characters, missing variables, etc.
"""

import pytest
from deepeval.prompt.utils import (
    interpolate_mustache,
    interpolate_mustache_with_space,
    interpolate_fstring,
    interpolate_dollar_brackets,
    interpolate_jinja,
)


class TestInterpolateMustache:
    """Tests for {{variable}} format"""

    def test_simple_variable(self):
        text = "Hello {{name}}"
        result = interpolate_mustache(text, name="World")
        assert result == "Hello World"

    def test_multiple_variables(self):
        text = "{{greeting}} {{name}}, you have {{count}} messages"
        result = interpolate_mustache(
            text, greeting="Hello", name="Alice", count=5
        )
        assert result == "Hello Alice, you have 5 messages"

    def test_variable_with_underscore(self):
        text = "User ID: {{user_id}}"
        result = interpolate_mustache(text, user_id="12345")
        assert result == "User ID: 12345"

    def test_variable_starting_with_underscore(self):
        text = "Private: {{_private}}"
        result = interpolate_mustache(text, _private="secret")
        assert result == "Private: secret"

    def test_variable_with_numbers(self):
        text = "Item: {{item123}}"
        result = interpolate_mustache(text, item123="value")
        assert result == "Item: value"

    def test_missing_variable_raises_keyerror(self):
        text = "Hello {{name}}"
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_mustache(text)

    def test_missing_one_of_multiple_variables(self):
        text = "{{greeting}} {{name}}"
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_mustache(text, greeting="Hello")

    def test_no_placeholders(self):
        text = "Just plain text with no variables"
        result = interpolate_mustache(text)
        assert result == "Just plain text with no variables"

    def test_empty_string(self):
        text = ""
        result = interpolate_mustache(text)
        assert result == ""

    def test_json_braces_not_replaced(self):
        """The original bug - JSON should remain untouched"""
        text = '{{name}} likes {"key": "value", "count": 42}'
        result = interpolate_mustache(text, name="Alice")
        assert result == 'Alice likes {"key": "value", "count": 42}'

    def test_json_array_not_replaced(self):
        text = '{{name}} items: [{"id": 1}, {"id": 2}]'
        result = interpolate_mustache(text, name="User")
        assert result == 'User items: [{"id": 1}, {"id": 2}]'

    def test_nested_json_structure(self):
        """Complex JSON with nested objects"""
        text = """{{title}}
        {
            "status": "active",
            "nested": {
                "key": "value"
            }
        }"""
        result = interpolate_mustache(text, title="Test")
        assert '"status": "active"' in result
        assert '"nested":' in result
        assert result.startswith("Test")

    def test_multiple_json_objects(self):
        text = '{{user}}: {"a": 1} and {"b": 2}'
        result = interpolate_mustache(text, user="Admin")
        assert result == 'Admin: {"a": 1} and {"b": 2}'

    def test_adjacent_variables(self):
        text = "{{first}}{{second}}"
        result = interpolate_mustache(text, first="Hello", second="World")
        assert result == "HelloWorld"

    def test_variable_at_start(self):
        text = "{{name}} is here"
        result = interpolate_mustache(text, name="Alice")
        assert result == "Alice is here"

    def test_variable_at_end(self):
        text = "Hello {{name}}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "Hello Alice"

    def test_only_variable(self):
        text = "{{name}}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "Alice"

    def test_same_variable_multiple_times(self):
        text = "{{name}} and {{name}} and {{name}}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "Alice and Alice and Alice"

    def test_integer_value(self):
        text = "Count: {{count}}"
        result = interpolate_mustache(text, count=42)
        assert result == "Count: 42"

    def test_float_value(self):
        text = "Price: {{price}}"
        result = interpolate_mustache(text, price=19.99)
        assert result == "Price: 19.99"

    def test_boolean_value(self):
        text = "Active: {{active}}"
        result = interpolate_mustache(text, active=True)
        assert result == "Active: True"

    def test_none_value(self):
        text = "Value: {{value}}"
        result = interpolate_mustache(text, value=None)
        assert result == "Value: None"

    def test_list_value_converts_to_string(self):
        text = "Items: {{items}}"
        result = interpolate_mustache(text, items=[1, 2, 3])
        assert result == "Items: [1, 2, 3]"

    def test_dict_value_converts_to_string(self):
        text = "Data: {{data}}"
        result = interpolate_mustache(text, data={"key": "value"})
        assert "key" in result and "value" in result

    def test_unicode_characters(self):
        text = "{{emoji}} {{chinese}}"
        result = interpolate_mustache(text, emoji="🎉", chinese="你好")
        assert result == "🎉 你好"

    def test_multiline_text(self):
        text = """Line 1: {{var1}}
Line 2: {{var2}}
Line 3: {{var3}}"""
        result = interpolate_mustache(text, var1="A", var2="B", var3="C")
        assert "Line 1: A" in result
        assert "Line 2: B" in result
        assert "Line 3: C" in result

    def test_special_characters_in_text(self):
        text = "Hello {{name}}! How are you? @#$%^&*()"
        result = interpolate_mustache(text, name="Alice")
        assert result == "Hello Alice! How are you? @#$%^&*()"

    def test_single_brace_not_replaced(self):
        """Single braces should be left as-is"""
        text = "{{name}} {alone}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "Alice {alone}"

    def test_triple_braces_not_replaced(self):
        """Triple braces should leave the inner content"""
        text = "{{{name}}}"
        result = interpolate_mustache(text, name="Alice")
        # Should replace {{name}} leaving outer braces
        assert result == "{Alice}"

    def test_invalid_identifier_not_replaced(self):
        """Variables starting with numbers should not be replaced"""
        text = "{{123invalid}} and {{name}}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "{{123invalid}} and Alice"

    def test_invalid_identifier_with_dash(self):
        """Variables with dashes are not valid Python identifiers"""
        text = "{{user-name}} and {{name}}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "{{user-name}} and Alice"

    def test_invalid_identifier_with_dot(self):
        """Variables with dots should not be replaced (we don't support nested access)"""
        text = "{{user.name}} and {{name}}"
        result = interpolate_mustache(text, name="Alice")
        assert result == "{{user.name}} and Alice"

    def test_whitespace_inside_placeholder_not_matched(self):
        """Mustache format doesn't have spaces, so this shouldn't match"""
        text = "{{ name }} is here"
        result = interpolate_mustache(text, name="Alice")
        # Should not replace because of spaces
        assert result == "{{ name }} is here"

    def test_extra_kwargs_ignored(self):
        """Extra kwargs that aren't used should not cause errors"""
        text = "Hello {{name}}"
        result = interpolate_mustache(
            text, name="Alice", extra="ignored", another=123
        )
        assert result == "Hello Alice"

    def test_very_long_variable_name(self):
        long_name = "a" * 100
        text = f"{{{{{long_name}}}}}"
        result = interpolate_mustache(text, **{long_name: "value"})
        assert result == "value"

    def test_case_sensitive(self):
        """Variable names should be case-sensitive"""
        text = "{{Name}} and {{name}}"
        result = interpolate_mustache(text, Name="Alice", name="Bob")
        assert result == "Alice and Bob"

    def test_real_world_prompt_template(self):
        """Test with a realistic prompt template"""
        text = """You are an AI assistant for {{company}}.
        
User: {{user_name}}
Query: {{query}}

Please provide a helpful response."""
        result = interpolate_mustache(
            text,
            company="TechCorp",
            user_name="Alice",
            query="How do I reset my password?",
        )
        assert "TechCorp" in result
        assert "Alice" in result
        assert "How do I reset my password?" in result


class TestInterpolateMustacheWithSpace:
    """Tests for {{ variable }} format"""

    def test_simple_variable(self):
        text = "Hello {{ name }}"
        result = interpolate_mustache_with_space(text, name="World")
        assert result == "Hello World"

    def test_multiple_variables(self):
        text = "{{ greeting }} {{ name }}"
        result = interpolate_mustache_with_space(
            text, greeting="Hello", name="Alice"
        )
        assert result == "Hello Alice"

    def test_json_not_replaced(self):
        text = '{{ name }} likes {"key": "value"}'
        result = interpolate_mustache_with_space(text, name="Alice")
        assert result == 'Alice likes {"key": "value"}'

    def test_without_spaces_not_matched(self):
        """Should NOT match {{name}} without spaces"""
        text = "{{name}} is here"
        result = interpolate_mustache_with_space(text, name="Alice")
        # Should not replace
        assert result == "{{name}} is here"

    def test_single_space_only(self):
        """Should match exactly one space on each side"""
        text = "{{  name  }} is here"  # Double spaces
        result = interpolate_mustache_with_space(text, name="Alice")
        # Should not match with double spaces
        assert result == "{{  name  }} is here"

    def test_missing_variable_raises_keyerror(self):
        text = "Hello {{ name }}"
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_mustache_with_space(text)

    def test_extra_kwargs_ignored(self):
        """Extra kwargs that aren't used should not cause errors"""
        text = "Hello {{ name }}"
        result = interpolate_mustache_with_space(
            text, name="Alice", extra="ignored", another=123
        )
        assert result == "Hello Alice"


class TestInterpolateFString:
    """Tests for {variable} format"""

    def test_simple_variable(self):
        text = "Hello {name}"
        result = interpolate_fstring(text, name="World")
        assert result == "Hello World"

    def test_multiple_variables(self):
        text = "{greeting} {name}, you have {count} messages"
        result = interpolate_fstring(
            text, greeting="Hello", name="Alice", count=5
        )
        assert result == "Hello Alice, you have 5 messages"

    def test_json_braces_not_replaced(self):
        """The key test - JSON should remain untouched"""
        text = '{name} likes {"key": "value", "count": 42}'
        result = interpolate_fstring(text, name="Alice")
        assert result == 'Alice likes {"key": "value", "count": 42}'

    def test_complex_json_structure(self):
        """Test with complex nested JSON structure"""
        text = """{title}
[
    {{
        "id": 1,
        "name": "Product A",
        "price": 29.99,
        "category": "Electronics"
    }}
]"""
        result = interpolate_fstring(text, title="Product List")
        assert result.startswith("Product List")
        assert '"id": 1' in result
        assert '"name":' in result

    def test_missing_variable_raises_keyerror(self):
        text = "Hello {name}"
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_fstring(text)

    def test_empty_braces_not_replaced(self):
        """Empty braces should be left alone"""
        text = "{name} and {}"
        result = interpolate_fstring(text, name="Alice")
        assert result == "Alice and {}"

    def test_nested_json_with_arrays(self):
        text = '{user} data: {{"items": [{{"id": 1}}, {{"id": 2}}]}}'
        result = interpolate_fstring(text, user="Admin")
        assert result == 'Admin data: {{"items": [{{"id": 1}}, {{"id": 2}}]}}'

    def test_variable_with_underscore(self):
        text = "ID: {user_id}"
        result = interpolate_fstring(text, user_id="12345")
        assert result == "ID: 12345"

    def test_integer_value(self):
        text = "Count: {count}"
        result = interpolate_fstring(text, count=42)
        assert result == "Count: 42"

    def test_same_variable_multiple_times(self):
        text = "{name} and {name} and {name}"
        result = interpolate_fstring(text, name="Alice")
        assert result == "Alice and Alice and Alice"

    def test_invalid_identifier_with_dot_not_replaced(self):
        """Dot notation should not be replaced"""
        text = "{user.name} and {name}"
        result = interpolate_fstring(text, name="Alice")
        assert result == "{user.name} and Alice"

    def test_invalid_identifier_with_brackets_not_replaced(self):
        """Bracket notation should not be replaced"""
        text = "{items[0]} and {name}"
        result = interpolate_fstring(text, name="Alice")
        assert result == "{items[0]} and Alice"

    def test_unicode_in_values(self):
        text = "Welcome {name}"
        result = interpolate_fstring(text, name="José")
        assert result == "Welcome José"

    def test_extra_kwargs_ignored(self):
        """Extra kwargs that aren't used should not cause errors"""
        text = "Hello {name}"
        result = interpolate_fstring(
            text, name="Alice", extra="ignored", another=123
        )
        assert result == "Hello Alice"


class TestInterpolateDollarBrackets:
    """Tests for ${variable} format"""

    def test_simple_variable(self):
        text = "Hello ${name}"
        result = interpolate_dollar_brackets(text, name="World")
        assert result == "Hello World"

    def test_multiple_variables(self):
        text = "${greeting} ${name}"
        result = interpolate_dollar_brackets(
            text, greeting="Hello", name="Alice"
        )
        assert result == "Hello Alice"

    def test_json_not_replaced(self):
        text = '${name} likes {"key": "value"}'
        result = interpolate_dollar_brackets(text, name="Alice")
        assert result == 'Alice likes {"key": "value"}'

    def test_regular_braces_not_replaced(self):
        text = "${name} and {other}"
        result = interpolate_dollar_brackets(text, name="Alice")
        assert result == "Alice and {other}"

    def test_dollar_without_braces(self):
        text = "${name} costs $50"
        result = interpolate_dollar_brackets(text, name="Item")
        assert result == "Item costs $50"

    def test_missing_variable_raises_keyerror(self):
        text = "Hello ${name}"
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_dollar_brackets(text)

    def test_variable_with_underscore(self):
        text = "Value: ${user_id}"
        result = interpolate_dollar_brackets(text, user_id="12345")
        assert result == "Value: 12345"

    def test_shell_style_variable_format(self):
        """Common in shell scripts"""
        text = "The path is ${HOME}/documents"
        result = interpolate_dollar_brackets(text, HOME="/home/user")
        assert result == "The path is /home/user/documents"

    def test_extra_kwargs_ignored(self):
        """Extra kwargs that aren't used should not cause errors"""
        text = "Hello ${name}"
        result = interpolate_dollar_brackets(
            text, name="Alice", extra="ignored", another=123
        )
        assert result == "Hello Alice"


class TestInterpolateJinja:
    """Tests for Jinja2 format - uses Jinja2 library directly"""

    def test_simple_variable(self):
        text = "Hello {{ name }}"
        result = interpolate_jinja(text, name="World")
        assert result == "Hello World"

    def test_multiple_variables(self):
        text = "{{ greeting }} {{ name }}"
        result = interpolate_jinja(text, greeting="Hello", name="Alice")
        assert result == "Hello Alice"

    def test_jinja_if_statement(self):
        """Jinja supports control structures"""
        text = "{% if show %}Hello {{ name }}{% endif %}"
        result = interpolate_jinja(text, show=True, name="Alice")
        assert result == "Hello Alice"

    def test_jinja_if_false(self):
        text = "{% if show %}Hello {{ name }}{% endif %}"
        result = interpolate_jinja(text, show=False, name="Alice")
        assert result == ""

    def test_jinja_for_loop(self):
        """Jinja supports loops"""
        text = "{% for item in items %}{{ item }} {% endfor %}"
        result = interpolate_jinja(text, items=["a", "b", "c"])
        assert result == "a b c "

    def test_jinja_filters(self):
        """Jinja supports filters"""
        text = "{{ name|upper }}"
        result = interpolate_jinja(text, name="alice")
        assert result == "ALICE"

    def test_json_with_jinja(self):
        """JSON should work fine with Jinja"""
        text = '{{ name }} data: {"key": "value"}'
        result = interpolate_jinja(text, name="User")
        assert result == 'User data: {"key": "value"}'


class TestEdgeCasesAcrossAllFormats:
    """Cross-cutting edge case tests"""

    def test_all_formats_handle_json(self):
        """All formats should handle JSON correctly"""
        json_text = '{"status": "complete", "count": 42}'

        # Mustache
        result = interpolate_mustache(f"{{{{name}}}} {json_text}", name="Test")
        assert json_text in result

        # Mustache with space
        result = interpolate_mustache_with_space(
            f"{{{{ name }}}} {json_text}", name="Test"
        )
        assert json_text in result

        # F-string
        result = interpolate_fstring(f"{{name}} {json_text}", name="Test")
        assert json_text in result

        # Dollar brackets
        result = interpolate_dollar_brackets(
            f"${{name}} {json_text}", name="Test"
        )
        assert json_text in result

        # Jinja
        result = interpolate_jinja(f"{{{{ name }}}} {json_text}", name="Test")
        assert json_text in result

    def test_all_formats_handle_empty_string(self):
        """All formats should handle empty strings"""
        assert interpolate_mustache("") == ""
        assert interpolate_mustache_with_space("") == ""
        assert interpolate_fstring("") == ""
        assert interpolate_dollar_brackets("") == ""
        assert interpolate_jinja("") == ""

    def test_all_formats_raise_on_missing_variable(self):
        """All formats should raise KeyError on missing variables"""

        with pytest.raises(KeyError):
            interpolate_mustache("{{missing}}")

        with pytest.raises(KeyError):
            interpolate_mustache_with_space("{{ missing }}")

        with pytest.raises(KeyError):
            interpolate_fstring("{missing}")

        with pytest.raises(KeyError):
            interpolate_dollar_brackets("${missing}")

        # Jinja is different - it returns empty string for missing variables by default
        result = interpolate_jinja("{{ missing }}")
        assert result == ""

    def test_all_formats_convert_values_to_string(self):
        """All formats should convert non-string values properly"""
        value = 42

        assert "42" in interpolate_mustache("{{val}}", val=value)
        assert "42" in interpolate_mustache_with_space("{{ val }}", val=value)
        assert "42" in interpolate_fstring("{val}", val=value)
        assert "42" in interpolate_dollar_brackets("${val}", val=value)
        assert "42" in interpolate_jinja("{{ val }}", val=value)

    def test_variable_in_template_but_not_passed_raises_error(self):
        """
        SCENARIO 1: Variable IS in template, but NOT passed as parameter → ERROR
        This ensures users don't forget to provide required template variables.
        """
        # Mustache
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_mustache(
                "Hello {{name}}"
            )  # ❌ name in template, not passed

        # Mustache with space
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_mustache_with_space(
                "Hello {{ name }}"
            )  # ❌ name in template, not passed

        # F-string
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_fstring(
                "Hello {name}"
            )  # ❌ name in template, not passed

        # Dollar brackets
        with pytest.raises(
            KeyError, match="Missing variable in template: name"
        ):
            interpolate_dollar_brackets(
                "Hello ${name}"
            )  # ❌ name in template, not passed

    def test_variable_passed_but_not_in_template_is_ignored(self):
        """
        SCENARIO 2: Variable IS passed, but NOT in template → NO ERROR (ignored)
        Extra parameters that aren't used in the template are silently ignored.
        """
        # Mustache
        result = interpolate_mustache(
            "Hello {{name}}",
            name="Alice",  # ✅ Used
            age=25,  # ✅ Ignored, no error
            city="NYC",  # ✅ Ignored, no error
        )
        assert result == "Hello Alice"

        # Mustache with space
        result = interpolate_mustache_with_space(
            "Hello {{ name }}", name="Bob", extra="ignored"
        )
        assert result == "Hello Bob"

        # F-string
        result = interpolate_fstring(
            "Hello {name}", name="Charlie", unused="ignored"
        )
        assert result == "Hello Charlie"

        # Dollar brackets
        result = interpolate_dollar_brackets(
            "Hello ${name}", name="Dave", other="ignored"
        )
        assert result == "Hello Dave"


class TestRealWorldScenarios:
    """Tests based on real-world use cases"""

    def test_product_catalog_with_json_output(self):
        """Test a product catalog template with JSON structure"""
        template = """Product Catalog

Available items in JSON format:
[
    {{
        "category": "Electronics",
        "name": "{product_name}",
        "sku": {sku},
        "description": "{description}"
    }}
]
"""
        result = interpolate_fstring(
            template,
            product_name="Laptop",
            sku=12345,
            description="High-performance laptop",
        )

        assert "Laptop" in result
        assert "12345" in result
        assert "High-performance laptop" in result
        assert '"category": "Electronics"' in result

    def test_api_request_template(self):
        """API request with JSON body"""
        template = """POST /api/users
Content-Type: application/json

{{
    "username": "{username}",
    "email": "{email}",
    "age": {age}
}}
"""
        result = interpolate_fstring(
            template, username="alice", email="alice@example.com", age=25
        )

        assert "alice" in result
        assert "alice@example.com" in result
        assert "25" in result
        assert '"username":' in result

    def test_markdown_with_code_blocks(self):
        """Markdown template with code blocks"""
        template = """# {title}

## Code Example

```python
def {function_name}():
    data = {{"key": "value"}}
    return data
```
"""
        result = interpolate_fstring(
            template, title="My Document", function_name="get_data"
        )

        assert "My Document" in result
        assert "get_data" in result
        assert '{"key": "value"}' in result

    def test_sql_template_with_json(self):
        """SQL query with JSON data"""
        template = """
        INSERT INTO logs (user_id, metadata)
        VALUES ({user_id}, '{{"action": "login", "timestamp": "2024-01-01"}}');
        """
        result = interpolate_fstring(template, user_id=123)

        assert "123" in result
        assert '"action": "login"' in result
