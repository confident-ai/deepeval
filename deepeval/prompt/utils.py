import re
import uuid
from jinja2 import Template
from typing import Any, Dict, Type, Optional, List, Match
from pydantic import BaseModel, create_model

from deepeval.prompt.api import (
    PromptInterpolationType,
    OutputSchema,
    SchemaDataType,
    OutputSchemaField,
)

###################################
# Interpolation
###################################


def interpolate_mustache(text: str, **kwargs: Any) -> str:
    """Interpolate using Mustache format: {{variable}}"""

    def replace_match(match: Match[str]) -> str:
        var_name = match.group(1)
        if var_name in kwargs:
            return str(kwargs[var_name])
        # Raise error for missing variables to maintain consistency
        raise KeyError(f"Missing variable in template: {var_name}")

    return re.sub(r"\{\{([a-zA-Z_][a-zA-Z0-9_]*)\}\}", replace_match, text)


def interpolate_mustache_with_space(text: str, **kwargs: Any) -> str:
    """Interpolate using Mustache with space format: {{ variable }}"""

    def replace_match(match: Match[str]) -> str:
        var_name = match.group(1)
        if var_name in kwargs:
            return str(kwargs[var_name])
        # Raise error for missing variables to maintain consistency
        raise KeyError(f"Missing variable in template: {var_name}")

    return re.sub(r"\{\{ ([a-zA-Z_][a-zA-Z0-9_]*) \}\}", replace_match, text)


def interpolate_fstring(text: str, **kwargs: Any) -> str:
    """Interpolate using F-string format: {variable}"""

    def replace_match(match: Match[str]) -> str:
        var_name = match.group(1)
        if var_name in kwargs:
            return str(kwargs[var_name])
        # Raise error for missing variables to maintain consistency
        raise KeyError(f"Missing variable in template: {var_name}")

    return re.sub(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace_match, text)


def interpolate_dollar_brackets(text: str, **kwargs: Any) -> str:
    """Interpolate using Dollar Brackets format: ${variable}"""

    def replace_match(match: Match[str]) -> str:
        var_name = match.group(1)
        if var_name in kwargs:
            return str(kwargs[var_name])
        # Raise error for missing variables to maintain consistency
        raise KeyError(f"Missing variable in template: {var_name}")

    return re.sub(r"\$\{([a-zA-Z_][a-zA-Z0-9_]*)\}", replace_match, text)


def interpolate_jinja(text: str, **kwargs: Any) -> str:
    template = Template(text)
    return template.render(**kwargs)


def interpolate_text(
    interpolation_type: PromptInterpolationType, text: str, **kwargs: Any
) -> str:
    """Apply the appropriate interpolation method based on the type"""
    if interpolation_type == PromptInterpolationType.MUSTACHE:
        return interpolate_mustache(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.MUSTACHE_WITH_SPACE:
        return interpolate_mustache_with_space(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.FSTRING:
        return interpolate_fstring(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.DOLLAR_BRACKETS:
        return interpolate_dollar_brackets(text, **kwargs)
    elif interpolation_type == PromptInterpolationType.JINJA:
        return interpolate_jinja(text, **kwargs)


###################################
# Output Schema Deconstruction
###################################

schema_type_map: Dict[str, Any] = {
    SchemaDataType.STRING.value: str,
    SchemaDataType.INTEGER.value: int,
    SchemaDataType.FLOAT.value: float,
    SchemaDataType.BOOLEAN.value: bool,
    SchemaDataType.NULL.value: type(None),
    SchemaDataType.OBJECT.value: dict,
}


def construct_nested_base_model(
    parent: OutputSchemaField,
    parent_id_map: Dict[Optional[str], List[OutputSchemaField]],
    model_name: str,
) -> Type[BaseModel]:
    child_fields: Dict[str, tuple] = {}
    for child in parent_id_map.get(parent.id, []):
        child_type = (
            child.type.value if hasattr(child.type, "value") else child.type
        )
        if child_type == SchemaDataType.OBJECT.value:
            python_type = construct_nested_base_model(
                child, parent_id_map, child.name
            )
        else:
            python_type = schema_type_map.get(child_type, Any)
        default = ... if child.required else None
        child_fields[child.name or child.id] = (python_type, default)
    return create_model(model_name, **child_fields)


def construct_base_model(
    schema: Optional[OutputSchema] = None,
) -> Type[BaseModel]:
    if not schema:
        return None
    if not schema.fields:
        return create_model(schema.name)

    parent_id_map: Dict[Optional[str], List[OutputSchemaField]] = {}
    for field in schema.fields:
        parent_id = field.parent_id or None
        if parent_id_map.get(parent_id) is None:
            parent_id_map[parent_id] = []
        parent_id_map[parent_id].append(field)

    root_fields: Dict[str, tuple] = {}
    for field in parent_id_map.get(None, []):
        field_type = (
            field.type.value if hasattr(field.type, "value") else field.type
        )
        if field_type == SchemaDataType.OBJECT.value:
            python_type = construct_nested_base_model(
                field, parent_id_map, field.name
            )
        else:
            python_type = schema_type_map.get(field_type, Any)
        default = ... if field.required else None
        root_fields[field.name] = (python_type, default)

    return create_model(schema.name, **root_fields)


###################################
# Output Schema Construction
###################################


def _process_model(
    model_class: Type[BaseModel],
    parent_id: Optional[str] = None,
) -> List[OutputSchemaField]:
    fields = []
    model_fields = model_class.model_fields
    for field_name, field_info in model_fields.items():
        field_id = str(uuid.uuid4())
        annotation = field_info.annotation
        field_type = "STRING"
        if annotation == str:
            field_type = "STRING"
        elif annotation == int:
            field_type = "INTEGER"
        elif annotation == float:
            field_type = "FLOAT"
        elif annotation == bool:
            field_type = "BOOLEAN"
        elif annotation == list:
            raise ValueError("Unsupported structured output: list")
        elif annotation == dict:
            raise ValueError("Unsupported structured output: dict")
        elif (
            hasattr(annotation, "__bases__")
            and BaseModel in annotation.__bases__
        ):
            field_type = "OBJECT"
            parent_field = OutputSchemaField(
                id=field_id,
                name=field_name,
                type=field_type,
                required=field_info.default is ...,
                parent_id=parent_id,
            )
            fields.append(parent_field)
            nested_fields = _process_model(annotation, field_id)
            fields.extend(nested_fields)
            continue
        required = field_info.default is ...
        fields.append(
            OutputSchemaField(
                id=field_id,
                name=field_name,
                type=field_type,
                required=required,
                parent_id=parent_id,
            )
        )
    return fields


def construct_output_schema(
    base_model_class: Optional[Type[BaseModel]] = None,
) -> Optional[OutputSchema]:
    if base_model_class is None:
        return None
    all_fields = _process_model(base_model_class)
    return OutputSchema(fields=all_fields, name=base_model_class.__name__)
