import re
import uuid
from jinja2 import Template
from typing import Any, Dict, Type, Optional, List
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


def interpolate_mustache(text: str, **kwargs) -> str:
    """Interpolate using Mustache format: {{variable}}"""
    formatted_template = re.sub(r"\{\{(\w+)\}\}", r"{\1}", text)
    return formatted_template.format(**kwargs)


def interpolate_mustache_with_space(text: str, **kwargs) -> str:
    """Interpolate using Mustache with space format: {{ variable }}"""
    formatted_template = re.sub(r"\{\{ (\w+) \}\}", r"{\1}", text)
    return formatted_template.format(**kwargs)


def interpolate_fstring(text: str, **kwargs) -> str:
    """Interpolate using F-string format: {variable}"""
    return text.format(**kwargs)


def interpolate_dollar_brackets(text: str, **kwargs) -> str:
    """Interpolate using Dollar Brackets format: ${variable}"""
    formatted_template = re.sub(r"\$\{(\w+)\}", r"{\1}", text)
    return formatted_template.format(**kwargs)


def interpolate_jinja(text: str, **kwargs) -> str:
    template = Template(text)
    return template.render(**kwargs)


def interpolate_text(
    interpolation_type: PromptInterpolationType, text: str, **kwargs
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

    raise ValueError(f"Unsupported interpolation type: {interpolation_type}")


###################################
# Output Schema Deconstruction
###################################

schema_type_map: Dict[SchemaDataType, Any] = {
    SchemaDataType.STRING: str,
    SchemaDataType.INTEGER: int,
    SchemaDataType.FLOAT: float,
    SchemaDataType.BOOLEAN: bool,
    SchemaDataType.NULL: type(None),
    SchemaDataType.OBJECT: dict,  # overridden with a nested base model
}


def construct_nested_base_model(
    parent: OutputSchemaField,
    parent_id_map: Dict[Optional[str], List[OutputSchemaField]],
    model_name: str,
) -> Type[BaseModel]:
    child_fields: Dict[str, tuple] = {}
    for child in parent_id_map.get(parent.id, []):
        if child.type == SchemaDataType.OBJECT:
            python_type = construct_nested_base_model(
                child, parent_id_map, child.name
            )
        else:
            python_type = schema_type_map.get(child.type, Any)
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
    for field in parent_id_map.get(None):
        if field.type == SchemaDataType.OBJECT:
            continue
        else:
            python_type = schema_type_map.get(field.type)
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
    print(base_model_class)
    if base_model_class is None:
        return None
    all_fields = _process_model(base_model_class)
    return OutputSchema(fields=all_fields, name=base_model_class.__name__)
