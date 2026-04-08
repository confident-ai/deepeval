import re
import uuid
from jinja2 import Template
from typing import Any, Dict, Type, Optional, List, Match, get_origin, get_args
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
    SchemaDataType.ARRAY.value: list,
}


def _resolve_field_type(
    field: OutputSchemaField,
    parent_id_map: Dict[Optional[str], List[OutputSchemaField]],
) -> Any:
    field_type = (
        field.type.value if hasattr(field.type, "value") else field.type
    )
    if field_type == SchemaDataType.OBJECT.value:
        return construct_nested_base_model(field, parent_id_map, field.name)
    elif field_type == SchemaDataType.ARRAY.value:
        children = parent_id_map.get(field.id, [])
        if children:
            item_type = _resolve_field_type(children[0], parent_id_map)
            return List[item_type]
        return List[Any]
    else:
        return schema_type_map.get(field_type, Any)


def construct_nested_base_model(
    parent: OutputSchemaField,
    parent_id_map: Dict[Optional[str], List[OutputSchemaField]],
    model_name: str,
) -> Type[BaseModel]:
    child_fields: Dict[str, tuple] = {}
    for child in parent_id_map.get(parent.id, []):
        python_type = _resolve_field_type(child, parent_id_map)
        default = ... if child.required else None
        child_fields[child.name or child.id] = (python_type, default)
    return create_model(model_name, **child_fields)


def construct_base_model(
    schema: Optional[OutputSchema] = None,
) -> Type[BaseModel]:
    if not schema:
        return None
    if not schema.fields:
        return create_model(schema.name or "EmptySchema")

    parent_id_map: Dict[Optional[str], List[OutputSchemaField]] = {}
    for field in schema.fields:
        parent_id = field.parent_id or None
        if parent_id_map.get(parent_id) is None:
            parent_id_map[parent_id] = []
        parent_id_map[parent_id].append(field)

    root_fields: Dict[str, tuple] = {}
    for field in parent_id_map.get(None, []):
        python_type = _resolve_field_type(field, parent_id_map)
        default = ... if field.required else None
        root_fields[field.name] = (python_type, default)

    return create_model(schema.name or "Schema", **root_fields)


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
        origin = get_origin(annotation)
        if annotation == str:
            field_type = "STRING"
        elif annotation == int:
            field_type = "INTEGER"
        elif annotation == float:
            field_type = "FLOAT"
        elif annotation == bool:
            field_type = "BOOLEAN"
        elif annotation == list:
            raise ValueError(
                "Unsupported structured output: bare list. "
                "Use List[str], List[int], or List[YourModel] instead."
            )
        elif annotation == dict:
            raise ValueError("Unsupported structured output: dict")
        elif origin is list:
            args = get_args(annotation)
            item_type = args[0] if args else str
            array_field = OutputSchemaField(
                id=field_id,
                name=field_name,
                type="ARRAY",
                required=field_info.is_required(),
                parent_id=parent_id,
            )
            fields.append(array_field)
            item_field_id = str(uuid.uuid4())
            if (
                hasattr(item_type, "__bases__")
                and BaseModel in item_type.__mro__
            ):
                item_field = OutputSchemaField(
                    id=item_field_id,
                    name=item_type.__name__,
                    type="OBJECT",
                    required=True,
                    parent_id=field_id,
                )
                fields.append(item_field)
                nested_fields = _process_model(item_type, item_field_id)
                fields.extend(nested_fields)
            else:
                primitive_map = {
                    str: "STRING",
                    int: "INTEGER",
                    float: "FLOAT",
                    bool: "BOOLEAN",
                }
                item_schema_type = primitive_map.get(item_type, "STRING")
                item_field = OutputSchemaField(
                    id=item_field_id,
                    name=field_name,
                    type=item_schema_type,
                    required=True,
                    parent_id=field_id,
                )
                fields.append(item_field)
            continue
        elif (
            hasattr(annotation, "__bases__")
            and BaseModel in annotation.__bases__
        ):
            field_type = "OBJECT"
            parent_field = OutputSchemaField(
                id=field_id,
                name=field_name,
                type=field_type,
                required=field_info.is_required(),
                parent_id=parent_id,
            )
            fields.append(parent_field)
            nested_fields = _process_model(annotation, field_id)
            fields.extend(nested_fields)
            continue
        required = field_info.is_required()
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


def output_schema_to_json_schema(
    schema: Optional[OutputSchema] = None,
) -> Dict[str, Any]:
    if not schema or not schema.fields:
        return {
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        }

    # Build parent-child mapping
    children_map: Dict[Optional[str], List[OutputSchemaField]] = {}
    for field in schema.fields:
        parent_id = field.parent_id
        children_map.setdefault(parent_id, []).append(field)

    # Map SchemaDataType to JSON Schema types
    def map_type(dtype: SchemaDataType) -> str:
        return {
            SchemaDataType.STRING: "string",
            SchemaDataType.INTEGER: "integer",
            SchemaDataType.FLOAT: "number",
            SchemaDataType.BOOLEAN: "boolean",
            SchemaDataType.OBJECT: "object",
            SchemaDataType.ARRAY: "array",
            SchemaDataType.NULL: "null",
        }.get(dtype, "string")

    def build_node(field_list: List[OutputSchemaField]) -> Dict[str, Any]:
        properties = {}
        required_fields = []

        for field in field_list:
            field_type = (
                field.type.value if hasattr(field.type, "value") else field.type
            )
            normalized_type = (
                SchemaDataType(field_type)
                if not isinstance(field_type, SchemaDataType)
                else field_type
            )

            field_schema = {"type": map_type(normalized_type)}

            # Add description if available
            if field.description:
                field_schema["description"] = field.description

            if field_type == SchemaDataType.ARRAY.value:
                children = children_map.get(field.id, [])
                if children:
                    item_field = children[0]
                    item_type = (
                        item_field.type.value
                        if hasattr(item_field.type, "value")
                        else item_field.type
                    )
                    item_normalized = (
                        SchemaDataType(item_type)
                        if not isinstance(item_type, SchemaDataType)
                        else item_type
                    )
                    item_schema = {"type": map_type(item_normalized)}
                    if item_type == SchemaDataType.OBJECT.value:
                        obj_children = children_map.get(item_field.id, [])
                        if obj_children:
                            nested = build_node(obj_children)
                            item_schema.update(nested)
                        else:
                            item_schema["properties"] = {}
                            item_schema["additionalProperties"] = False
                    field_schema["items"] = item_schema
                else:
                    field_schema["items"] = {}

            elif field_type == SchemaDataType.OBJECT.value:
                children = children_map.get(field.id, [])
                if children:
                    nested = build_node(children)
                    field_schema.update(nested)
                else:
                    field_schema["properties"] = {}
                    field_schema["additionalProperties"] = False

            properties[field.name] = field_schema
            if field.required:
                required_fields.append(field.name)

        schema_dict = {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        }

        if required_fields:
            schema_dict["required"] = required_fields

        return schema_dict

    root_fields = children_map.get(None, [])
    return build_node(root_fields)
