import re
import uuid
from jinja2 import Template
from typing import (
    Any,
    Dict,
    Type,
    Optional,
    List,
    Match,
    Tuple,
    Union,
    get_origin,
    get_args,
)
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

        # Unwrap Optional[X] (Union[X, None]) to its inner type
        origin = get_origin(annotation)
        if origin is Union:
            args = [a for a in get_args(annotation) if a is not type(None)]
            if len(args) == 1:
                annotation = args[0]
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
        elif hasattr(annotation, "__mro__") and BaseModel in annotation.__mro__:
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


###################################
# Tool Input Schema Construction
###################################

_TOOL_SCHEMA_KEYWORDS = {
    "type",
    "description",
    "enum",
    "properties",
    "items",
    "required",
}

_TOOL_SCHEMA_ANNOTATIONS = {"title", "default", "examples"}

_UNION_KEYWORDS = ("anyOf", "oneOf", "allOf")
_NULL_SCHEMA = {"type": "null"}


def _inline_tool_schema_ref(
    node: Dict[str, Any],
    defs: Dict[str, Any],
    location: str,
    expanding: Tuple[str, ...],
) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    name = node.pop("$ref").rsplit("/", 1)[-1]
    if name in expanding:
        raise ValueError(
            f"{location} refers to '{name}' recursively, which a tool "
            "schema can't express."
        )
    return {**defs[name], **node}, (*expanding, name)


def _unwrap_tool_schema_union(
    node: Dict[str, Any], keyword: str, location: str
) -> Dict[str, Any]:
    branches = [b for b in node.pop(keyword) if b != _NULL_SCHEMA]
    if len(branches) != 1:
        raise ValueError(
            f"{location} is a union of several types. A tool schema supports "
            "only Optional[...] of one type."
        )
    return {**branches[0], **node}


def _resolve_tool_schema_node(
    node: Dict[str, Any],
    defs: Dict[str, Any],
    location: str,
    expanding: Tuple[str, ...],
) -> Tuple[Dict[str, Any], Tuple[str, ...]]:
    node = dict(node)
    while True:
        if "$ref" in node:
            node, expanding = _inline_tool_schema_ref(
                node, defs, location, expanding
            )
            continue
        keyword = next((k for k in _UNION_KEYWORDS if k in node), None)
        if keyword is None:
            return node, expanding
        node = _unwrap_tool_schema_union(node, keyword, location)


def _normalize_tool_schema_node(
    node: Dict[str, Any],
    defs: Dict[str, Any],
    path: List[str],
    expanding: Tuple[str, ...],
) -> Dict[str, Any]:
    location = ".".join(["root", *path])
    node, expanding = _resolve_tool_schema_node(node, defs, location, expanding)

    if "const" in node:
        node["enum"] = [node.pop("const")]
    if node.pop("additionalProperties", False) is not False:
        raise ValueError(
            f"{location} is a free-form dict. Describe its keys with a "
            "pydantic model instead."
        )

    for keyword in _TOOL_SCHEMA_ANNOTATIONS:
        node.pop(keyword, None)
    unsupported = sorted(set(node) - _TOOL_SCHEMA_KEYWORDS)
    if unsupported:
        raise ValueError(
            f"{location} uses {', '.join(unsupported)}, which a tool schema "
            f"doesn't support. Use only {', '.join(sorted(_TOOL_SCHEMA_KEYWORDS))}."
        )
    if "type" not in node:
        raise ValueError(
            f"{location} has no type. Give the field a concrete type."
        )

    if "properties" in node:
        node["properties"] = {
            name: _normalize_tool_schema_node(
                child, defs, [*path, name], expanding
            )
            for name, child in node["properties"].items()
        }
    if "items" in node:
        node["items"] = _normalize_tool_schema_node(
            node["items"], defs, [*path, "items"], expanding
        )
    return node


def construct_tool_input_schema(
    base_model_class: Type[BaseModel],
) -> Dict[str, Any]:
    schema = base_model_class.model_json_schema()
    defs = schema.pop("$defs", {})
    return _normalize_tool_schema_node(schema, defs, [], ())
