from typing import Dict, Optional, Type, Any, Union

from pydantic import BaseModel
from typing_extensions import TypedDict


class ToolSpec(TypedDict):
    """Specification for a tool that can be used by an LLM.

    Attributes:
        description: A human-readable description of what the tool does.
        inputSchema: JSON Schema defining the expected input parameters.
        name: The unique name of the tool.
    """

    description: str
    inputSchema: dict
    name: str


def _process_referenced_models(
    schema: Dict[str, Any], model: Type[BaseModel]
) -> None:
    """Process referenced models to ensure their docstrings are included.

    This updates the schema in place.

    Args:
        schema: The JSON schema to process
        model: The Pydantic model class
    """
    # Process $defs to add docstrings from the referenced models
    if "$defs" in schema:
        # Look through model fields to find referenced models
        for _, field in model.model_fields.items():
            field_type = field.annotation

            # Handle Optional types - with null checks
            if field_type is not None and hasattr(field_type, "__origin__"):
                origin = field_type.__origin__
                if origin is Union and hasattr(field_type, "__args__"):
                    # Find the non-None type in the Union (for Optional fields)
                    for arg in field_type.__args__:
                        if arg is not type(None):
                            field_type = arg
                            break

            # Check if this is a BaseModel subclass
            if isinstance(field_type, type) and issubclass(
                field_type, BaseModel
            ):
                # Update $defs with this model's information
                ref_name = field_type.__name__
                if ref_name in schema.get("$defs", {}):
                    ref_def = schema["$defs"][ref_name]

                    # Add docstring as description if available
                    if field_type.__doc__ and not ref_def.get("description"):
                        ref_def["description"] = field_type.__doc__.strip()

                    # Recursively process properties in the referenced model
                    _process_properties(ref_def, field_type)


def _process_properties(
    schema_def: Dict[str, Any], model: Type[BaseModel]
) -> None:
    """Process properties in a schema definition to add descriptions from field metadata.

    Args:
        schema_def: The schema definition to update
        model: The model class that defines the schema
    """
    if "properties" in schema_def:
        for prop_name, prop_info in schema_def["properties"].items():
            field = model.model_fields.get(prop_name)

            # Add field description if available and not already set
            if field and field.description and not prop_info.get("description"):
                prop_info["description"] = field.description


def _process_schema_object(
    schema_obj: Dict[str, Any], defs: Dict[str, Any], fully_expand: bool = True
) -> Dict[str, Any]:
    """Process a schema object, typically from $defs, to resolve all nested properties.

    Args:
        schema_obj: The schema object to process
        defs: The definitions dictionary for resolving references
        fully_expand: Whether to fully expand nested properties

    Returns:
        Processed schema object with all properties resolved
    """
    result = {}

    # Copy basic attributes
    for key, value in schema_obj.items():
        if key != "properties" and key != "required" and key != "$defs":
            result[key] = value

    # Process properties if present
    if "properties" in schema_obj:
        result["properties"] = {}
        required_props = []

        # Get required fields list
        required_fields = schema_obj.get("required", [])

        for prop_name, prop_value in schema_obj["properties"].items():
            # Process each property
            is_required = prop_name in required_fields
            processed = _process_property(
                prop_value, defs, is_required, fully_expand
            )
            result["properties"][prop_name] = processed

            # Track which properties are actually required after processing
            if is_required and "null" not in str(processed.get("type", "")):
                required_props.append(prop_name)

        # Add required fields if any
        if required_props:
            result["required"] = required_props

    return result


def _process_nested_dict(
    d: Dict[str, Any], defs: Dict[str, Any]
) -> Dict[str, Any]:
    """Recursively processes nested dictionaries and resolves $ref references.

    Args:
        d: The dictionary to process
        defs: The definitions dictionary for resolving references

    Returns:
        Processed dictionary
    """
    result: Dict[str, Any] = {}

    # Handle direct reference
    if "$ref" in d:
        ref_path = d["$ref"].split("/")[-1]
        if ref_path in defs:
            ref_dict = defs[ref_path]
            # Recursively process the referenced object
            return _process_schema_object(ref_dict, defs)
        else:
            # Handle missing reference path gracefully
            raise ValueError(f"Missing reference: {ref_path}")

    # Process each key-value pair
    for key, value in d.items():
        if key == "$ref":
            # Already handled above
            continue
        elif isinstance(value, dict):
            result[key] = _process_nested_dict(value, defs)
        elif isinstance(value, list):
            # Process lists (like for enum values)
            result[key] = [
                (
                    _process_nested_dict(item, defs)
                    if isinstance(item, dict)
                    else item
                )
                for item in value
            ]
        else:
            result[key] = value

    return result


def _process_property(
    prop: Dict[str, Any],
    defs: Dict[str, Any],
    is_required: bool = False,
    fully_expand: bool = True,
) -> Dict[str, Any]:
    """Process a property in a schema, resolving any references.

    Args:
        prop: The property to process
        defs: The definitions dictionary for resolving references
        is_required: Whether this property is required
        fully_expand: Whether to fully expand nested properties

    Returns:
        Processed property
    """
    result = {}
    is_nullable = False

    # Handle anyOf for optional fields (like Optional[Type])
    if "anyOf" in prop:
        # Check if this is an Optional[...] case (one null, one type)
        null_type = False
        non_null_type = None

        for option in prop["anyOf"]:
            if option.get("type") == "null":
                null_type = True
                is_nullable = True
            elif "$ref" in option:
                ref_path = option["$ref"].split("/")[-1]
                if ref_path in defs:
                    non_null_type = _process_schema_object(
                        defs[ref_path], defs, fully_expand
                    )
                else:
                    # Handle missing reference path gracefully
                    raise ValueError(f"Missing reference: {ref_path}")
            else:
                non_null_type = option

        if null_type and non_null_type:
            # For Optional fields, we mark as nullable but copy all properties from the non-null option
            result = (
                non_null_type.copy() if isinstance(non_null_type, dict) else {}
            )

            # For type, ensure it includes "null"
            if "type" in result and isinstance(result["type"], str):
                result["type"] = [result["type"], "null"]
            elif (
                "type" in result
                and isinstance(result["type"], list)
                and "null" not in result["type"]
            ):
                result["type"].append("null")
            elif "type" not in result:
                # Default to object type if not specified
                result["type"] = ["object", "null"]

            # Copy description if available in the property
            if "description" in prop:
                result["description"] = prop["description"]

            # Need to process item refs as well (#337)
            if "items" in result:
                result["items"] = _process_property(result["items"], defs)

            return result

    # Handle direct references
    elif "$ref" in prop:
        # Resolve reference
        ref_path = prop["$ref"].split("/")[-1]
        if ref_path in defs:
            ref_dict = defs[ref_path]
            # Process the referenced object to get a complete schema
            result = _process_schema_object(ref_dict, defs, fully_expand)
        else:
            # Handle missing reference path gracefully
            raise ValueError(f"Missing reference: {ref_path}")

    # For regular fields, copy all properties
    for key, value in prop.items():
        if key not in ["$ref", "anyOf"]:
            if isinstance(value, dict):
                result[key] = _process_nested_dict(value, defs)
            elif key == "type" and not is_required and not is_nullable:
                # For non-required fields, ensure type is a list with "null"
                if isinstance(value, str):
                    result[key] = [value, "null"]
                elif isinstance(value, list) and "null" not in value:
                    result[key] = value + ["null"]
                else:
                    result[key] = value
            else:
                result[key] = value

    return result


def _flatten_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Flattens a JSON schema by removing $defs and resolving $ref references.

    Handles required vs optional fields properly.

    Args:
        schema: The JSON schema to flatten

    Returns:
        Flattened JSON schema
    """
    # Extract required fields list
    required_fields = schema.get("required", [])

    # Initialize the flattened schema with basic properties
    flattened = {
        "type": schema.get("type", "object"),
        "properties": {},
    }

    # Add title if present
    if "title" in schema:
        flattened["title"] = schema["title"]

    # Add description from schema if present, or use model docstring
    if "description" in schema and schema["description"]:
        flattened["description"] = schema["description"]

    # Process properties
    required_props: list[str] = []
    if "properties" in schema:
        required_props = []
        for prop_name, prop_value in schema["properties"].items():
            # Process the property and add to flattened properties
            is_required = prop_name in required_fields

            # If the property already has nested properties (expanded), preserve them
            if "properties" in prop_value:
                # This is an expanded nested schema, preserve its structure
                processed_prop = {
                    "type": prop_value.get("type", "object"),
                    "description": prop_value.get("description", ""),
                    "properties": {},
                }

                # Process each nested property
                for nested_prop_name, nested_prop_value in prop_value[
                    "properties"
                ].items():
                    is_required = (
                        "required" in prop_value
                        and nested_prop_name in prop_value["required"]
                    )
                    sub_property = _process_property(
                        nested_prop_value, schema.get("$defs", {}), is_required
                    )
                    processed_prop["properties"][
                        nested_prop_name
                    ] = sub_property

                # Copy required fields if present
                if "required" in prop_value:
                    processed_prop["required"] = prop_value["required"]
            else:
                # Process as normal
                processed_prop = _process_property(
                    prop_value, schema.get("$defs", {}), is_required
                )

            flattened["properties"][prop_name] = processed_prop

            # Track which properties are actually required after processing
            if is_required and "null" not in str(
                processed_prop.get("type", "")
            ):
                required_props.append(prop_name)

    # Add required fields if any (only those that are truly required after processing)
    # Check if required props are empty, if so, raise an error because it means there is a circular reference

    if len(required_props) > 0:
        flattened["required"] = required_props
    else:
        raise ValueError("Circular reference detected and not supported")

    return flattened


def _expand_nested_properties(
    schema: Dict[str, Any], model: Type[BaseModel]
) -> None:
    """Expand the properties of nested models in the schema to include their full structure.

    This updates the schema in place.

    Args:
        schema: The JSON schema to process
        model: The Pydantic model class
    """
    # First, process the properties at this level
    if "properties" not in schema:
        return

    # Create a modified copy of the properties to avoid modifying while iterating
    for prop_name, prop_info in list(schema["properties"].items()):
        field = model.model_fields.get(prop_name)
        if not field:
            continue

        field_type = field.annotation

        # Handle Optional types
        is_optional = False
        if (
            field_type is not None
            and hasattr(field_type, "__origin__")
            and field_type.__origin__ is Union
            and hasattr(field_type, "__args__")
        ):
            # Look for Optional[BaseModel]
            for arg in field_type.__args__:
                if arg is type(None):
                    is_optional = True
                elif isinstance(arg, type) and issubclass(arg, BaseModel):
                    field_type = arg

        # If this is a BaseModel field, expand its properties with full details
        if isinstance(field_type, type) and issubclass(field_type, BaseModel):
            # Get the nested model's schema with all its properties
            nested_model_schema = field_type.model_json_schema()

            # Create a properly expanded nested object
            expanded_object = {
                "type": ["object", "null"] if is_optional else "object",
                "description": prop_info.get(
                    "description", field.description or f"The {prop_name}"
                ),
                "properties": {},
            }

            # Copy all properties from the nested schema
            if "properties" in nested_model_schema:
                expanded_object["properties"] = nested_model_schema[
                    "properties"
                ]

            # Copy required fields
            if "required" in nested_model_schema:
                expanded_object["required"] = nested_model_schema["required"]

            # Replace the original property with this expanded version
            schema["properties"][prop_name] = expanded_object


def convert_pydantic_to_tool_spec(
    model: Type[BaseModel],
    description: Optional[str] = None,
) -> ToolSpec:
    """Converts a Pydantic model to a tool description for the Amazon Bedrock Converse API.

    Handles optional vs. required fields, resolves $refs, and uses docstrings.

    Args:
        model: The Pydantic model class to convert
        description: Optional description of the tool's purpose

    Returns:
        ToolSpec: Dict containing the Bedrock tool specification
    """
    name = model.__name__

    # Get the JSON schema
    input_schema = model.model_json_schema()

    # Get model docstring for description if not provided
    model_description = description
    if not model_description and model.__doc__:
        model_description = model.__doc__.strip()

    # Process all referenced models to ensure proper docstrings
    # This step is important for gathering descriptions from referenced models
    _process_referenced_models(input_schema, model)

    # Now, let's fully expand the nested models with all their properties
    _expand_nested_properties(input_schema, model)

    # Flatten the schema
    flattened_schema = _flatten_schema(input_schema)

    final_schema = flattened_schema

    # Construct the tool specification
    return ToolSpec(
        name=name,
        description=model_description or f"{name} structured output tool",
        inputSchema={"json": final_schema},
    )
