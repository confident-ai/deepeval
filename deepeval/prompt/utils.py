import re
from jinja2 import Template

from deepeval.prompt.api import PromptInterpolationType
from pydantic import BaseModel, create_model
from typing import Any, Dict, Type, Optional, List


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


def reconstruct_basemodel_from_schema(output_schema: Dict[str, Any], model_name: str = "DynamicModel") -> Type[BaseModel]:
    """
    Reconstruct a Pydantic BaseModel from a JSON schema.
    
    Args:
        output_schema: JSON schema dictionary
        model_name: Name for the dynamically created model
        
    Returns:
        Dynamically created Pydantic BaseModel class
    """
    if not output_schema or not isinstance(output_schema, dict):
        raise ValueError("output_schema must be a non-empty dictionary")
    
    # Handle different schema formats
    properties = output_schema.get("properties", {})
    required_fields = set(output_schema.get("required", []))
    
    # If no properties found, try to extract from other common schema formats
    if not properties and "type" in output_schema:
        if output_schema["type"] == "object":
            properties = output_schema.get("properties", {})
        else:
            # Simple type schema
            return create_model(model_name, value=(Any, ...))
    
    if not properties:
        raise ValueError("No properties found in schema")
    
    # Build field definitions for create_model
    field_definitions = {}
    
    for field_name, field_schema in properties.items():
        field_type = _schema_type_to_python_type(field_schema)
        
        # Determine if field is required
        if field_name in required_fields:
            field_definitions[field_name] = (field_type, ...)
        else:
            field_definitions[field_name] = (Optional[field_type], None)
    
    # Create the dynamic model
    return create_model(model_name, **field_definitions)


def _schema_type_to_python_type(field_schema: Dict[str, Any]) -> Type:
    """
    Convert JSON schema type to Python type.
    
    Args:
        field_schema: Field schema dictionary
        
    Returns:
        Python type
    """
    schema_type = field_schema.get("type", "string")
    
    type_mapping = {
        "string": str,
        "integer": int,
        "number": float,
        "boolean": bool,
        "array": List,
        "object": Dict,
        "null": type(None)
    }
    
    base_type = type_mapping.get(schema_type, str)
    
    # Handle array types
    if schema_type == "array":
        items_schema = field_schema.get("items", {})
        if items_schema:
            item_type = _schema_type_to_python_type(items_schema)
            return List[item_type]
        return List[Any]
    
    # Handle object types
    if schema_type == "object":
        properties = field_schema.get("properties", {})
        if properties:
            # Create nested model for complex objects
            nested_model = reconstruct_basemodel_from_schema(field_schema, "NestedModel")
            return nested_model
        return Dict[str, Any]
    
    # Handle enum types
    if "enum" in field_schema:
        enum_values = field_schema["enum"]
        if all(isinstance(v, str) for v in enum_values):
            return str
        elif all(isinstance(v, int) for v in enum_values):
            return int
        else:
            return Any
    
    return base_type
