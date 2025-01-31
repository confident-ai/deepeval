from typing import Optional, Dict, Union, List
from deepeval.monitor.api import Link, CustomProperty, CustomPropertyType


def process_additional_data(
    additional_data: Optional[
        Dict[str, Union[str, Link, List[Link], Dict]]
    ] = None,
):
    custom_properties = None
    if additional_data:
        custom_properties = {}
        for key, value in additional_data.items():
            if isinstance(value, str):
                custom_properties[key] = CustomProperty(
                    value=value, type=CustomPropertyType.TEXT
                )
            elif isinstance(value, dict):
                custom_properties[key] = CustomProperty(
                    value=value, type=CustomPropertyType.JSON
                )
            elif isinstance(value, Link):
                custom_properties[key] = CustomProperty(
                    value=value.value, type=CustomPropertyType.LINK
                )
            elif isinstance(value, list):
                if not all(isinstance(item, Link) for item in value):
                    raise ValueError(
                        "All values in 'additional_data' must be either of type 'string', 'Link', list of 'Link', or 'dict'."
                    )
                custom_properties[key] = [
                    CustomProperty(
                        value=item.value, type=CustomPropertyType.LINK
                    )
                    for item in value
                ]
            else:
                raise ValueError(
                    "All values in 'additional_data' must be either of type 'string', 'Link', list of 'Link', or 'dict'."
                )

    return custom_properties
