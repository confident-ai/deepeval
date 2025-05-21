def to_hex_string(id_value: int | bytes, length: int = 32) -> str:
    """
    Convert a trace ID or span ID to a hex string.

    Args:
        id_value: The ID value to convert, either as an integer or bytes
        length: The expected length of the hex string (32 for trace IDs, 16 for span IDs)

    Returns:
        A hex string representation of the ID
    """
    if isinstance(id_value, int):
        return format(id_value, f"0{length}x")
    return id_value.hex()
