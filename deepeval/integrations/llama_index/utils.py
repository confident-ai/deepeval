def parse_id(id_: str) -> tuple[str, str]:
    """
    Parse the id_ into a tuple of class name and method name, ignoring any suffix after '-'.
    """
    # Ignore everything after the first '-'
    main_part = id_.split("-", 1)[0]
    # Split by '.' to get class and method
    class_name, method_name = main_part.rsplit(".", 1)
    return class_name, method_name