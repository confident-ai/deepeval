def set_attr_path(obj, attr_path: str, value):
    *pre_path, final_attr = attr_path.split(".")
    for attr in pre_path:
        obj = getattr(obj, attr, None)
        if obj is None:
            return
    setattr(obj, final_attr, value)


def get_attr_path(obj, attr_path: str):
    for attr in attr_path.split("."):
        obj = getattr(obj, attr, None)
        if obj is None:
            return None
    return obj

