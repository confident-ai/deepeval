from enum import Enum


class PathTraversalType(Enum):
    """
    Enum for Path Traversal vulnerability types.

    - Relative path traversal: ../ sequences that climb out of the intended directory.
    - Absolute path traversal: absolute paths to sensitive files outside the intended scope.
    - Encoded path traversal: URL-, unicode-, or double-encoded traversal that evades naive filters.
    """

    RELATIVE_PATH_TRAVERSAL = "relative_path_traversal"
    ABSOLUTE_PATH_TRAVERSAL = "absolute_path_traversal"
    ENCODED_PATH_TRAVERSAL = "encoded_path_traversal"


# List of all available types for easy access
PATH_TRAVERSAL_TYPES = [
    PathTraversalType.RELATIVE_PATH_TRAVERSAL,
    PathTraversalType.ABSOLUTE_PATH_TRAVERSAL,
    PathTraversalType.ENCODED_PATH_TRAVERSAL,
]
