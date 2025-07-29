from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Testing MCP")

file = "notes.txt"

@mcp.tool()
def get_notes() -> str:
    """Gets all the notes"""
    with open(file) as note:
        notes = note.read().strip()
    return notes

@mcp.tool()
def write_notes(note: str) -> bool:
    """Writes a note into my existing notes, returns true if success and false otherwise
    """
    try:
        with open(file, "a") as notes:
            notes.write(note + "\n")
            return True
    except:
        return False
    
@mcp.resource(uri="file:///project/src/notes.txt")
def getNotes() -> str:
    """A resource on whatever is inside the notes"""
    with open(file) as note:
        notes = note.read().strip()
    return notes

@mcp.prompt()
def addingNote() -> list:
    """Has a template of how to add prompts whenever they're added"""
    content = getNotes()
    return content

# @mcp.tool()
# def get_notes_length() -> int:
#     """Gets the length of all available notes"""
#     return notes.get_len()

if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport='stdio')