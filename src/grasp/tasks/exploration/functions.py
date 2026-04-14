from grasp.manager import KgManager
from grasp.utils import FunctionCallException, format_notes


def note_functions(managers: list[KgManager]) -> list[dict]:
    kgs: list[str | None] = [manager.kg for manager in managers]
    kgs.append(None)
    return [
        {
            "name": "add_note",
            "description": "Add a general or knowledge graph specific note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to add the note (null for general notes)",
                    },
                    "note": {
                        "type": "string",
                        "description": "The note to add",
                    },
                },
                "required": ["kg", "note"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "delete_note",
            "description": "Delete a general or knowledge graph specific note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to delete the note (null for general notes)",
                    },
                    "num": {
                        "type": "number",
                        "description": "The number of the note to delete",
                    },
                },
                "required": ["kg", "num"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "update_note",
            "description": "Update a general or knowledge graph specific note.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to update the note (null for general notes)",
                    },
                    "num": {
                        "type": "number",
                        "description": "The number of the note to update",
                    },
                    "note": {
                        "type": "string",
                        "description": "The new note replacing the old one",
                    },
                },
                "required": ["kg", "num", "note"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "show_notes",
            "description": "Show current general or knowledge graph specific notes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": ["string", "null"],
                        "enum": kgs,
                        "description": "The knowledge graph for which to show the notes (null for general notes)",
                    },
                },
                "required": ["kg"],
                "additionalProperties": False,
            },
            "strict": True,
        },
        {
            "name": "stop",
            "description": "Stop the note taking process.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
                "additionalProperties": False,
            },
            "strict": True,
        },
    ]


def check_note(note: str, max_note_length: int) -> None:
    if len(note) > max_note_length:
        raise FunctionCallException(
            f"Note exceeds maximum length of {max_note_length} characters"
        )


def show_notes(notes: list[str]) -> str:
    return format_notes(notes, enumerated=True)


def add_note(notes: list[str], note: str, max_notes: int, max_note_length: int) -> str:
    if len(notes) >= max_notes:
        raise FunctionCallException(f"Cannot add more than {max_notes} notes")

    check_note(note, max_note_length)

    notes.append(note)
    return f"Added note {len(notes)}: {notes[-1]}"


def delete_note(notes: list[str], num: int | float) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException("Note number out of range")

    num -= 1
    note = notes.pop(num)
    return f"Deleted note {num + 1}: {note}"


def update_note(
    notes: list[str],
    num: int | float,
    note: str,
    max_note_length: int,
) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException("Note number out of range")

    check_note(note, max_note_length)

    num -= 1
    notes[num] = note
    return f"Updated note {num + 1}: {note}"


def call_function(
    kg_notes: dict[str, list[str]],
    notes: list[str],
    fn_name: str,
    fn_args: dict,
    max_notes: int,
    max_note_length: int,
) -> str:
    if fn_name == "stop":
        return "Stopped process"

    # kg should be there for every function call
    kg = fn_args.get("kg", None)
    if kg is None:
        notes_to_use = notes
    else:
        if kg not in kg_notes:
            kg_notes[kg] = []
        notes_to_use = kg_notes[kg]

    if fn_name == "add_note":
        return add_note(notes_to_use, fn_args["note"], max_notes, max_note_length)
    elif fn_name == "delete_note":
        return delete_note(notes_to_use, fn_args["num"])
    elif fn_name == "update_note":
        return update_note(
            notes_to_use,
            fn_args["num"],
            fn_args["note"],
            max_note_length,
        )
    elif fn_name == "show_notes":
        return show_notes(notes_to_use)
    else:
        raise ValueError(f"Unknown function {fn_name}")
