from copy import deepcopy

from universal_ml_utils.ops import partition_by

from grasp.sparql.manager import KgManager
from grasp.utils import FunctionCallException, format_enumerate

MAX_NOTES = 16
MAX_NOTE_LENGTH = 256


def get_note_functions(managers: list[KgManager]) -> list[dict]:
    kgs = [manager.kg for manager in managers]
    return [
        {
            "name": "add_note",
            "description": "Add a note to a specific knowledge graph or in general.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph for which to add the note, omit for general notes",
                    },
                    "note": {
                        "type": "string",
                        "description": "The note to add to the knowledge graph",
                    },
                },
                "required": ["note"],
                "additionalProperties": False,
            },
        },
        {
            "name": "delete_note",
            "description": "Delete a note for a specific knowledge graph or in general.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph for which to delete the note, omit for general notes",
                    },
                    "num": {
                        "type": "number",
                        "description": "The number of the note to delete",
                    },
                },
                "required": ["num"],
                "additionalProperties": False,
            },
        },
        {
            "name": "update_note",
            "description": "Update a note for a specific knowledge graph or in general.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kg": {
                        "type": "string",
                        "enum": kgs,
                        "description": "The knowledge graph for which to update the note, omit for general notes",
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
                "required": ["num", "note"],
                "additionalProperties": False,
            },
        },
        {
            "name": "stop",
            "description": "Stop the process.",
        },
    ]


def format_general_notes(notes: list[str]) -> str:
    if not notes:
        return "No general notes for all knowledge graphs available"

    return f"General notes for all knowledge graphs:\n{format_enumerate(notes)}"


def check_note(note: str) -> None:
    if len(note) > MAX_NOTE_LENGTH:
        raise FunctionCallException(
            f"Note exceeds maximum length of {MAX_NOTE_LENGTH} characters"
        )


def add_note_to_kg(manager: KgManager, note: str) -> str:
    if len(manager.notes) >= MAX_NOTES:
        raise FunctionCallException(f"Cannot add more than {MAX_NOTES} notes")

    check_note(note)

    manager.notes.append(note)
    return f"Added note for {manager.kg}:\n{format_enumerate(manager.notes)}"


def delete_note_from_kg(manager: KgManager, num: int | float) -> str:
    num = int(num)
    if num < 1 or num > len(manager.notes):
        raise FunctionCallException("Note number out of range")

    num -= 1
    _ = manager.notes.pop(num)
    return f"Deleted note for {manager.kg}:\n{format_enumerate(manager.notes)}"


def update_note_of_kg(manager: KgManager, num: int | float, note: str) -> str:
    num = int(num)
    if num < 1 or num > len(manager.notes):
        raise FunctionCallException("Note number out of range")

    check_note(note)

    num -= 1
    manager.notes[num] = note
    return f"Updated note for {manager.kg}:\n{format_enumerate(manager.notes)}"


def add_note(notes: list[str], note: str) -> str:
    if len(notes) >= MAX_NOTES:
        raise FunctionCallException(f"Cannot add more than {MAX_NOTES} notes")

    check_note(note)

    notes.append(note)
    return f"Added general note:\n{format_enumerate(notes)}"


def delete_note(notes: list[str], num: int | float) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException("Note number out of range")

    num -= 1
    _ = notes.pop(num)
    return f"Deleted general note:\n{format_enumerate(notes)}"


def update_note(notes: list[str], num: int | float, note: str) -> str:
    num = int(num)
    if num < 1 or num > len(notes):
        raise FunctionCallException("Note number out of range")

    check_note(note)

    num -= 1
    notes[num] = note
    return f"Updated general note:\n{format_enumerate(notes)}"


def call_function(
    managers: list[KgManager],
    notes: list[str],
    fn_name: str,
    fn_args: dict,
) -> str:
    if fn_name == "stop":
        return "Stopped process"

    # kg should be there for every function call
    fn_args = deepcopy(fn_args)
    kg = fn_args.pop("kg", None)
    if kg is None:
        if fn_name == "add_note":
            return add_note(notes, fn_args["note"])
        elif fn_name == "delete_note":
            return delete_note(notes, fn_args["num"])
        elif fn_name == "update_note":
            return update_note(notes, fn_args["num"], fn_args["note"])
        else:
            raise ValueError(f"Unknown function {fn_name}")

    managers, others = partition_by(managers, lambda m: m.kg == kg)

    if len(managers) != 1:
        kgs = "\n".join(manager.kg for manager in managers + others)
        return f"Unknown knowledge graph {kg}, expected one of:\n{kgs}"

    manager = managers[0]
    if fn_name == "add_note":
        return add_note_to_kg(manager, fn_args["note"])
    elif fn_name == "delete_note":
        return delete_note_from_kg(manager, fn_args["num"])
    elif fn_name == "update_note":
        return update_note_of_kg(manager, fn_args["num"], fn_args["note"])
    else:
        raise ValueError(f"Unknown function {fn_name}")
