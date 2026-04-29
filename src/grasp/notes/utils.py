from typing import Any, Iterator

from grasp.model import Message, Response
from grasp.utils import format_list


def format_arguments(args, depth: int = 0) -> str:
    if isinstance(args, list):
        return "[" + ", ".join(format_arguments(i, depth + 1) for i in args) + "]"
    elif isinstance(args, dict):
        return (
            "{" * (depth > 0)
            + ", ".join(
                f"{k}={format_arguments(v, depth + 1)}" for k, v in args.items()
            )
            + "}" * (depth > 0)
        )
    elif isinstance(args, str):
        return f'"{args}"'
    else:
        return str(args)


def format_output(output: Any | None, messages: list[Message]) -> str:
    fmt = []
    step = 1
    for message in messages[2:]:
        if message.role == "feedback":
            fmt.append(f"Feedback:\n{message.content}")
            continue

        elif message.role == "user":
            fmt.append(f"User:\n{message.content}")
            continue

        assert isinstance(message.content, Response)

        assistant = message.content
        ass_content = assistant.get_content()

        contents = []
        if "reasoning" in ass_content:
            contents.append(f"Agent reasoning:\n{ass_content['reasoning']}")

        if "content" in ass_content:
            contents.append(f"Agent message:\n{ass_content['content']}")

        for tool_call in assistant.tool_calls:
            tool_call_content = f'Call of "{tool_call.name}" function'
            if tool_call.args:
                tool_call_content += f" with {format_arguments(tool_call.args)}"

            tool_call_content += f":\n{tool_call.result}"
            contents.append(tool_call_content)

        content = f"Step {step}:\n{format_list(contents)}"
        fmt.append(content)
        step += 1

    if output is not None and "formatted" in output:
        fmt.append(f"Output after {step} steps:\n{output['formatted']}")

    return "\n\n".join(fmt)


def consume_iterator(iterator: Iterator) -> None:
    for _ in iterator:
        pass
