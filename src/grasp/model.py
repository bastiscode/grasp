import json
from typing import Any, Callable
from uuid import uuid4

from litellm import (
    Choices,
    ResponseFunctionToolCall,
    ResponsesAPIResponse,
    completion,
    responses,
)
from litellm.types.responses.main import (
    GenericResponseOutputItem,
    OutputFunctionToolCall,
)
from litellm.types.utils import ModelResponse
from openai.types.responses.response_output_message import ResponseOutputMessage
from openai.types.responses.response_reasoning_item import ResponseReasoningItem
from pydantic import BaseModel

from grasp.configs import ModelConfig


class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any]
    result: str | None = None
    error: str | None = None


class Reasoning(BaseModel):
    id: str
    content: str | None = None
    summary: str | None = None
    encrypted_content: str | None = None


def strip_none(s: str | None) -> str | None:
    if s is None:
        return None

    s = s.strip()
    if not s:
        return None

    return s


class Response(BaseModel):
    id: str
    message: str | None
    reasoning: Reasoning | None = None
    tool_calls: list[ToolCall]
    usage: dict | None = None
    prompt_token_ids: list[int] | None = None
    token_ids: list[int] | None = None
    token_logprobs: list[float] | None = None
    token_texts: list[str] | None = None

    @property
    def is_empty(self) -> bool:
        return self.message is None and self.reasoning is None and not self.tool_calls

    @property
    def has_content(self) -> bool:
        return self.message is not None or self.has_reasoning_content

    @property
    def has_reasoning_content(self) -> bool:
        return self.reasoning_content is not None

    @property
    def reasoning_content(self) -> str | None:
        if self.reasoning is None:
            return None
        return self.reasoning.content or self.reasoning.summary

    def get_content(self) -> dict[str, str]:
        content = {}

        if self.has_reasoning_content:
            reasoning: str = self.reasoning.content or self.reasoning.summary  # type: ignore
            content["reasoning"] = reasoning

        if self.message is not None:
            content["content"] = self.message

        return content

    @staticmethod
    def from_completions_api(response: ModelResponse) -> "Response":
        id = uuid4().hex
        if not response.choices:
            return Response(id=id, message=None, tool_calls=[])

        choice: Choices = response.choices[0]  # type: ignore
        if choice.finish_reason not in ["tool_calls", "stop", "length"]:
            raise ValueError(f"Unexpected finish reason {choice.finish_reason}")

        message = strip_none(choice.message.content)
        reasoning = None
        if hasattr(choice.message, "reasoning_content"):
            reasoning = Reasoning(
                id=uuid4().hex,
                content=strip_none(choice.message.reasoning_content),
            )

        tool_calls = []
        for tool_call in choice.message.tool_calls or []:
            if tool_call.type != "function":
                continue

            assert tool_call.function.name is not None

            tool_calls.append(
                ToolCall(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    args=json.loads(tool_call.function.arguments),
                )
            )

        # Extract logprobs from choice.logprobs.content (standard OpenAI format)
        token_logprobs = None
        token_texts = None
        logprobs = getattr(choice, "logprobs", None)
        if logprobs and getattr(logprobs, "content", None):
            token_logprobs = [entry.logprob for entry in logprobs.content]
            token_texts = [entry.token for entry in logprobs.content]

        # Extract token_ids and prompt_token_ids from provider_specific_fields (vLLM extension)
        token_ids = None
        prompt_token_ids = None
        if (
            hasattr(choice, "provider_specific_fields")
            and choice.provider_specific_fields
        ):
            token_ids = choice.provider_specific_fields.get("token_ids")
            prompt_token_ids = choice.provider_specific_fields.get("prompt_token_ids")
        # Also check response-level attributes
        if token_ids is None and hasattr(response, "token_ids"):
            token_ids = response.token_ids  # type: ignore
        if prompt_token_ids is None and hasattr(response, "prompt_token_ids"):
            prompt_token_ids = response.prompt_token_ids  # type: ignore

        return Response(
            id=id,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(exclude_defaults=True),  # type: ignore
            prompt_token_ids=prompt_token_ids,
            token_ids=token_ids,
            token_logprobs=token_logprobs,
            token_texts=token_texts,
        )

    @staticmethod
    def from_responses_api(response: ResponsesAPIResponse) -> "Response":
        id = None
        message = None
        reasoning = None
        tool_calls = []

        for output in response.output:
            if isinstance(output, (GenericResponseOutputItem, OutputFunctionToolCall)):
                # vLLM responses API
                raise NotImplementedError(
                    "You are most likely using the vLLM respones API, "
                    "which is not supported yet, switch to completions API instead."
                )

            elif isinstance(output, ResponseOutputMessage):
                id = output.id

                # assistant
                if output.content:
                    message = strip_none(output.content[0].text)  # type: ignore

            elif isinstance(output, ResponseReasoningItem):
                # reasoning
                reasoning = Reasoning(id=output.id)
                if output.summary:
                    reasoning.summary = strip_none(output.summary[0].text)

                if output.content:
                    reasoning.content = strip_none(output.content[0].text)

                if output.encrypted_content is not None:
                    reasoning.encrypted_content = output.encrypted_content

            elif isinstance(output, ResponseFunctionToolCall):
                # tool call
                tool_calls.append(
                    ToolCall(
                        id=output.call_id,
                        name=output.name,
                        args=json.loads(output.arguments),
                    )
                )

            else:
                raise ValueError(f"Unknown output type {type(output)}")

        return Response(
            id=id or uuid4().hex,
            message=message,
            reasoning=reasoning,
            tool_calls=tool_calls,
            usage=response.usage.model_dump(exclude_defaults=True),  # type: ignore
        )

    def hash(self) -> str:
        msg: dict[str, Any] = {
            "msg": self.message,
            "reasoning": self.reasoning.model_dump(exclude={"id"})
            if self.reasoning
            else None,
            "tool_calls": sorted(
                (tc.name, json.dumps(tc.args, sort_keys=True)) for tc in self.tool_calls
            ),
        }
        return json.dumps(msg, sort_keys=True)


class Message(BaseModel):
    role: str
    content: str | Response


ModelFn = Callable[[list[Message], list[dict], ModelConfig], Response]


def completions_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    msgs = []
    for message in messages:
        if isinstance(message.content, str):
            # feedback is treated as coming from user
            role = message.role if message.role != "feedback" else "user"
            msgs.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )
            continue

        # response content
        assistant = message.content
        tool_calls = []
        tool_call_results = []
        for tool_call in assistant.tool_calls:
            tool_calls.append(
                {
                    "id": tool_call.id,
                    "type": "function",
                    "function": {
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.args),
                    },
                }
            )
            assert tool_call.result is not None, "Expected tool call result"
            tool_call_results.append(
                {
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": tool_call.result,
                }
            )

        msg = {
            "role": message.role,
            "content": assistant.message,
        }
        if assistant.reasoning is not None:
            msg["reasoning_content"] = assistant.reasoning.content

        if tool_calls:
            msg["tool_calls"] = tool_calls

        msgs.append(msg)

        if tool_call_results:
            msgs.extend(tool_call_results)

    return msgs


def responses_api_messages(messages: list[Message]) -> list[dict[str, Any]]:
    msgs = []

    for message in messages:
        if isinstance(message.content, str):
            # feedback is treated as coming from user
            role = message.role if message.role != "feedback" else "user"
            msgs.append(
                {
                    "type": "message",
                    "role": role,
                    "content": message.content,
                }
            )
            continue

        # response content
        assistant = message.content

        reasoning = assistant.reasoning
        if reasoning is not None:
            msgs.append(
                {
                    "id": reasoning.id,
                    "type": "reasoning",
                    "content": [{"text": reasoning.content, "type": "reasoning_text"}]
                    if reasoning.content is not None
                    else [],
                    "summary": [{"text": reasoning.summary, "type": "summary_text"}]
                    if reasoning.summary is not None
                    else [],
                    "encrypted_content": reasoning.encrypted_content,
                }
            )

        if assistant.message is not None:
            msgs.append(
                {
                    "id": assistant.id,
                    "type": "message",
                    "role": message.role,
                    "content": assistant.message,
                }
            )

        for tool_call in assistant.tool_calls:
            msgs.append(
                {
                    "type": "custom_tool_call",
                    "call_id": tool_call.id,
                    "name": tool_call.name,
                    "input": json.dumps(tool_call.args),
                }
            )
            assert tool_call.result is not None, "Expected tool call result"
            msgs.append(
                {
                    "type": "custom_tool_call_output",
                    "call_id": tool_call.id,
                    "output": tool_call.result,
                }
            )

    return msgs


def call_model(
    messages: list[Message],
    functions: list[dict],
    config: ModelConfig,
    custom_model: ModelFn | None = None,
) -> Response:
    if custom_model is not None:
        return custom_model(messages, functions, config)

    if config.api is None:
        api = "responses" if config.model.startswith("openai") else "completions"
    else:
        api = config.api

    if api == "completions":
        # Auto-enable logprobs when return_token_ids is requested
        request_logprobs = (
            config.model_kwargs.get("return_token_ids", False)
            if config.model_kwargs
            else False
        )

        # use old chat completions API
        completions_resp: ModelResponse = completion(
            model=config.model,
            messages=completions_api_messages(messages),
            tools=[{"type": "function", "function": fn} for fn in functions],
            tool_choice=config.tool_choice,
            parallel_tool_calls=config.parallel_tool_calls,
            # decoding parameters
            temperature=config.temperature,
            top_p=config.top_p,
            reasoning_effort=config.reasoning_effort,  # type: ignore
            # should be set to more than enough until the next function call
            max_completion_tokens=config.max_completion_tokens,
            base_url=config.model_endpoint,
            timeout=config.completion_timeout,
            seed=config.seed,
            extra_body=config.model_kwargs,
            # logprobs (auto-enabled with return_token_ids for vLLM)
            logprobs=True if request_logprobs else None,
            # drop unsupported parameters
            drop_params=True,
            num_retries=config.num_retries,
        )
        return Response.from_completions_api(completions_resp)

    elif api == "responses":
        # use responses API
        responses_resp: ResponsesAPIResponse = responses(
            model=config.model,
            input=responses_api_messages(messages),  # type: ignore
            include=["reasoning.encrypted_content"],
            tools=[{"type": "function", **fn} for fn in functions],  # type: ignore
            tool_choice=config.tool_choice,  # type: ignore
            parallel_tool_calls=config.parallel_tool_calls,
            # decoding parameters
            temperature=config.temperature,
            top_p=config.top_p,
            reasoning={
                "effort": config.reasoning_effort,
                "summary": config.reasoning_summary,
            },  # type: ignore
            truncation="auto",
            # should be set to more than enough until the next function call
            max_output_tokens=config.max_completion_tokens,
            base_url=config.model_endpoint,
            timeout=config.completion_timeout,
            seed=config.seed,
            extra_body=config.model_kwargs,
            # drop unsupported parameters
            drop_params=True,
            store=False,
            num_retries=config.num_retries,
        )
        return Response.from_responses_api(responses_resp)

    else:
        raise ValueError(f"Unknown API {api}")
