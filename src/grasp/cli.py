import argparse
import json
import os
import random
import time

from fastapi import WebSocketDisconnect
from pydantic import BaseModel, conlist
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_jsonl, load_jsonl
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import extract_field, partition_by

from grasp.adapt import adapt
from grasp.configs import Adapt, Config
from grasp.core import generate, get_system_message, setup
from grasp.functions import get_functions
from grasp.utils import is_invalid_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="grasp",
        description="GRASP: Generic Reasoning and SPARQL generation across Knowledge Graphs",
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to the configuration file",
    )
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["sparql-qa", "general-qa"],
        default="sparql-qa",
        help="Task to perform, either 'sparql-qa' for SPARQL question answering or "
        "'general-qa' for general knowledge graph assisted question answering (ignored"
        " if --serve is used)",
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--serve",
        type=int,
        default=8000,
        help="Start a WebSocket server on this port",
    )
    input_group.add_argument(
        "-q",
        "--question",
        type=str,
        help="Question to answer",
    )
    input_group.add_argument(
        "-a",
        "--adapt",
        type=str,
        help="Adapt the GRASP system and save results in this directory",
    )
    input_group.add_argument(
        "-i",
        "--question-file",
        type=str,
        help="Input file with questions to run GRASP on",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="File to write the output to (only used with --question-file)",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output (used with --question-file and --adapt)",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the input (only used with --question-file)",
    )
    parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Number of samples to take from input (only used with --question-file)",
    )
    parser.add_argument(
        "--question-field",
        type=str,
        default="question",
        help="Field to extract as question from the input (only used with --question-file)",
    )
    parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed questions (only used with --question-file and --output-file)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=22,
        help="Seed for the random number generator",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Log level for the logger",
    )
    parser.add_argument(
        "--all-loggers",
        action="store_true",
        help="Enable logging for all loggers, not only the GRASP-specific ones",
    )
    return parser.parse_args()


def run_grasp(args: argparse.Namespace) -> None:
    logger = get_logger("GRASP", args.log_level)
    config = Config(**load_config(args.config))

    managers, example_indices, notes = setup(config)

    functions = get_functions(
        managers,
        args.task,
        config.fn_set,
        example_indices,
        config.num_examples,
        config.random_examples,
    )

    system_message = get_system_message(args.task, managers, notes)

    outputs = []
    if args.question_file is not None:
        random.seed(args.seed)
        assert args.output_file is not None, "Output file is required with --run"

        inputs = load_jsonl(args.question_file)
        if args.shuffle:
            random.shuffle(inputs)

        take = args.take or len(inputs)
        inputs = inputs[:take]

        exists = os.path.exists(args.output_file)
        if exists and not args.overwrite:
            outputs = load_jsonl(args.output_file)

        # save this in a config file
        output_stem, _ = os.path.splitext(args.output_file)
        config_file = output_stem + ".config.json"
        with open(config_file, "w") as f:
            json.dump(
                {
                    "config": config.model_dump(),
                    "functions": functions,
                    "system_message": system_message,
                },
                f,
                indent=2,
            )

    else:
        assert args.question is not None
        inputs = [{"id": 0, "question": args.question}]
        args.question_field = "question"

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id")
        question = extract_field(ipt, args.question_field)
        assert id is not None and question is not None, "id and question are required"

        if i < len(outputs):
            output = outputs[i]
            if not args.retry_failed or not is_invalid_model_output(output):
                continue

        *_, output = generate(
            args.task,
            question,
            config,
            managers,
            notes,
            functions,
            example_indices,
            logger=logger,
        )

        output["id"] = id
        if args.question:
            print(json.dumps(output))
            return

        if i < len(outputs):
            outputs[i] = output
        else:
            outputs.append(output)

        dump_jsonl(outputs, args.output_file)


# keep track of connections and limit to 10 concurrent connections
active_connections = 0
MAX_CONNECTIONS = 10
# maximum duration for a query in seconds
MAX_QUERY_DURATION = 300.0


class Past(BaseModel):
    questions: conlist(str, min_length=1)  # type: ignore
    messages: conlist(dict, min_length=1)  # type: ignore
    known: set[str]


class Request(BaseModel):
    task: str
    question: str
    knowledge_graphs: conlist(str, min_length=1)  # type: ignore
    past: Past | None = None


def serve_grasp(args: argparse.Namespace) -> None:
    # create a fast api websocket server to serve the generate_sparql function
    import uvicorn
    from fastapi import FastAPI, WebSocket

    app = FastAPI()
    logger = get_logger("GRASP SERVER", args.log_level)

    # add cors
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    config = Config(**load_config(args.config))
    managers, example_indices, notes = setup(config)
    kgs = [manager.kg for manager in managers]

    @app.get("/knowledge_graphs")
    async def _knowledge_graphs():
        return kgs

    @app.get("/config")
    async def _config():
        return config.model_dump()

    @app.websocket("/live")
    async def _live(websocket: WebSocket):
        global active_connections

        # Check if we've reached the maximum number of connections
        if active_connections >= MAX_CONNECTIONS:
            await websocket.close(code=1008)  # HTTP Status 503: Service Unavailable
            return

        await websocket.accept()
        active_connections += 1

        try:
            while True:
                data = await websocket.receive_json()
                try:
                    request = Request(**data)
                except Exception:
                    await websocket.send_json({"error": "Invalid request format"})
                    continue

                sel = request.knowledge_graphs
                if not sel or not all(kg in kgs for kg in sel):
                    await websocket.send_json(
                        {"error": "Unsupported knowledge graph selection"}
                    )
                    continue

                sel_managers, _ = partition_by(managers, lambda m: m.kg in sel)
                sel_example_indices = {
                    kg: example_indices[kg] for kg in sel if kg in example_indices
                }

                functions = get_functions(
                    sel_managers,
                    request.task,
                    config.fn_set,
                    sel_example_indices,
                    config.num_examples,
                    config.random_examples,
                )

                system_message = get_system_message(request.task, sel_managers, notes)
                past_questions = []
                past_messages = []
                known = set()
                if request.past is None:
                    past_messages.append(system_message)
                else:
                    # overwrite system message because new set of
                    # knowledge graphs might be present
                    past_messages = request.past.messages
                    past_messages[0] = system_message
                    # update questions
                    past_questions = request.past.questions
                    # update known set
                    known = request.past.known

                await websocket.send_json(
                    {
                        "typ": "system",
                        "functions": functions,
                        "system_message": system_message["content"],
                    }
                )
                await websocket.receive_json()

                # Setup generator
                generator = generate(
                    request.task,
                    request.question,
                    config,
                    sel_managers,
                    notes,
                    functions,
                    sel_example_indices,
                    past_questions,
                    past_messages,
                    known,
                    logger=logger,
                )

                # Track start time for timeout
                start_time = time.perf_counter()

                # Process generator outputs with timeout check
                for output in generator:
                    # Check if we've exceeded the time limit
                    current_time = time.perf_counter()
                    if current_time - start_time > MAX_QUERY_DURATION:
                        # Send timeout message to client
                        await websocket.send_json(
                            {
                                "error": f"Operation timed out after {MAX_QUERY_DURATION} seconds",
                            }
                        )
                        break

                    # Process the output normally
                    await websocket.send_json(output)
                    data = await websocket.receive_json()

                    # Check if client requested cancellation
                    if data.get("cancel", False):
                        # Send cancellation confirmation to client
                        await websocket.send_json({"cancelled": True})
                        break

        except WebSocketDisconnect:
            pass

        except Exception as e:
            await websocket.send_json({"error": f"Failed to handle request:\n{e}"})

        finally:
            active_connections -= 1

    uvicorn.run(app, host="0.0.0.0", port=args.serve)


def adapt_grasp(args: argparse.Namespace) -> None:
    if os.path.exists(args.adapt) and not args.overwrite:
        raise FileExistsError(
            f"Output directory {args.adapt} already exists. Use --overwrite to overwrite."
        )
    config = Adapt(**load_config(args.config))
    adapt(args.task, config, args.adapt, args.log_level)


def main():
    args = parse_args()
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.adapt is not None:
        adapt_grasp(args)
    elif args.question_file is not None or args.question is not None:
        run_grasp(args)
    else:
        serve_grasp(args)


if __name__ == "__main__":
    main()
