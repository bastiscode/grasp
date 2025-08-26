import argparse
import json
import os
import random
import time

from fastapi import WebSocketDisconnect
from pydantic import BaseModel, conlist
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_json, dump_jsonl, load_jsonl, load_text
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import extract_field, partition_by

from grasp.adapt import adapt
from grasp.build import build_indices, get_data
from grasp.configs import Adapt, Config
from grasp.core import generate, setup
from grasp.evaluate import evaluate
from grasp.examples import ExampleIndex
from grasp.utils import is_invalid_model_output, parse_parameters


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "config",
        type=str,
        help="Path to the GRASP configuration file",
    )


def add_task_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "-t",
        "--task",
        type=str,
        choices=["sparql-qa", "general-qa"],
        default="sparql-qa",
        help="Task to run",
    )


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="grasp",
        description="GRASP: Generic Reasoning and SPARQL generation across Knowledge Graphs",
    )

    subparsers = parser.add_subparsers(
        title="commands",
        description="Available commands",
        dest="command",
        required=True,
    )

    # run GRASP server
    server_parser = subparsers.add_parser(
        "serve",
        help="Start a WebSocket server to serve GRASP",
    )
    add_config_arg(server_parser)
    server_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the GRASP server on",
    )

    # run GRASP on a single question
    question_parser = subparsers.add_parser(
        "run",
        help="Run GRASP on a single question",
    )
    add_config_arg(question_parser)
    question_parser.add_argument(
        "question",
        type=str,
        help="Question to answer",
    )
    add_task_arg(question_parser)

    # run GRASP on file
    file_parser = subparsers.add_parser(
        "file",
        help="Run GRASP on a file with questions",
    )
    add_config_arg(file_parser)
    file_parser.add_argument(
        "question_file",
        type=str,
        help="Path to file in JSONL format to run GRASP on",
    )
    file_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the questions",
    )
    file_parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Limit number of questions",
    )
    file_parser.add_argument(
        "--question-field",
        type=str,
        default="question",
        help="Field to extract as question",
    )
    file_parser.add_argument(
        "--output-file",
        type=str,
        help="File to write the output to",
    )
    file_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed questions (only used with --output-file)",
    )
    add_task_arg(file_parser)
    add_overwrite_arg(file_parser)

    # evaluate GRASP output
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate GRASP output against a reference file",
    )
    eval_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with input questions in JSONL format",
    )
    eval_parser.add_argument(
        "prediction_file",
        type=str,
        help="Path to file with GRASP predictions as produced by the 'file' command",
    )
    eval_parser.add_argument(
        "endpoint",
        type=str,
        help="SPARQL endpoint to use for evaluation",
    )
    eval_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum duration for a single query in seconds",
    )
    eval_parser.add_argument(
        "--exact-after",
        type=int,
        default=1024,
        help="Result size after which exact F1 score instead of assignment F1 score "
        "is used (due to performance reasons)",
    )
    eval_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun failed evaluations due to timeouts or errors",
    )
    add_overwrite_arg(eval_parser)

    # run GRASP adaptation
    adapt_parser = subparsers.add_parser(
        "adapt",
        help="Adapt GRASP to one or more knowledge graphs",
    )
    add_config_arg(adapt_parser)
    adapt_parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        required=True,
        help="Save results in this directory",
    )
    add_task_arg(adapt_parser)
    add_overwrite_arg(adapt_parser)

    # get data for GRASP indices
    data_parser = subparsers.add_parser(
        "data",
        help="Get entity and property data for a knowledge graph",
    )
    add_config_arg(data_parser)
    data_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to get data for (must be defined in the config)",
    )
    data_parser.add_argument(
        "--entity-sparql",
        type=str,
        help="Path to file with custom entity SPARQL query",
    )
    data_parser.add_argument(
        "--property-sparql",
        type=str,
        help="Path to file with custom property SPARQL query",
    )
    data_parser.add_argument(
        "--parameters",
        type=str,
        nargs="*",
        help="Extra query parameters sent to the knowledge graph endpoint",
    )
    add_overwrite_arg(data_parser)

    # build GRASP indices
    index_parser = subparsers.add_parser(
        "index",
        help="Build entity and property indices for a knowledge graph",
    )
    add_config_arg(index_parser)
    index_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to build indices for (must be defined in the config)",
    )
    index_parser.add_argument(
        "--sim-precision",
        type=str,
        choices=["float32", "ubinary"],
        help="Precision when building similarity index",
    )
    index_parser.add_argument(
        "--sim-embedding-dim",
        type=int,
        help="Embedding dimensionality when building similarity index",
    )
    index_parser.add_argument(
        "--sim-batch-size",
        type=int,
        default=256,
        help="Batch size when building similarity index",
    )
    add_overwrite_arg(index_parser)

    # build example index
    example_parser = subparsers.add_parser(
        "examples",
        help="Build an example index used for few-shot learning",
    )
    example_parser.add_argument(
        "examples_file",
        type=str,
        help="Path to file with examples in JSONL format",
    )
    example_parser.add_argument(
        "output_dir",
        type=str,
        help="Directory to save the example index",
    )
    example_parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for building the example index",
    )
    add_overwrite_arg(example_parser)

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="GRASP log level",
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

    managers, notes = setup(config)

    run_on_file = args.command == "file"
    outputs = []
    if run_on_file:
        inputs = load_jsonl(args.question_file)
        question_field = args.question_field
        if args.shuffle:
            assert config.seed is not None, (
                "Seed must be set for deterministic shuffling"
            )
            random.seed(config.seed)
            random.shuffle(inputs)

        take = args.take or len(inputs)
        inputs = inputs[:take]

        if args.output_file:
            if os.path.exists(args.output_file) and not args.overwrite:
                outputs = load_jsonl(args.output_file)

            # save info in config file next to output file
            output_stem, _ = os.path.splitext(args.output_file)
            config_file = output_stem + ".config.json"

            dump_json(config.model_dump(), config_file, indent=2)

    else:
        inputs = [{"id": 0, "question": args.question}]
        question_field = "question"

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id")
        question = extract_field(ipt, question_field)
        assert id is not None and question is not None, "id and question are required"

        if i < len(outputs) and (
            not args.retry_failed or not is_invalid_model_output(outputs[i])
        ):
            continue

        *_, output = generate(
            args.task,
            question,
            config,
            managers,
            notes,
            logger=logger,
        )

        output["id"] = id
        if not run_on_file:
            print(json.dumps(output))
            break

        elif args.output_file is None:
            print(json.dumps(output))
            continue

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
    config = Config(**load_config(args.config))

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

    managers, notes = setup(config)
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

                past_questions = None
                past_messages = None
                past_known = None
                if request.past is not None:
                    # set past
                    past_messages = request.past.messages
                    past_questions = request.past.questions
                    past_known = request.past.known

                # Setup generator
                generator = generate(
                    request.task,
                    request.question,
                    config,
                    sel_managers,
                    notes,
                    past_questions,
                    past_messages,
                    past_known,
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

    uvicorn.run(app, host="0.0.0.0", port=args.port)


def adapt_grasp(args: argparse.Namespace) -> None:
    config = Adapt(**load_config(args.config))
    adapt(args.task, config, args.adapt, args.overwrite, args.log_level)


def get_grasp_data(args: argparse.Namespace) -> None:
    config = Config(**load_config(args.config))

    cfg = next(
        (c for c in config.knowledge_graphs if c.kg == args.knowledge_graph),
        None,
    )
    assert cfg is not None, (
        f"Knowledge graph {args.knowledge_graph} not found in config"
    )

    params = parse_parameters(args.parameters or [])

    if args.entity_sparql is not None:
        args.entity_sparql = load_text(args.entity_sparql).strip()

    if args.property_sparql is not None:
        args.property_sparql = load_text(args.property_sparql).strip()

    get_data(
        cfg,
        args.entity_sparql,
        args.property_sparql,
        params,
        args.overwrite,
        args.log_level,
    )


def build_grasp_index(args: argparse.Namespace) -> None:
    config = Config(**load_config(args.config))

    cfg = next(
        (c for c in config.knowledge_graphs if c.kg == args.knowledge_graph),
        None,
    )
    assert cfg is not None, (
        f"Knowledge graph {args.knowledge_graph} not found in config"
    )

    build_indices(
        cfg,
        args.overwrite,
        args.log_level,
        sim_batch_size=args.sim_batch_size,
        sim_precision=args.sim_precision,
        sim_embedding_dim=args.sim_embedding_dim,
    )


def evaluate_grasp(args: argparse.Namespace) -> None:
    evaluate(
        args.input_file,
        args.prediction_file,
        args.endpoint,
        args.overwrite,
        args.log_level,
        args.timeout,
        args.retry_failed,
        args.exact_after,
    )


def main():
    args = parse_args()
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.command == "data":
        get_grasp_data(args)
    elif args.command == "index":
        build_grasp_index(args)
    elif args.command == "adapt":
        adapt_grasp(args)
    elif args.command == "run" or args.command == "file":
        run_grasp(args)
    elif args.command == "serve":
        serve_grasp(args)
    elif args.command == "evaluate":
        evaluate_grasp(args)
    elif args.command == "examples":
        ExampleIndex.build(
            args.examples_file,
            args.output_dir,
            args.batch_size,
            args.overwrite,
            args.log_level,
        )


if __name__ == "__main__":
    main()
