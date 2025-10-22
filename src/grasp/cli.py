import argparse
import json
import os
import random
import sys

from tqdm import tqdm
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import (
    dump_json,
    dump_jsonl,
    load_jsonl,
    load_text,
)
from universal_ml_utils.logging import get_logger, setup_logging
from universal_ml_utils.ops import extract_field

from grasp.build import build_indices, get_data
from grasp.build.data import merge_kgs
from grasp.configs import (
    GraspConfig,
    ModelConfig,
    NotesFromExplorationConfig,
    NotesFromOutputsConfig,
    NotesFromSamplesConfig,
)
from grasp.core import generate, load_notes, setup
from grasp.evaluate import evaluate_f1, evaluate_with_judge
from grasp.manager import find_embedding_model
from grasp.notes import (
    take_notes_from_exploration,
    take_notes_from_outputs,
    take_notes_from_samples,
)
from grasp.server import serve
from grasp.tasks import Task, default_input_field
from grasp.tasks.examples import ExampleIndex, load_example_indices
from grasp.utils import (
    get_available_knowledge_graphs,
    is_invalid_model_output,
    parse_parameters,
)


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
        choices=[task.value for task in Task],
        default=Task.SPARQL_QA.value,
        help="Task to run/consider",
    )


def add_overwrite_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output",
    )


def parse_args() -> argparse.Namespace:
    available_kgs = get_available_knowledge_graphs()

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
        default=6789,
        help="Port to run the GRASP server on",
    )
    server_parser.add_argument(
        "--max-connections",
        type=int,
        default=10,
        help="Maximum number of concurrent connections",
    )
    server_parser.add_argument(
        "--max-idle-time",
        type=int,
        default=300,
        help="Maximum idle time for a connection in seconds (after which the connection is closed)",
    )
    server_parser.add_argument(
        "--max-generation-time",
        type=int,
        default=300,
        help="Maximum time for a single generation in seconds (after which the generation is cancelled)",
    )
    server_parser.add_argument(
        "--log-outputs",
        type=str,
        help="File to log all inputs and outputs to (in JSONL format)",
    )

    # run GRASP on a single input
    run_parser = subparsers.add_parser(
        "run",
        help="Run GRASP on a single input",
    )
    add_config_arg(run_parser)
    run_parser.add_argument(
        "-i",
        "--input",
        type=str,
        help="Input for task (e.g., a question for 'sparql-qa'), "
        "if not given, read from stdin",
    )
    run_parser.add_argument(
        "-if",
        "--input-format",
        type=str,
        choices=["text", "json"],
        default="text",
        help="Format of the input (raw text or JSON)",
    )
    run_parser.add_argument(
        "--input-field",
        type=str,
        default=None,
        help="Field to extract input from (if None, a task-specific default is used, "
        "but only if input format is 'json')",
    )
    add_task_arg(run_parser)

    # run GRASP on file with inputs
    file_parser = subparsers.add_parser(
        "file",
        help="Run GRASP on a file with inputs in JSONL format",
    )
    add_config_arg(file_parser)
    file_parser.add_argument(
        "--progress",
        action="store_true",
        help="Show a progress bar",
    )
    file_parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        help="Path to file in JSONL format to run GRASP on, if not given, read JSONL from stdin",
    )
    file_parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the inputs",
    )
    file_parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N inputs",
    )
    file_parser.add_argument(
        "--take",
        type=int,
        default=None,
        help="Limit number of inputs (after skipping) to N",
    )
    file_parser.add_argument(
        "--input-field",
        type=str,
        default=None,
        help="Field to extract input from (if None, a task-specific default is used)",
    )
    file_parser.add_argument(
        "--output-file",
        type=str,
        help="File to write the output to",
    )
    file_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Retry failed inputs (only used with --output-file)",
    )
    file_parser.add_argument(
        "--none-output-invalid",
        action="store_true",
        help="Consider None outputs as invalid when retrying failed inputs",
    )
    add_task_arg(file_parser)
    add_overwrite_arg(file_parser)

    # run GRASP note taking
    note_parser = subparsers.add_parser(
        "notes",
        help="Take notes on interactions of GRASP with one or more knowledge graphs",
    )

    # add second level for note parser
    note_subparsers = note_parser.add_subparsers(
        title="note commands",
        description="Available note commands",
        dest="note_command",
        required=True,
    )

    note_samples_parser = note_subparsers.add_parser(
        "samples",
        help="Take notes for a task and one or more knowledge graphs "
        "by running GRASP on exemplary task samples",
    )
    add_config_arg(note_samples_parser)
    note_samples_parser.add_argument(
        "output_dir",
        type=str,
        help="Save note taking results in this directory",
    )
    add_task_arg(note_samples_parser)
    add_overwrite_arg(note_samples_parser)

    note_interactions_parser = note_subparsers.add_parser(
        "outputs",
        help="Take notes from existing outputs / runs of GRASP",
    )
    add_config_arg(note_interactions_parser)
    note_interactions_parser.add_argument(
        "output_dir",
        type=str,
        help="Save note taking results in this directory",
    )
    add_task_arg(note_interactions_parser)
    add_overwrite_arg(note_interactions_parser)

    note_explore_parser = note_subparsers.add_parser(
        "explore",
        help="Take notes for a task and one or more knowledge graphs "
        "by exploring the knowledge graphs (without any task samples or outputs)",
    )
    add_config_arg(note_explore_parser)
    note_explore_parser.add_argument(
        "output_dir",
        type=str,
        help="Save note taking results in this directory",
    )
    add_overwrite_arg(note_explore_parser)

    # evaluate GRASP output
    eval_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate GRASP output against a reference file (only for 'sparql-qa' task)",
    )

    eval_subparsers = eval_parser.add_subparsers(
        title="evaluation commands",
        description="Available evaluation commands",
        dest="evaluate_command",
        required=True,
    )
    eval_f1_parser = eval_subparsers.add_parser(
        "f1",
        help="Evaluate GRASP output using F1 score based on query results",
    )
    eval_f1_parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=available_kgs,
        help="Knowledge graph the input questions refer to",
    )
    eval_f1_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with question-sparql pairs in JSONL format",
    )
    eval_f1_parser.add_argument(
        "prediction_file",
        type=str,
        help="Path to file with GRASP predictions as produced by the 'file' command",
    )
    eval_f1_parser.add_argument(
        "--endpoint",
        type=str,
        help="SPARQL endpoint to use for evaluation",
    )
    eval_f1_parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Maximum duration for a single query in seconds",
    )
    eval_f1_parser.add_argument(
        "--exact-after",
        type=int,
        default=1024,
        help="Result size after which exact F1 score instead of assignment F1 score "
        "is used (due to performance reasons)",
    )

    eval_judge_parser = eval_subparsers.add_parser(
        "judge",
        help="Evaluate GRASP outputs by picking the best using a judge model",
    )
    eval_judge_parser.add_argument(
        "config",
        type=str,
        help="Path to the GRASP configuration file (used for the judge)",
    )
    eval_judge_parser.add_argument(
        "input_file",
        type=str,
        help="Path to file with inputs in JSONL format",
    )
    eval_judge_parser.add_argument(
        "prediction_files",
        type=str,
        nargs="+",
        help="Paths to files with GRASP predictions as produced by the 'file' command",
    )
    eval_judge_parser.add_argument(
        "evaluation_file",
        type=str,
        help="Path to file to write the evaluation results to",
    )

    eval_parser.add_argument(
        "--retry-failed",
        action="store_true",
        help="Rerun failed evaluations",
    )
    add_overwrite_arg(eval_parser)

    # get data for GRASP indices
    data_parser = subparsers.add_parser(
        "data",
        help="Get entity and property data for a knowledge graph",
    )
    data_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Knowledge graph to get data for",
    )
    data_parser.add_argument(
        "--endpoint",
        type=str,
        help="SPARQL endpoint of the knowledge graph "
        "(if not given, the endpoint at qlever.cs.uni-freiubrg.de/api/<kg> is used)",
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
        "--query-parameters",
        type=str,
        nargs="*",
        help="Extra query parameters sent to the knowledge graph endpoint",
    )
    data_parser.add_argument(
        "--replace",
        type=str,
        nargs="*",
        help="Variables with format {key} in SPARQL queries to replace with values in format key:value",
    )
    data_parser.add_argument(
        "--disable-id-fallback",
        action="store_true",
        help="Disable fallback to using IDs as labels if no label is found",
    )
    add_overwrite_arg(data_parser)

    # merge multiple knowledge graphs
    merge_parser = subparsers.add_parser(
        "merge",
        help="Merge data from multiple knowledge graphs. The first knowledge graph is the primary one, "
        "to which data from the other knowledge graphs is added. Therefore, the merged knowledge graph will "
        "have the same number of entities and properties as the first knowledge graph.",
    )
    merge_parser.add_argument(
        "knowledge_graphs",
        type=str,
        nargs="+",
        choices=available_kgs,
        help="Knowledge graphs to merge",
    )
    merge_parser.add_argument(
        "knowledge_graph",
        type=str,
        help="Name of the merged knowledge graph",
    )
    add_overwrite_arg(merge_parser)

    # build GRASP indices
    index_parser = subparsers.add_parser(
        "index",
        help="Build entity and property indices for a knowledge graph",
    )
    index_parser.add_argument(
        "knowledge_graph",
        type=str,
        choices=available_kgs,
        help="Knowledge graph to build indices for",
    )
    index_parser.add_argument(
        "--entities-type",
        type=str,
        choices=["prefix", "similarity"],
        default="prefix",
        help="Type of entity index to build",
    )
    index_parser.add_argument(
        "--properties-type",
        type=str,
        choices=["prefix", "similarity"],
        default="similarity",
        help="Type of property index to build",
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
        help="Build an example index used for few-shot learning (only for 'sparql-qa' task)",
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
    config = GraspConfig(**load_config(args.config))

    managers = setup(config)

    model = find_embedding_model(managers)
    example_indices = load_example_indices(args.task, config, model=model)

    notes, kg_notes = load_notes(config)

    if args.input_field is None:
        input_field = default_input_field(args.task)
    else:
        input_field = args.input_field

    run_on_file = args.command == "file"
    outputs = []
    if run_on_file:
        if args.input_file is None:
            inputs = [json.loads(line) for line in sys.stdin]
        else:
            inputs = load_jsonl(args.input_file)

        # id fallback in case of missing ids
        # before shuffling/skipping/taking
        for i, ipt in enumerate(inputs):
            id = extract_field(ipt, "id")
            if id is None:
                ipt["id"] = str(i)

        if args.shuffle:
            assert config.seed is not None, (
                "Seed must be set for deterministic shuffling"
            )
            random.seed(config.seed)
            random.shuffle(inputs)

        skip = max(0, args.skip)
        take = args.take or len(inputs)
        inputs = inputs[skip : skip + take]

        if args.output_file:
            if os.path.exists(args.output_file) and not args.overwrite:
                outputs = load_jsonl(args.output_file)

            # save info in config file next to output file
            output_stem, _ = os.path.splitext(args.output_file)
            config_file = output_stem + ".config.json"

            dump_json(config.model_dump(), config_file, indent=2)

        if args.progress:
            # wrap with tqdm
            inputs = tqdm(inputs, desc=f"GRASP for {args.task}")

    else:
        if args.input is None:
            ipt = sys.stdin.read()
        else:
            ipt = args.input

        if args.input_format == "json":
            inputs = [json.loads(ipt)]
        else:
            inputs = [{"input": ipt}]
            input_field = "input"  # overwrite

    for i, ipt in enumerate(inputs):
        id = extract_field(ipt, "id") or "unknown"

        if input_field is not None:
            ipt = extract_field(ipt, input_field)

        assert ipt is not None, f"Input not found for input {i:,}"

        if i < len(outputs):
            # overwrite id
            output = outputs[i]
            output["id"] = id
            if not args.retry_failed or not is_invalid_model_output(
                output,
                args.none_output_invalid,
            ):
                continue

        *_, output = generate(
            args.task,
            ipt,
            config,
            managers,
            kg_notes,
            notes,
            example_indices=example_indices,
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

    if run_on_file and args.output_file is not None:
        # final dump, necessary if no new outputs were added
        # but some outputs were updated with ids
        dump_jsonl(outputs, args.output_file)


def serve_grasp(args: argparse.Namespace) -> None:
    config = GraspConfig(**load_config(args.config))
    serve(
        config,
        args.port,
        args.log_level,
        args.log_outputs,
        args.max_connections,
        args.max_generation_time,
        args.max_idle_time,
    )


def get_grasp_data(args: argparse.Namespace) -> None:
    replace = parse_parameters(args.replace or [])
    query_params = parse_parameters(args.query_parameters or [])

    if args.entity_sparql is not None:
        args.entity_sparql = load_text(args.entity_sparql).strip()
        for key, value in replace.items():
            args.entity_sparql = args.entity_sparql.replace(f"{{{key}}}", value)

    if args.property_sparql is not None:
        args.property_sparql = load_text(args.property_sparql).strip()
        for key, value in replace.items():
            args.property_sparql = args.property_sparql.replace(f"{{{key}}}", value)

    get_data(
        args.knowledge_graph,
        args.endpoint,
        args.entity_sparql,
        args.property_sparql,
        query_params,
        args.overwrite,
        args.disable_id_fallback,
        args.log_level,
    )


def take_grasp_notes(args: argparse.Namespace) -> None:
    note_cmd = args.note_command

    config = load_config(args.config)

    if note_cmd == "samples":
        take_notes_from_samples(
            args.task,
            NotesFromSamplesConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )
    elif note_cmd == "outputs":
        take_notes_from_outputs(
            args.task,
            NotesFromOutputsConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )
    elif note_cmd == "explore":
        take_notes_from_exploration(
            NotesFromExplorationConfig(**config),
            args.output_dir,
            args.overwrite,
            args.log_level,
        )


def evaluate_grasp(args: argparse.Namespace) -> None:
    eval_cmd = args.evaluate_command

    if eval_cmd == "f1":
        evaluate_f1(
            args.knowledge_graph,
            args.input_file,
            args.prediction_file,
            args.endpoint,
            args.overwrite,
            args.timeout,
            args.retry_failed,
            args.exact_after,
            args.log_level,
        )

    elif eval_cmd == "judge":
        judge_config = ModelConfig(**load_config(args.config))
        evaluate_with_judge(
            args.input_file,
            args.prediction_files,
            args.evaluation_file,
            judge_config,
            args.overwrite,
            args.retry_failed,
            args.log_level,
        )


def main():
    args = parse_args()
    if args.all_loggers:
        setup_logging(args.log_level)

    if args.command == "data":
        get_grasp_data(args)

    elif args.command == "merge":
        merge_kgs(
            args.knowledge_graphs,
            args.knowledge_graph,
            args.overwrite,
            args.log_level,
        )

    elif args.command == "index":
        build_indices(
            args.knowledge_graph,
            args.entities_type,
            args.properties_type,
            args.overwrite,
            args.log_level,
            sim_batch_size=args.sim_batch_size,
            sim_precision=args.sim_precision,
            sim_embedding_dim=args.sim_embedding_dim,
        )

    elif args.command == "notes":
        take_grasp_notes(args)

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
