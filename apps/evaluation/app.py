import json
import os
import re
import sys
from collections import defaultdict
from functools import reduce
from pathlib import Path

import natsort
import pandas as pd
import streamlit as st
from streamlit_autorefresh import st_autorefresh
from universal_ml_utils.io import load_json, load_jsonl

from grasp.model import Message
from grasp.utils import is_invalid_evaluation, is_invalid_output

# Set page configuration
st.set_page_config(page_title="SPARQL QA Evaluation", page_icon="📊", layout="wide")


def parse_model_name(filename):
    """Parse model name and additional info from filename."""
    # Remove .jsonl extension
    basename = os.path.basename(filename)
    if basename.endswith(".jsonl"):
        basename = basename[:-6]

    # Split by the first dot
    parts = basename.split(".", 1)
    model_name = parts[0]
    additional_info = parts[1] if len(parts) > 1 else ""

    return model_name, additional_info


# Using universal_ml_utils.io.load_json and load_jsonl functions


def load_available_data():
    """Find all available benchmarks and models."""
    data_root = Path(sys.argv[1])
    benchmarks = {}

    # Find all directories that might contain our data structure
    for kg_dir in data_root.glob("*"):
        if not kg_dir.is_dir():
            continue

        kg = kg_dir.name

        for benchmark_dir in kg_dir.glob("*"):
            if not benchmark_dir.is_dir():
                continue

            benchmark = benchmark_dir.name

            test_file = benchmark_dir / "test.jsonl"
            outputs_dir = benchmark_dir / "outputs"

            if test_file.exists() and outputs_dir.exists() and outputs_dir.is_dir():
                # This directory follows our expected structure
                # Find model output files (.jsonl) and their evaluation files (.evaluation.json)
                model_files = []
                for model_file in outputs_dir.glob("*.jsonl"):
                    # Skip evaluation jsonl files
                    if ".evaluation." in model_file.name:
                        continue

                    # Check if there's an evaluation file
                    eval_file = model_file.parent / f"{model_file.stem}.evaluation.json"

                    model_files.append((model_file, eval_file))

                if model_files:
                    if kg not in benchmarks:
                        benchmarks[kg] = {}

                    # Parse model names and additional info
                    models_info = {}
                    for model_file, eval_file in model_files:
                        model_name, additional_info = parse_model_name(model_file.stem)
                        display_name = (
                            f"{model_name} ({additional_info})"
                            if additional_info
                            else model_name
                        )

                        models_info[display_name] = {
                            "output_file": str(model_file),
                            "eval_file": str(eval_file),
                        }

                    benchmarks[kg][benchmark] = {
                        "test_file": str(test_file),
                        "models": models_info,
                    }

    return benchmarks


def load_ranking_data():
    """Find all available ranking evaluation files, organized by ranking filename."""
    data_root = Path(sys.argv[1])
    rankings = {}

    # Find all directories that might contain ranking data
    for kg_dir in data_root.glob("*"):
        if not kg_dir.is_dir():
            continue

        kg = kg_dir.name

        for benchmark_dir in kg_dir.glob("*"):
            if not benchmark_dir.is_dir():
                continue

            benchmark = benchmark_dir.name
            rank_dir = benchmark_dir / "rank"

            if rank_dir.exists() and rank_dir.is_dir():
                # Find all ranking evaluation JSON files
                for rank_file in rank_dir.glob("*.json"):
                    # Group by filename (without extension)
                    filename = rank_file.stem

                    if filename not in rankings:
                        rankings[filename] = []

                    rankings[filename].append(
                        {"kg": kg, "benchmark": benchmark, "filepath": str(rank_file)}
                    )

    return rankings


def load_model_outputs(output_file):
    """Load model outputs from a JSONL file and convert to dictionary by ID."""
    outputs_list = load_jsonl(output_file)
    outputs_dict = {}

    # Convert list to dictionary by ID
    for output in outputs_list:
        if output is None:
            continue

        assert output["id"] not in outputs_dict, (
            f"Duplicate id {output['id']} in {output_file}"
        )
        outputs_dict[output["id"]] = output

    return outputs_dict


def calculate_average_steps_and_time(outputs_dict):
    """
    Calculate average steps and time from model outputs.

    Returns:
        tuple: (avg_steps, avg_time) or (None, None) if no data
    """
    if not outputs_dict:
        return None, None

    total_steps = 0
    total_time = 0
    count = 0

    for output in outputs_dict.values():
        if "messages" not in output:
            continue

        # Count steps (messages that are not user or system)
        steps = sum(
            1 for msg in output["messages"] if msg.get("role") not in ["user", "system"]
        )
        total_steps += steps

        # Get elapsed time
        if "elapsed" in output:
            total_time += output["elapsed"]

        count += 1

    if count == 0:
        return None, None

    return (total_steps / count, total_time / count)


def calculate_metrics(
    ground_truth,
    model_outputs,
    model_evaluations,
    empty_target_valid=False,
):
    total = len(ground_truth)

    num_outputs = len(model_outputs)
    num_invalid_outputs = sum(
        is_invalid_output(output) for output in model_outputs.values()
    )

    num_evaluations = len(model_evaluations)

    num_invalid_evaluations = 0
    total_f1 = 0.0
    total_accuracy = 0.0
    total_time = 0.0
    time_count = 0

    for id, evaluation in model_evaluations.items():
        invalid_evaluation = is_invalid_evaluation(
            evaluation,
            empty_target_valid,
        )
        num_invalid_evaluations += invalid_evaluation
        if invalid_evaluation:
            continue

        # get f1 score
        if "prediction" in evaluation:
            f1_score = evaluation["prediction"]["score"]
            # Get elapsed time directly from the prediction
            elapsed = evaluation["prediction"]["elapsed"]
            total_time += elapsed
            time_count += 1
        else:
            f1_score = 0.0

        total_f1 += f1_score
        total_accuracy += float(f1_score == 1.0)

    num_valid_evaluations = num_evaluations - num_invalid_evaluations

    f1_score = total_f1 / max(num_valid_evaluations, 1)
    accuracy = total_accuracy / max(num_valid_evaluations, 1)
    avg_time = total_time / max(time_count, 1)

    # Calculate average steps using the average steps and time function
    avg_steps, _ = calculate_average_steps_and_time(model_outputs)

    return {
        "num_total": total,
        "num_outputs": num_outputs,
        "num_invalid_outputs": num_invalid_outputs,
        "num_evaluations": num_evaluations,
        "num_invalid_evaluations": num_invalid_evaluations,
        "accuracy": accuracy,
        "f1": f1_score,
        "time": avg_time,
        "steps": avg_steps if avg_steps is not None else 0,
    }


def load_and_process_data(
    test_file,
    model_info,
    restrict_to_common_valid=False,
    empty_target_valid=False,
):
    """Load and process data for the selected benchmark and models."""
    # Load test data (ground truth)
    ground_truth = load_jsonl(test_file)

    # Load model outputs and evaluation data
    model_outputs = {}
    model_eval_data = {}

    for model_name, model_files in model_info.items():
        output_file = model_files["output_file"]
        eval_file = model_files["eval_file"]

        # Load model output file
        try:
            model_outputs[model_name] = load_model_outputs(output_file)
        except Exception as e:
            # Log error but don't display in UI
            print(f"Error loading model output file {output_file}: {e}")
            model_outputs[model_name] = {}

        # Load evaluation data
        try:
            eval_data = load_json(eval_file)
            # restrict to those for which we have model outputs
            eval_data = {
                id: eval
                for id, eval in eval_data.items()
                if id in model_outputs[model_name]
            }
            model_eval_data[model_name] = eval_data
        except Exception as e:
            # Log error but don't display in UI
            print(f"Error loading evaluation file {eval_file}: {e}")
            model_eval_data[model_name] = {}

    # Find common ids that are valid (output and evaluation) across
    # all SELECTED models (only those in model_info)
    if restrict_to_common_valid:
        all_ids = []
        for model_name, evaluations in model_eval_data.items():
            # Only consider models that were explicitly selected
            if model_name not in model_info:
                continue

            outputs = model_outputs[model_name]
            valid_ids = set(
                id
                for id, evaluation in evaluations.items()
                if not is_invalid_evaluation(evaluation, empty_target_valid)
                and not is_invalid_output(outputs[id])
            )
            all_ids.append(valid_ids)

        common_ids = set()
        if all_ids:
            common_ids = reduce(lambda x, y: x.intersection(y), all_ids)

        # Filter outputs and evaluations to common IDs
        model_outputs = {
            model_name: {
                id: output for id, output in outputs.items() if id in common_ids
            }
            for model_name, outputs in model_outputs.items()
        }

        model_eval_data = {
            model_name: {id: eval for id, eval in evals.items() if id in common_ids}
            for model_name, evals in model_eval_data.items()
        }

        # Filter ground truth to common IDs
        ground_truth = [gt for gt in ground_truth if gt["id"] in common_ids]

    # Calculate metrics
    metrics = {}
    for model_name, outputs in model_outputs.items():
        metrics[model_name] = calculate_metrics(
            ground_truth,
            outputs,
            model_eval_data[model_name],
            empty_target_valid=empty_target_valid,
        )

    return ground_truth, model_outputs, model_eval_data, metrics


def setup_model_selection(available_models, selected_models_dict=None):
    """
    Setup model selection UI with regex filtering and checkboxes in expanders.

    Parameters:
    - available_models: Dictionary or list of available models
    - selected_models_dict: Optional dictionary to populate with selected models

    Returns:
    - Dictionary with selected models (key: model name, value: True if selected)
    """
    # Initialize session state for model selection
    if "model_regex" not in st.session_state:
        st.session_state.model_regex = ""

    if "model_selections" not in st.session_state:
        st.session_state.model_selections = {}

    # Add regex filter for model selection
    model_regex = st.sidebar.text_input(
        "Filter models by regex pattern",
        value=st.session_state.model_regex,
        key="model_regex_input",
        help="Enter a regex pattern to automatically select matching models and deselect non-matching ones. Example: 'llama|phi' selects all LLaMA and Phi models.",
    )

    # Update stored regex value
    st.session_state.model_regex = model_regex

    # Check if regex changed (compare to previous value)
    regex_changed = False
    if "previous_model_regex" not in st.session_state:
        st.session_state.previous_model_regex = ""
        regex_changed = model_regex != ""
    else:
        regex_changed = model_regex != st.session_state.previous_model_regex

    # Update previous regex pattern
    st.session_state.previous_model_regex = model_regex

    # Dictionary to track selected models
    if selected_models_dict is None:
        selected_models = {}
    else:
        selected_models = selected_models_dict

    # Group models by name (before the first dot)
    model_groups = defaultdict(list)
    model_list = (
        available_models.keys()
        if isinstance(available_models, dict)
        else available_models
    )
    for model_display_name in model_list:
        model_name = (
            model_display_name.split(" (")[0]
            if " (" in model_display_name
            else model_display_name
        )
        model_groups[model_name].append(model_display_name)

    # Find models that match the regex if provided
    matching_models = set()

    # If regex changed, handle selection updates
    if regex_changed:
        # If empty regex, select all models
        if not model_regex:
            for model in model_list:
                st.session_state.model_selections[model] = True
        else:
            # Otherwise, use regex to determine selections
            try:
                regex = re.compile(model_regex)
                # Find all models that match the regex
                for model_name, variants in model_groups.items():
                    for variant in variants:
                        if regex.search(variant):
                            matching_models.add(variant)

                # Show a warning if no models match at all
                if not matching_models:
                    st.sidebar.warning(f"No models match the pattern '{model_regex}'")

                # Update selections based on matches
                for model in model_list:
                    matches = model in matching_models
                    st.session_state.model_selections[model] = matches

            except re.error as e:
                st.sidebar.error(f"Invalid regex pattern: {e}")
                # In case of error, don't change selections
    elif model_regex:
        # If regex didn't change but is non-empty, still build matching_models for display
        try:
            regex = re.compile(model_regex)
            for model_name, variants in model_groups.items():
                for variant in variants:
                    if regex.search(variant):
                        matching_models.add(variant)

            if not matching_models:
                st.sidebar.warning(f"No models match the pattern '{model_regex}'")
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {e}")

    # Initialize any missing models in state with default selection True
    for model in model_list:
        if model not in st.session_state.model_selections:
            # Default is selected (True)
            st.session_state.model_selections[model] = True

    # Display checkboxes for each model in expanders grouped by model name
    for model_name, variants in sorted(model_groups.items()):
        # Create an expander for each model family
        with st.sidebar.expander(f"**{model_name}**", expanded=False):
            for variant in sorted(variants):
                # Get display name without parentheses
                display_name = variant.replace(model_name + " (", "").replace(")", "")
                if display_name == variant:  # No parentheses found
                    checkbox_label = "default"
                else:
                    checkbox_label = display_name

                # Use the stored value from session state
                current_value = st.session_state.model_selections.get(variant, True)

                # Create checkbox with current value
                selected = st.checkbox(
                    checkbox_label, value=current_value, key=f"model_{variant}"
                )

                # Update both the returned dictionary and the session state
                if isinstance(selected_models, dict):
                    selected_models[variant] = selected
                elif isinstance(selected_models, list) and selected:
                    selected_models.append(variant)

                # Update session state
                st.session_state.model_selections[variant] = selected

    return selected_models


# Additional view functions
def show_predictions_view(available_data):
    """Show a view focused on examining model outputs in detail."""
    st.title("Outputs Analysis")

    # Sidebar for benchmark and model selection
    st.sidebar.title("Benchmark Settings")

    kg_options = list(available_data.keys())
    # Set Wikidata as default if available
    default_index = kg_options.index("wikidata") if "wikidata" in kg_options else 0
    selected_kg = st.sidebar.selectbox("Select Group", kg_options, index=default_index)

    benchmark_options = list(available_data[selected_kg].keys())
    # Set default benchmark based on selected knowledge graph
    default_benchmark = (
        "qald10"
        if selected_kg == "wikidata"
        else "wqsp"
        if selected_kg == "freebase"
        else benchmark_options[0]
    )
    # Make sure the default benchmark exists in the options
    default_index = (
        benchmark_options.index(default_benchmark)
        if default_benchmark in benchmark_options
        else 0
    )
    selected_benchmark = st.sidebar.selectbox(
        "Select Benchmark", benchmark_options, index=default_index
    )

    # Get available models for this benchmark
    benchmark_info = available_data[selected_kg][selected_benchmark]
    available_models = benchmark_info["models"]

    # Add empty ground truth handling option
    empty_target_valid = st.sidebar.checkbox(
        "Count empty ground truth as valid",
        value=False,
        help="When checked, ground truth with size 0 (empty result sets) will be counted as valid",
    )

    # Allow user to select a model
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select a Model")
    model_options = list(available_models.keys())

    # Preferred model to select by default when first loading
    preferred_model = "gpt-41 (search_extended_with_feedback)"

    # Store the selected model in session state to persist between benchmark changes
    if "predictions_view_model" not in st.session_state:
        # Initialize with preferred model
        st.session_state.predictions_view_model = preferred_model

    # Add regex filter for model selection
    model_regex = st.sidebar.text_input(
        "Filter models by regex pattern", st.session_state.stored_model_regex
    )
    # Update session state with current filter
    st.session_state.stored_model_regex = model_regex

    # Determine which model options to display
    display_options = model_options
    if model_regex:
        try:
            regex = re.compile(model_regex)
            filtered_model_options = [
                model for model in model_options if regex.search(model)
            ]
            if filtered_model_options:
                display_options = filtered_model_options
            else:
                st.sidebar.warning(f"No models match the pattern '{model_regex}'")
        except re.error as e:
            st.sidebar.error(f"Invalid regex pattern: {e}")

    # Find index for the model selection based on stored value or preferred model
    if st.session_state.predictions_view_model in display_options:
        # Use previously selected model if available in current options
        default_index = display_options.index(st.session_state.predictions_view_model)
    else:
        # Otherwise use preferred model if available, or first model if not
        default_index = next(
            (i for i, m in enumerate(display_options) if preferred_model == m), 0
        )

    # Show select box with appropriate default index
    selected_model = st.sidebar.selectbox("Model", display_options, index=default_index)

    # Store selected model in session state for next time
    st.session_state.predictions_view_model = selected_model

    # Filter type
    prediction_options = ["All Outputs", "Invalid Outputs", "Invalid Evaluations"]
    prediction_type = st.sidebar.radio("Output Type", prediction_options)

    # Load data
    test_file = benchmark_info["test_file"]
    model_info = {selected_model: available_models[selected_model]}

    ground_truth, model_outputs, model_eval_data, metrics = load_and_process_data(
        test_file,
        model_info,
        restrict_to_common_valid=False,
        empty_target_valid=empty_target_valid,
    )

    # Load configuration from external config file instead of model output
    config_file = available_models[selected_model]["output_file"].replace(
        ".jsonl", ".config.json"
    )
    try:
        config_data = load_json(config_file)
    except Exception as e:
        print(f"Error loading config file {config_file}: {e}")
        config_data = {}

    # Get outputs and evaluations for the selected model
    model_name = selected_model
    outputs = model_outputs[model_name]
    evaluations = model_eval_data[model_name] if model_name in model_eval_data else {}

    # Filter outputs based on the selected type
    filtered_outputs = {}
    if prediction_type == "Invalid Outputs":
        filtered_outputs = {
            id: output for id, output in outputs.items() if is_invalid_output(output)
        }
    elif prediction_type == "Invalid Evaluations":
        filtered_outputs = {
            id: output
            for id, output in outputs.items()
            if id in evaluations
            and is_invalid_evaluation(
                evaluations[id], empty_target_valid=empty_target_valid
            )
        }
    else:  # All Outputs
        filtered_outputs = outputs

    # Show info about the number of outputs
    st.info(
        f"Found {len(filtered_outputs)} {prediction_type.lower()} for model {model_name}"
    )

    if not filtered_outputs:
        st.warning(f"No {prediction_type.lower()} found for this model.")
        return

    # Always sort by ID using natural sorting
    sorted_ids = natsort.natsorted(filtered_outputs.keys())

    # Create a selection dropdown with ID and question
    # Create a dictionary mapping IDs to questions from ground truth
    id_to_question = {
        ex.get("id"): ex.get("question", "No question") for ex in ground_truth
    }

    # Create selection options with ID and question
    selection_options = [
        f"{id} - {id_to_question.get(id, 'No question')}" for id in sorted_ids
    ]

    # Show the dropdown
    selected_option = st.selectbox("Select an example:", selection_options)

    # Extract the ID from the selected option
    selected_id = selected_option.split(" - ")[0] if selected_option else None

    if not selected_id:
        return

    # Display the selected example
    output = filtered_outputs[selected_id]

    # Find corresponding ground truth
    gt_example = next((ex for ex in ground_truth if ex.get("id") == selected_id), None)

    # Main container for the example
    with st.container():
        # Display question from ground truth and ID
        st.subheader("Question")
        question = id_to_question.get(selected_id, "No question found")
        st.write(question)
        st.write(f"ID: {selected_id}")

        # Display ground truth if available
        if gt_example:
            st.subheader("Ground Truth SPARQL")
            st.code(gt_example.get("sparql", ""), language="sparql")

        # Display model output
        st.subheader("Model Output")

        new_format = "output" in output
        if new_format:
            sparql_query = (output["output"] or {}).get(
                "sparql", "No SPARQL query generated or found"
            )
        else:
            sparql_query = output.get("sparql", "No SPARQL query generated or found")

        st.code(sparql_query, language="sparql")

        # Display evaluation if available
        if selected_id in evaluations:
            eval_data = evaluations[selected_id]
            st.subheader("Evaluation")

            # Create columns for evaluation metrics (use more columns for better spacing)
            eval_cols = st.columns([1, 1, 1.5])
            with eval_cols[0]:
                if "prediction" in eval_data and "score" in eval_data["prediction"]:
                    f1_score = eval_data["prediction"]["score"]
                    st.metric("F1", f"{f1_score:.2f}")
                else:
                    st.metric("F1", "N/A")

            with eval_cols[1]:
                if "prediction" in eval_data and "elapsed" in eval_data["prediction"]:
                    elapsed = eval_data["prediction"]["elapsed"]
                    st.metric("Time (s)", f"{elapsed:.3f}")
                else:
                    st.metric("Time (s)", "N/A")

            with eval_cols[2]:
                if is_invalid_evaluation(eval_data, empty_target_valid):
                    # Check if invalid due to empty ground truth
                    if (
                        not empty_target_valid
                        and "target" in eval_data
                        and eval_data["target"].get("size", None) == 0
                        and eval_data["target"].get("err", None) is None
                    ):
                        st.metric("Status", "❌ Empty Ground Truth")
                    else:
                        st.metric("Status", "❌ Invalid")
                elif (
                    "prediction" in eval_data
                    and eval_data["prediction"].get("score", 0) == 1.0
                ):
                    st.metric("Status", "✅ Exact Match")
                else:
                    st.metric("Status", "⚠️ Partial Match")

            # Show any error message
            if "error" in eval_data:
                st.error(f"Error: {eval_data['error']}")

            # Show ground truth errors if available
            if (
                "target" in eval_data
                and eval_data["target"] is not None
                and "err" in eval_data["target"]
                and eval_data["target"]["err"] is not None
            ):
                st.error(f"Ground Truth Error: {eval_data['target']['err']}")

            # Show prediction errors if available
            if (
                "prediction" in eval_data
                and eval_data["prediction"] is not None
                and "err" in eval_data["prediction"]
                and eval_data["prediction"]["err"] is not None
            ):
                st.error(f"Prediction Error: {eval_data['prediction']['err']}")

        # Display model configuration if available
        if config_data:
            with st.expander("Model Configuration"):
                st.json(config_data)

        # Display full generation process (messages)
        if "messages" not in output:
            st.info("No generation process (messages) available for this output.")
            return

        st.subheader("Generation Process")

        if new_format:
            # try new message format first
            try:
                messages = [Message(**msg) for msg in output["messages"]]
                for i, message in enumerate(messages):
                    role = message.role.capitalize()
                    if isinstance(message.content, str):
                        if not message.content:
                            continue  # Skip empty messages

                        st.markdown(f"**{role}:**")
                        st.markdown(message.content)

                    else:
                        content = message.content.get_content()
                        if not content and not message.content.tool_calls:
                            continue  # Skip empty messages

                        if "reasoning" in content:
                            st.markdown("**Reasoning:**")
                            st.markdown(content["reasoning"])

                        if "content" in content:
                            if "reasoning" in content:
                                st.markdown("**Content:**")
                            st.markdown(content["content"])

                        for tool_call in message.content.tool_calls:
                            st.markdown(f"**Tool: {tool_call.name}**")
                            st.code(
                                json.dumps(tool_call.args, indent=2), language="json"
                            )
                            st.markdown("**Result:**")
                            st.markdown(tool_call.result)

                    if i < len(messages) - 1:
                        st.markdown("---")

                return
            except Exception:
                pass

        # fallback to old message format
        def display_tool_call(call, tool_responses):
            """Display a single tool call and its response."""
            tool_call_id = call.get("id")
            tool_name = call.get("function", {}).get("name", "unknown")
            tool_args = call.get("function", {}).get("arguments", "{}")

            # Format arguments
            formatted_args = json.dumps(json.loads(tool_args), indent=2)

            # Show tool call
            st.markdown(f"**Tool: {tool_name}**")
            st.code(formatted_args, language="json")

            # Show corresponding tool response if available
            if tool_call_id in tool_responses:
                tool_response = tool_responses[tool_call_id]
                tool_content = tool_response.get("content", "")
                st.markdown("**Result:**")
                st.markdown(tool_content)

        # First, build a lookup map for tool responses by tool_call_id
        tool_responses = {}
        for msg in output["messages"]:
            if msg["role"] == "tool":
                tool_responses[msg["tool_call_id"]] = msg

        # Now process messages with tool calls integrated
        for i, message in enumerate(output["messages"]):
            role = message.get("role", "unknown")

            # Skip tool messages as we'll show them with their calls
            if role == "tool":
                continue

            reasoning_content = message.get("reasoning_content", "").strip()
            content = message.get("content", "").strip()
            tool_calls = message.get("tool_calls", [])
            if not reasoning_content and not content and not tool_calls:
                continue  # Skip empty messages

            # Display role header
            st.markdown(f"**{role.capitalize()}:**")

            # Show reasoning content if available
            if reasoning_content:
                st.markdown("**Reasoning:**")
                st.markdown(reasoning_content)

            # Show message content
            if content:
                if reasoning_content:
                    st.markdown("**Content:**")
                st.markdown(content)

            # Handle tool calls in assistant messages
            for call in tool_calls:
                display_tool_call(call, tool_responses)

            # Add separator between messages
            if i < len(output["messages"]) - 1:
                st.markdown("---")


def validate_ranking_consistency(benchmark_entries):
    """
    Validate consistency across ranking files.

    Checks:
    1. Judge model consistency across all KGs and benchmarks
    2. Prediction file paths don't reference other directories
    3. Same set of models compared across all benchmarks

    Displays warnings in Streamlit UI for any inconsistencies found.
    """
    judge_configs = {}
    prediction_file_sets = {}
    path_issues = []

    for entry in benchmark_entries:
        kg = entry["kg"]
        benchmark = entry["benchmark"]
        rank_file = entry["filepath"]

        try:
            rank_data = load_json(rank_file)

            # Collect judge config
            if "judge_config" in rank_data:
                judge_key = f"{kg}/{benchmark}"
                judge_configs[judge_key] = rank_data["judge_config"]

            # Collect and validate prediction file paths
            if "summary" in rank_data:
                prediction_files = set()
                for key in rank_data["summary"].keys():
                    if key != "tie":
                        # Extract basename for comparison
                        file_path = Path(key)
                        prediction_files.add(file_path.stem)

                        # Check if prediction file path matches the KG/benchmark context
                        # The key should be a relative path like "outputs/model.jsonl"
                        # We need to verify it's from the same kg/benchmark
                        if "../" in key or key.startswith("/"):
                            path_issues.append(
                                f"{kg}/{benchmark}: Prediction file '{key}' uses absolute or parent directory path"
                            )

                benchmark_key = f"{kg}/{benchmark}"
                prediction_file_sets[benchmark_key] = prediction_files

        except Exception as e:
            st.warning(f"Error loading ranking file {rank_file}: {e}")
            continue

    # Check 1: Judge model consistency
    if judge_configs:
        judge_models = {}
        for benchmark_key, judge_config in judge_configs.items():
            # Only compare the model name
            model_name = judge_config.get("model", "Unknown")
            if model_name not in judge_models:
                judge_models[model_name] = []
            judge_models[model_name].append(benchmark_key)

        if len(judge_models) > 1:
            warning_msg = "⚠️ **Inconsistent judge models detected!** Different ranking files use different judge models:\n"
            for model_name, benchmarks in judge_models.items():
                warning_msg += f"\n  {model_name} (used by {', '.join(benchmarks)})"
            st.warning(warning_msg)

    # Check 2: Prediction file path issues
    if path_issues:
        warning_msg = "⚠️ **Prediction file path issues detected!**\n"
        for issue in path_issues:
            warning_msg += f"\n  - {issue}"
        st.warning(warning_msg)

    # Check 3: Same set of models compared
    if prediction_file_sets:
        # Get all unique sets of prediction files
        unique_sets = {}
        for benchmark_key, file_set in prediction_file_sets.items():
            set_key = frozenset(file_set)
            if set_key not in unique_sets:
                unique_sets[set_key] = []
            unique_sets[set_key].append(benchmark_key)

        if len(unique_sets) > 1:
            warning_msg = "⚠️ **Inconsistent prediction files detected!** Different benchmarks are comparing different sets of models:\n"
            for i, (file_set, benchmarks) in enumerate(unique_sets.items(), 1):
                warning_msg += (
                    f"\n  Set {i} (used by {', '.join(benchmarks)}): {sorted(file_set)}"
                )
            st.warning(warning_msg)


def show_ranking_view(ranking_data):
    """Show a view for ranking evaluations across multiple benchmarks."""
    st.title("Ranking View - Cross-Benchmark Comparison")

    if not ranking_data:
        st.warning(
            "No ranking evaluation files found. Please make sure ranking files are in the 'rank' subdirectories."
        )
        return

    # Sidebar for ranking model selection
    st.sidebar.title("Ranking Settings")

    # Get all available ranking filenames
    ranking_options = sorted(ranking_data.keys())

    if not ranking_options:
        st.warning("No ranking files found.")
        return

    # Select ranking comparison to view
    selected_ranking = st.sidebar.selectbox(
        "Select Ranking Comparison", ranking_options, index=0
    )

    # Parse and display the ranking name nicely
    model_name, additional_info = parse_model_name(selected_ranking)
    display_name = (
        f"{model_name} ({additional_info})" if additional_info else model_name
    )

    st.subheader(f"Comparison: {display_name}")

    # Get all benchmarks for this ranking
    benchmark_entries = ranking_data[selected_ranking]

    if not benchmark_entries:
        st.warning(f"No benchmark data found for {selected_ranking}.")
        return

    # Validate consistency across all ranking files
    validate_ranking_consistency(benchmark_entries)

    # Organize benchmarks by knowledge graph for selection
    entries_by_kg = defaultdict(list)
    for entry in benchmark_entries:
        entries_by_kg[entry["kg"]].append(entry)

    kg_options = sorted(entries_by_kg.keys())
    if not kg_options:
        st.warning("No knowledge graphs found for the selected ranking.")
        return

    default_kg_index = kg_options.index("wikidata") if "wikidata" in kg_options else 0
    selected_kg = st.sidebar.selectbox(
        "Select Group",
        kg_options,
        index=default_kg_index,
    )

    benchmark_options = sorted(
        {entry["benchmark"] for entry in entries_by_kg[selected_kg]}
    )

    if not benchmark_options:
        st.warning(f"No benchmarks available for knowledge graph {selected_kg}.")
        return

    default_benchmark_index = 0
    selected_benchmark = st.sidebar.selectbox(
        "Select Benchmark",
        benchmark_options,
        index=default_benchmark_index,
    )

    selected_entry = next(
        (
            entry
            for entry in entries_by_kg[selected_kg]
            if entry["benchmark"] == selected_benchmark
        ),
        None,
    )

    # Extract judge model information from the first available ranking file
    judge_model_info = None
    for entry in benchmark_entries:
        try:
            rank_data = load_json(entry["filepath"])
            if "judge_config" in rank_data and "model" in rank_data["judge_config"]:
                judge_model_info = rank_data["judge_config"]["model"]
                break
        except Exception:
            continue

    # Display judge model if found
    if judge_model_info:
        st.caption(f"**Judge Model:** {judge_model_info}")

    # First pass: collect all unique models across all benchmarks to establish global ordering
    sorted_models = None
    for entry in benchmark_entries:
        try:
            rank_data = load_json(entry["filepath"])

            prediction_models = []
            for file in rank_data["prediction_files"]:
                file_path = Path(file)
                model_name_parsed, additional_info_parsed = parse_model_name(
                    file_path.stem
                )
                model_display_name = (
                    f"{model_name_parsed} ({additional_info_parsed})"
                    if additional_info_parsed
                    else model_name_parsed
                )
                prediction_models.append(model_display_name)

            if sorted_models is None:
                sorted_models = prediction_models
            elif sorted(sorted_models) != sorted(prediction_models):
                sorted_models = None
                break
        except Exception:
            continue

    if sorted_models is None:
        st.warning(
            "Could not establish a consistent set of models across all benchmarks for this ranking comparison."
        )
        return

    # Sort models and assign letters
    model_to_letter = {
        model: chr(65 + i) for i, model in enumerate(sorted_models)
    }  # A=65 in ASCII

    # Display model legend
    if sorted_models:
        st.markdown("**Model Legend:**")
        legend_items = "\n".join(
            [f"- **{model_to_letter[model]}**: {model}" for model in sorted_models]
        )
        st.markdown(legend_items)

    # Process all benchmarks to build comprehensive table (one row per benchmark)
    table_rows = []

    for entry in benchmark_entries:
        kg = entry["kg"]
        benchmark = entry["benchmark"]
        rank_file = entry["filepath"]

        try:
            rank_data = load_json(rank_file)

            if "summary" not in rank_data:
                continue

            summary = rank_data["summary"]

            # Load model outputs to calculate additional metrics
            rank_dir = Path(rank_file).parent
            model_outputs_cache = {}

            for key in summary.keys():
                if key != "tie":
                    # Key is a path relative to the data root, resolve it relative to rank_dir
                    output_file = rank_dir.parent / "outputs" / Path(key).name
                    try:
                        model_outputs_cache[key] = load_model_outputs(str(output_file))
                    except Exception as e:
                        print(f"Failed to load {output_file}: {e}")
                        model_outputs_cache[key] = {}

            # Get evaluation counts
            total_evals = len(rank_data.get("evaluations", {}))
            valid_evals = sum(
                1
                for eval_data in rank_data.get("evaluations", {}).values()
                if eval_data.get("err") is None
            )

            # Build a single row for this benchmark
            row_data = {
                "KG": kg,
                "Benchmark": benchmark,
                "Valid Evals": f"{valid_evals}/{total_evals}",
            }

            # Initialize wins, steps, and time for each model
            model_wins = {}
            model_steps = {}
            model_time = {}
            tie_count = 0

            # Process each model in the summary
            for key, value in summary.items():
                if key == "tie":
                    tie_count = value["count"]
                else:
                    # Parse model name from the prediction file path
                    file_path = Path(key)
                    model_name_parsed, additional_info_parsed = parse_model_name(
                        file_path.stem
                    )
                    model_display_name = (
                        f"{model_name_parsed} ({additional_info_parsed})"
                        if additional_info_parsed
                        else model_name_parsed
                    )

                    # Calculate average metrics
                    avg_steps, avg_time = calculate_average_steps_and_time(
                        model_outputs_cache.get(key, {})
                    )

                    model_wins[model_display_name] = value["count"]
                    model_steps[model_display_name] = avg_steps
                    model_time[model_display_name] = avg_time

            # Calculate total for percentages
            total_comparisons = sum(model_wins.values()) + tie_count

            # Add columns for each model in alphabetical letter order
            for model in sorted_models:
                letter = model_to_letter[model]
                wins = model_wins.get(model, 0)
                percentage = (
                    (wins / total_comparisons * 100) if total_comparisons > 0 else 0
                )
                row_data[f"{letter} Wins"] = f"{percentage:.1f}% ({wins})"
                # Store raw value for determining winner
                row_data[f"_{letter}_wins_raw"] = wins

            # Add ties column
            tie_percentage = (
                (tie_count / total_comparisons * 100) if total_comparisons > 0 else 0
            )
            row_data["Ties"] = f"{tie_percentage:.1f}% ({tie_count})"
            row_data["_ties_raw"] = tie_count

            # Format aggregate steps and time columns
            steps_parts = []
            time_parts = []
            for model in sorted_models:
                steps = model_steps.get(model)
                time = model_time.get(model)
                steps_parts.append(f"{steps:.1f}" if steps is not None else "—")
                time_parts.append(f"{time:.2f}" if time is not None else "—")

            row_data["Avg Steps"] = " / ".join(steps_parts)
            row_data["Avg Time"] = " / ".join(time_parts)

            # Determine which letter or "Ties" has the max wins
            max_wins = max(
                [
                    row_data.get(f"_{model_to_letter[model]}_wins_raw", 0)
                    for model in sorted_models
                ]
                + [row_data["_ties_raw"]]
            )
            row_data["_max_wins"] = max_wins

            winner = None
            for model in sorted_models:
                letter = model_to_letter[model]
                if row_data.get(f"_{letter}_wins_raw", 0) == max_wins and max_wins > 0:
                    winner = letter
                    break
            if winner is None and row_data["_ties_raw"] == max_wins and max_wins > 0:
                winner = "Ties"
            row_data["_winner"] = winner

            table_rows.append(row_data)

        except Exception as e:
            st.error(f"Error loading {kg}/{benchmark}: {str(e)}")
            continue

    if not table_rows:
        st.warning(f"No valid ranking data found for {selected_ranking}.")
        return

    # Create DataFrame
    df = pd.DataFrame(table_rows)

    # Define column order
    display_columns = ["KG", "Benchmark"]
    for model in sorted_models:
        letter = model_to_letter[model]
        display_columns.append(f"{letter} Wins")
    display_columns.extend(["Ties", "Avg Steps", "Avg Time", "Valid Evals"])

    df_display = df[display_columns]

    # Create styling function to highlight winning column
    def highlight_winner(row):
        styles = [""] * len(row)
        row_data = df.iloc[row.name]
        winner = row_data.get("_winner")

        if winner:
            # Find the column index for the winner
            winner_col = f"{winner} Wins" if winner != "Ties" else "Ties"
            if winner_col in display_columns:
                col_idx = display_columns.index(winner_col)
                styles[col_idx] = (
                    "background-color: #005500; color: white; font-weight: bold"
                )

        return styles

    styled_df = df_display.style.apply(highlight_winner, axis=1)
    st.dataframe(styled_df, use_container_width=True, hide_index=True)

    # Show summary statistics
    st.caption(f"Showing {len(benchmark_entries)} benchmark(s) for {display_name}")

    # Detailed sample view for the selected KG and benchmark
    if not selected_entry:
        return

    try:
        selected_rank_data = load_json(selected_entry["filepath"])
    except Exception as exc:
        st.warning(f"Failed to load ranking data for detailed view: {exc}")
        return

    evaluations = selected_rank_data.get("evaluations", {})
    if not evaluations:
        st.info("No detailed evaluations available for the selected benchmark.")
        return

    benchmark_dir = Path(selected_entry["filepath"]).parent.parent
    test_file = benchmark_dir / "test.jsonl"

    try:
        ground_truth_examples = load_jsonl(test_file)
    except Exception as exc:
        print(f"Failed to load test file {test_file}: {exc}")
        ground_truth_examples = []

    id_to_question = {}
    id_to_ground_truth = {}
    for example in ground_truth_examples:
        if not isinstance(example, dict):
            continue
        example_id = example.get("id")
        if example_id is None:
            continue
        id_to_question[example_id] = example.get("question", "No question provided")
        id_to_ground_truth[example_id] = example

    sorted_ids = natsort.natsorted(evaluations.keys())
    if not sorted_ids:
        st.info("No sample identifiers found in the evaluations.")
        return

    st.markdown("---")
    st.subheader(f"Sample Explorer: {selected_kg} / {selected_benchmark}")

    selected_id = st.selectbox(
        "Select an example:",
        sorted_ids,
        format_func=lambda x: f"{x} - {id_to_question.get(x, 'No question provided')}",
        key=f"ranking_sample_{selected_kg}_{selected_benchmark}",
    )

    if not selected_id:
        return

    summary = selected_rank_data.get("summary", {})
    outputs_dir = benchmark_dir / "outputs"
    model_outputs = {}
    summary_model_entries = []
    model_display_order = []
    seen_models = set()

    for path in selected_rank_data["prediction_files"]:
        file_path = Path(path)
        model_name_parsed, additional_info_parsed = parse_model_name(file_path.stem)
        model_display_name = (
            f"{model_name_parsed} ({additional_info_parsed})"
            if additional_info_parsed
            else model_name_parsed
        )

        output_path = outputs_dir / file_path.name
        summary_model_entries.append(
            {
                "display_name": model_display_name,
                "output_path": output_path,
                "summary_key": path,
            }
        )

        if model_display_name in seen_models:
            continue

        try:
            model_outputs[model_display_name] = load_model_outputs(str(output_path))
        except Exception as exc:
            print(f"Failed to load model outputs from {output_path}: {exc}")
            model_outputs[model_display_name] = {}

        model_display_order.append(model_display_name)
        seen_models.add(model_display_name)

    evaluation_entry = evaluations.get(selected_id, {})

    question_text = id_to_question.get(selected_id)
    if question_text:
        st.markdown(f"**Question:** {question_text}")

    ground_truth_entry = id_to_ground_truth.get(selected_id)
    if ground_truth_entry:
        ground_truth_sparql = ground_truth_entry.get("sparql")
        if ground_truth_sparql:
            with st.expander("Ground Truth SPARQL"):
                st.code(ground_truth_sparql, language="sparql")

    st.markdown("**Judge Verdict**")
    if not evaluation_entry:
        st.info("Judge verdict not available for the selected example.")
    else:
        verdict_value = evaluation_entry.get("verdict")

        winning_model_name = None
        if isinstance(verdict_value, int) and 0 <= verdict_value < len(
            summary_model_entries
        ):
            winning_model_name = summary_model_entries[verdict_value]["display_name"]

        verdict_display = (
            f"- Verdict Index: `{verdict_value}`"
            if verdict_value is not None
            else "- Verdict Index: `None`"
        )
        if winning_model_name:
            verdict_display += f" → **{winning_model_name}**"
        st.markdown(verdict_display)

        explanation_text = evaluation_entry.get("explanation")
        err_text = evaluation_entry.get("err")

        if explanation_text:
            st.markdown(explanation_text)
        if err_text:
            st.error(f"Judge error: {err_text}")

    st.markdown("**Model Outputs**")
    if not model_display_order:
        st.info("No model outputs available for this benchmark.")
    else:
        columns_per_row = 3
        for i in range(0, len(model_display_order), columns_per_row):
            current_models = model_display_order[i : i + columns_per_row]
            cols = st.columns(len(current_models))
            for col, model_name in zip(cols, current_models):
                outputs = model_outputs.get(model_name, {})
                output_entry = outputs.get(selected_id)
                output_payload = {}

                if isinstance(output_entry, dict):
                    output_field = output_entry.get("output", output_entry)
                    if isinstance(output_field, dict):
                        output_payload = output_field

                container_cm = None
                try:
                    container_cm = col.container(border=True)
                except TypeError:
                    container_cm = col.container()

                with container_cm:
                    st.markdown(f"**{model_name}**")

                    if not output_entry:
                        st.info("No output available for this example.")
                        continue

                    rendered_any = False
                    sparql_query = output_payload.get("sparql")
                    if sparql_query:
                        st.markdown("**SPARQL**")
                        st.code(sparql_query, language="sparql")
                        rendered_any = True

                    result_data = output_payload.get("result")
                    if result_data is not None:
                        st.markdown("**Result**")
                        if isinstance(result_data, (dict, list)):
                            st.json(result_data)
                        else:
                            st.code(str(result_data), language="json")
                        rendered_any = True

                    selections_data = output_payload.get("selections")
                    if selections_data:
                        st.markdown("**Selections**")
                        if isinstance(selections_data, (dict, list)):
                            st.json(selections_data)
                        else:
                            st.write(selections_data)
                        rendered_any = True

                    answer_text = output_payload.get("answer")
                    if answer_text:
                        with st.expander("Answer"):
                            st.markdown(answer_text)
                        rendered_any = True

                    if not rendered_any:
                        fallback_data = (
                            output_entry.get("output", output_entry)
                            if isinstance(output_entry, dict)
                            else output_entry
                        )
                        st.info(
                            "No structured SPARQL/result/selections/answer available."
                        )
                        st.write(fallback_data)


def show_comprehensive_view(available_data):
    """Show a comprehensive view with a large table of metrics across KGs and benchmarks."""
    st.title("Comprehensive Model Comparison")

    # Add settings to sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Comprehensive View Settings")

    # Add metric selector - only allow accuracy and f1
    metric_options = {
        "F1 Score (%)": "avg_f1",
        "Accuracy (%)": "accuracy",
    }
    selected_metric = st.sidebar.selectbox(
        "Select metric to display:", options=list(metric_options.keys()), index=0
    )
    metric_key = metric_options[selected_metric]

    # Option to restrict evaluation to common examples
    restrict_to_common = st.sidebar.checkbox(
        "Only evaluate on examples where all models have valid outputs and evaluations",
        value=False,
        help="When checked, only examples where all selected models have valid outputs and evaluations will be included in the comparison",
    )

    # Add empty ground truth handling option
    empty_target_valid = st.sidebar.checkbox(
        "Count empty ground truth as valid",
        value=False,
        help="When checked, ground truth with size 0 (empty result sets) will be counted as valid",
    )

    # We'll set up the benchmark checkboxes after gathering all the data for the table
    # Create an empty dictionary to track selected benchmarks
    selected_benchmarks = {}

    # Create a list of all available models across all benchmarks
    all_available_models = set()
    for kg_data in available_data.values():
        for benchmark_info in kg_data.values():
            all_available_models.update(benchmark_info["models"].keys())

    # Group models by name (before the first dot)
    model_groups = defaultdict(list)
    for model_display_name in all_available_models:
        model_name = (
            model_display_name.split(" (")[0]
            if " (" in model_display_name
            else model_display_name
        )
        model_groups[model_name].append(model_display_name)

    # Add model selection
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Models to Include")

    # Use shared model selection function
    selected_models = setup_model_selection(available_models=all_available_models)

    # Create a dictionary to hold all metrics across all KGs and benchmarks
    all_metrics = {}

    # Keep track of KGs and their benchmarks for hierarchical columns
    kg_benchmarks = defaultdict(list)

    # Process each KG and benchmark
    with st.spinner(
        "Loading comprehensive metrics across all knowledge graphs and benchmarks..."
    ):
        for kg_name, kg_data in available_data.items():
            for benchmark_name, benchmark_info in kg_data.items():
                # Don't filter here, we'll do it after gathering all data

                # Load test data
                test_file = benchmark_info["test_file"]

                # Get models for this benchmark
                model_info = benchmark_info["models"]

                # Skip if no models
                if not model_info:
                    continue

                # Track this benchmark under its KG
                kg_benchmarks[kg_name].append(benchmark_name)

                # Filter models based on user selection
                filtered_model_info = {
                    model_name: model_details
                    for model_name, model_details in model_info.items()
                    if model_name in selected_models and selected_models[model_name]
                }

                # Skip if no selected models for this benchmark
                if not filtered_model_info:
                    continue

                # Load and process data with restriction if selected
                _, _, _, metrics = load_and_process_data(
                    test_file,
                    filtered_model_info,
                    restrict_to_common_valid=restrict_to_common,
                    empty_target_valid=empty_target_valid,
                )

                # Store metrics for each model
                for model_name, model_metrics in metrics.items():
                    if model_name not in all_metrics:
                        all_metrics[model_name] = {}

                    # Store with separate KG and benchmark keys for hierarchical display
                    if kg_name not in all_metrics[model_name]:
                        all_metrics[model_name][kg_name] = {}

                    # Extract or calculate metrics values
                    num_outputs = model_metrics["num_outputs"]
                    num_evaluations = model_metrics["num_evaluations"]
                    num_without_evaluation = num_outputs - num_evaluations
                    num_invalid_outputs = model_metrics["num_invalid_outputs"]

                    all_metrics[model_name][kg_name][benchmark_name] = {
                        "avg_f1": model_metrics["f1"],
                        "accuracy": model_metrics["accuracy"],
                        "predictions": num_outputs,
                        "evaluated": num_evaluations,
                        "without_evaluation": num_without_evaluation,
                        "avg_time": model_metrics.get("time", 0),
                        "invalid_targets": model_metrics["num_invalid_evaluations"],
                        "invalid_preds": num_invalid_outputs,
                        "invalid_evaluation": 0,  # Not tracked separately
                    }

    # If there are no metrics, show a warning and return
    if not all_metrics:
        st.warning("No metrics available for comprehensive view.")
        return

    # Group benchmarks by knowledge graph
    benchmark_by_kg = {}
    for kg, benchmarks in kg_benchmarks.items():
        benchmark_by_kg[kg] = sorted(benchmarks)

    # Setup benchmark selection in its own section in sidebar
    st.sidebar.markdown("---")
    st.sidebar.subheader("Select Benchmarks to Include")

    # Get all benchmarks from all KGs
    all_benchmarks = []
    for kg, benchmarks in benchmark_by_kg.items():
        all_benchmarks.extend([(kg, b) for b in benchmarks])

    # Initialize selected benchmarks in session state if not already present
    # Use (kg, benchmark) tuples as keys for better organization
    if "selected_benchmarks" not in st.session_state:
        st.session_state.selected_benchmarks = {
            (kg, b): True for kg, b in all_benchmarks
        }

    # Initialize benchmark_regex in session state if not present
    if "benchmark_regex" not in st.session_state:
        st.session_state["benchmark_regex"] = ""

    # Store previous regex pattern to detect changes
    if "previous_benchmark_regex" not in st.session_state:
        st.session_state["previous_benchmark_regex"] = ""

    # Add regex filter for benchmark selection with persistent state
    # Use stored value for initial value, but don't update directly
    benchmark_regex = st.sidebar.text_input(
        "Filter benchmarks by regex pattern",
        value=st.session_state.stored_benchmark_regex,
        key="benchmark_regex_widget",
        help="Enter a regex pattern to automatically select matching benchmarks and deselect non-matching ones. Example: 'wwq|lcquad' selects WWQ and LC-QuAD benchmarks.",
    )

    # Update stored value for next view
    if "benchmark_regex_widget" in st.session_state:
        st.session_state.stored_benchmark_regex = (
            st.session_state.benchmark_regex_widget
        )

    # Check if regex changed
    regex_changed = benchmark_regex != st.session_state["previous_benchmark_regex"]

    # Update previous regex pattern
    st.session_state["previous_benchmark_regex"] = benchmark_regex

    # No need for a separate "Select All" option as it's redundant
    # Users can select/deselect individual benchmarks

    # Create a checkbox for each benchmark, grouped by knowledge graph
    selected_benchmarks = {}

    # Process each knowledge graph
    for kg in sorted(benchmark_by_kg.keys()):
        benchmarks = benchmark_by_kg[kg]

        # All benchmarks should be visible in the sidebar, but we'll use regex to auto-select
        filtered_benchmarks = benchmarks
        matching_benchmarks = set()

        # If we have a regex pattern, find which benchmarks match it
        if benchmark_regex:
            try:
                regex = re.compile(benchmark_regex)
                matching_benchmarks = {b for b in benchmarks if regex.search(b)}
                # Removed the "No benchmarks match" warning message
            except re.error as e:
                st.sidebar.error(f"Invalid regex pattern: {e}")
        else:
            # If no regex pattern, all benchmarks match (for default selection)
            matching_benchmarks = set(benchmarks)

        # If we have benchmarks to show for this KG, create a section with expander
        if filtered_benchmarks:
            # Create an expander for each KG
            with st.sidebar.expander(f"**{kg}**", expanded=False):
                for benchmark in sorted(filtered_benchmarks):
                    # Create a unique key for the checkbox
                    key = (kg, benchmark)

                    # If regex pattern changed, auto-select checkboxes
                    if regex_changed:
                        if not benchmark_regex:
                            # Empty regex means select all benchmarks
                            should_select = True
                        else:
                            # If benchmark matches regex, select it; otherwise unselect
                            should_select = benchmark in matching_benchmarks
                        # Update the session state
                        st.session_state.selected_benchmarks[key] = should_select

                    # Get the current value from session state
                    current_value = st.session_state.selected_benchmarks.get(key, True)

                    # Create the checkbox
                    selected = st.checkbox(
                        benchmark,
                        value=current_value,
                        key=f"benchmark_{kg}_{benchmark}",
                    )

                    # Store the selection
                    selected_benchmarks[key] = selected
                    # Update session state
                    st.session_state.selected_benchmarks[key] = selected

    # For benchmarks not shown due to filtering, preserve their state
    for kg, benchmarks in benchmark_by_kg.items():
        for benchmark in benchmarks:
            key = (kg, benchmark)
            if key not in selected_benchmarks:
                selected_benchmarks[key] = st.session_state.selected_benchmarks.get(
                    key, True
                )

    # If no benchmarks selected, select all
    if not any(selected_benchmarks.values()):
        selected_benchmarks = {key: True for key in all_benchmarks}
        st.sidebar.warning("No benchmarks selected. Showing all benchmarks.")
        # Update session state
        st.session_state.selected_benchmarks = selected_benchmarks

    # Now filter benchmarks based on selected options
    filtered_kg_benchmarks = defaultdict(list)
    for kg, benchmarks in kg_benchmarks.items():
        for benchmark in benchmarks:
            # Check if this (kg, benchmark) tuple is selected
            if selected_benchmarks.get((kg, benchmark), True):
                filtered_kg_benchmarks[kg].append(benchmark)

    # Replace kg_benchmarks with filtered version
    kg_benchmarks = filtered_kg_benchmarks

    # Prepare data for a pandas MultiIndex DataFrame
    # Sort knowledge graphs and benchmarks for consistent display
    sorted_kgs = sorted(kg_benchmarks.keys())

    # Create a list of tuples for the MultiIndex columns
    column_tuples = [("Model", "")]  # First column is just the model name

    # Add tuples for each KG and benchmark combination
    for kg in sorted_kgs:
        for benchmark in sorted(kg_benchmarks[kg]):
            column_tuples.append((kg, benchmark))

    # Create MultiIndex
    columns = pd.MultiIndex.from_tuples(column_tuples)

    # Prepare data rows
    data_rows = []
    for model_name in sorted(all_metrics.keys()):
        row = [model_name]  # Start with model name

        # Add data for each KG and benchmark
        for kg in sorted_kgs:
            for benchmark in sorted(kg_benchmarks[kg]):
                # Check if we have metrics for this combination
                if (
                    kg in all_metrics[model_name]
                    and benchmark in all_metrics[model_name][kg]
                ):
                    metrics_data = all_metrics[model_name][kg][benchmark]

                    # Format the selected metric value
                    if metric_key in ["avg_f1", "accuracy"]:
                        # Format as percentage with 1 decimal
                        value = metrics_data[metric_key] * 100
                        # Show as 0 if predicted examples is 0
                        if metrics_data["evaluated"] == 0 and metric_key != "evaluated":
                            formatted_value = "0.0"
                        else:
                            formatted_value = f"{value:.1f}"
                    elif metric_key == "avg_time":
                        # Format time with 3 decimals
                        value = metrics_data[metric_key]
                        formatted_value = f"{value:.3f}"
                    elif metric_key in ["evaluated", "predictions"]:
                        # For count metrics, show as integer
                        formatted_value = str(int(metrics_data[metric_key]))
                    else:
                        # For any other metrics
                        formatted_value = str(metrics_data[metric_key])

                    row.append(formatted_value)
                else:
                    row.append("—")  # Em dash for missing data

        data_rows.append(row)

    # Create the DataFrame
    df = pd.DataFrame(data_rows, columns=columns)

    # Display the table with the metric name in the title
    st.subheader(f"{selected_metric} Across All Knowledge Graphs and Benchmarks")

    # Find best and second-best models for each benchmark
    if metric_key in ["avg_f1", "accuracy"]:
        # Dictionary to store rankings for each KG-benchmark pair
        rankings = {}

        # For each KG-benchmark combination, find the best and second-best models
        for kg in sorted_kgs:
            for benchmark in sorted(kg_benchmarks[kg]):
                # Collect all values for this benchmark
                model_values = []
                for model_name in all_metrics:
                    if (
                        kg in all_metrics[model_name]
                        and benchmark in all_metrics[model_name][kg]
                    ):
                        metrics_data = all_metrics[model_name][kg][benchmark]
                        # Only rank if there are evaluated examples
                        if metrics_data["evaluated"] > 0:
                            model_values.append((model_name, metrics_data[metric_key]))

                # Sort by metric value in descending order
                model_values.sort(key=lambda x: x[1], reverse=True)

                # Store the rankings
                rankings[(kg, benchmark)] = {
                    model: rank for rank, (model, _) in enumerate(model_values)
                }

        # Create style dictionaries based on rankings

        # Create a style DataFrame with the same shape as our data DataFrame
        style_df = pd.DataFrame("", index=df.index, columns=df.columns)

        # Fill in the style DataFrame with CSS styles
        for i, row_idx in enumerate(df.index):
            model_name = df.iloc[i, 0]  # Get model name from first column
            for j, col in enumerate(df.columns[1:], 1):  # Skip the first column (Model)
                cell_value = df.iloc[i, j]
                if cell_value == "—":
                    style_df.iloc[i, j] = "background-color: #444444; color: #ffffff"
                else:
                    kg, benchmark = col
                    # Check if this is an invalid model output
                    for model_name_key, metric_data in all_metrics.items():
                        if (
                            model_name == model_name_key
                            and kg in metric_data
                            and benchmark in metric_data[kg]
                        ):
                            if (
                                "invalid_preds" in metric_data[kg][benchmark]
                                and metric_data[kg][benchmark]["invalid_preds"] > 0
                            ):
                                # Apply red color to indicate invalid output
                                style_df.iloc[i, j] = (
                                    "background-color: #990000; color: white"
                                )
                                break

                    # Apply ranking colors if not invalid and in rankings
                    if (
                        not style_df.iloc[i, j]
                        and (kg, benchmark) in rankings
                        and model_name in rankings[(kg, benchmark)]
                    ):
                        rank = rankings[(kg, benchmark)][model_name]
                        if rank == 0:  # Best model
                            style_df.iloc[i, j] = (
                                "background-color: #005500; color: white; font-weight: bold"
                            )
                        elif rank == 1:  # Second best
                            style_df.iloc[i, j] = (
                                "background-color: #003366; color: white"
                            )

        # Apply the styling directly
        styled_df = df.style.apply(lambda _: style_df, axis=None)
        st.dataframe(styled_df, use_container_width=True)
    else:
        # For count metrics, use similar ranking logic
        # Dictionary to store rankings for each KG-benchmark pair
        rankings = {}

        # For each KG-benchmark combination, find the best and second-best models
        for kg in sorted_kgs:
            for benchmark in sorted(kg_benchmarks[kg]):
                # Collect all values for this benchmark
                model_values = []
                for model_name, row_idx in zip(
                    sorted(all_metrics.keys()), range(len(df))
                ):
                    cell_value = df.iloc[row_idx][kg, benchmark]
                    if cell_value != "—":
                        try:
                            # Convert to number for ranking
                            val = int(cell_value)
                            model_values.append((model_name, val))
                        except ValueError:
                            continue

                # Sort by metric value in descending order
                model_values.sort(key=lambda x: x[1], reverse=True)

                # Store the rankings
                rankings[(kg, benchmark)] = {
                    model: rank for rank, (model, _) in enumerate(model_values)
                }

        # Use direct styling with style_df instead of functions

        # Create a style DataFrame with the same shape as our data DataFrame
        style_df = pd.DataFrame("", index=df.index, columns=df.columns)

        # Fill in the style DataFrame with CSS styles
        for i, row_idx in enumerate(df.index):
            model_name = df.iloc[i, 0]  # Get model name from first column
            for j, col in enumerate(df.columns[1:], 1):  # Skip the first column (Model)
                cell_value = df.iloc[i, j]
                if cell_value == "—":
                    style_df.iloc[i, j] = "background-color: #444444; color: #ffffff"
                else:
                    kg, benchmark = col
                    # Check if this is an invalid model output
                    for model_name_key, metric_data in all_metrics.items():
                        if (
                            model_name == model_name_key
                            and kg in metric_data
                            and benchmark in metric_data[kg]
                        ):
                            if (
                                "invalid_preds" in metric_data[kg][benchmark]
                                and metric_data[kg][benchmark]["invalid_preds"] > 0
                            ):
                                # Apply red color to indicate invalid output
                                style_df.iloc[i, j] = (
                                    "background-color: #990000; color: white"
                                )
                                break

                    # Apply ranking colors if not invalid and in rankings
                    if (
                        not style_df.iloc[i, j]
                        and (kg, benchmark) in rankings
                        and model_name in rankings[(kg, benchmark)]
                    ):
                        rank = rankings[(kg, benchmark)][model_name]
                        if rank == 0:  # Best model
                            style_df.iloc[i, j] = (
                                "background-color: #005500; color: white; font-weight: bold"
                            )
                        elif rank == 1:  # Second best
                            style_df.iloc[i, j] = (
                                "background-color: #003366; color: white"
                            )

        # Apply the styling directly
        styled_df = df.style.apply(lambda _: style_df, axis=None)
        st.dataframe(styled_df, use_container_width=True)

    # Add a note about the data and color coding
    st.caption(
        "Note: Dark gray cells indicate no data available for that combination. Best model per benchmark is highlighted in dark green, second best in dark blue. Red cells indicate models with invalid outputs."
    )

    # Show summary statistics
    st.subheader("Summary Statistics")

    # Count total models and benchmarks
    total_models = len(all_metrics)
    total_kgs = len(sorted_kgs)
    total_benchmarks = sum(len(benchmarks) for benchmarks in kg_benchmarks.values())

    # Create 3 columns for the statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Models", total_models)
    with col2:
        st.metric("Knowledge Graphs", total_kgs)
    with col3:
        st.metric("Benchmarks", total_benchmarks)


# Main app
def main():
    st.title("SPARQL Question-Answering Evaluation")

    # Load available benchmarks and models
    available_data = load_available_data()

    if not available_data:
        st.error(
            "No benchmarks found. Please make sure the data structure follows the expected pattern."
        )
        return

    # Initialize session state variables that need to be preserved between views
    # Use different keys for storing values vs widget keys
    if "stored_benchmark_regex" not in st.session_state:
        st.session_state.stored_benchmark_regex = ""

    if "stored_model_regex" not in st.session_state:
        st.session_state.stored_model_regex = ""

    # Create a view selector
    view_options = [
        "Benchmark View",
        "Comprehensive View",
        "Outputs View",
        "Ranking View",
    ]
    # Benchmark View is the default (index=0)
    selected_view = st.sidebar.radio("Select View", view_options, index=0)

    # Show the appropriate view based on selection
    if selected_view == "Comprehensive View":
        # The select_only variable is defined inside the function, so we don't need to pass it
        show_comprehensive_view(available_data)
    elif selected_view == "Outputs View":
        show_predictions_view(available_data)
    elif selected_view == "Ranking View":
        # Load ranking data and show ranking view
        ranking_data = load_ranking_data()
        show_ranking_view(ranking_data)
    else:  # Benchmark View
        # Sidebar for benchmark and model selection
        st.sidebar.title("Benchmark Settings")

        kg_options = list(available_data.keys())
        # Set Wikidata as default if available
        default_index = kg_options.index("wikidata") if "wikidata" in kg_options else 0
        selected_kg = st.sidebar.selectbox(
            "Select Group", kg_options, index=default_index
        )

        benchmark_options = list(available_data[selected_kg].keys())
        # Set default benchmark based on selected knowledge graph
        default_benchmark = (
            "qald10"
            if selected_kg == "wikidata"
            else "wqsp"
            if selected_kg == "freebase"
            else benchmark_options[0]
        )
        # Make sure the default benchmark exists in the options
        default_index = (
            benchmark_options.index(default_benchmark)
            if default_benchmark in benchmark_options
            else 0
        )
        selected_benchmark = st.sidebar.selectbox(
            "Select Benchmark", benchmark_options, index=default_index
        )

        # Option to restrict evaluation to common examples - moved to top, without heading
        restrict_to_common = st.sidebar.checkbox(
            "Only evaluate on examples where all models have valid outputs and evaluations",
            value=False,
            help="When checked, only examples where all selected models have valid outputs and evaluations will be included in the comparison",
        )

        # Add empty ground truth handling option
        empty_target_valid = st.sidebar.checkbox(
            "Count empty ground truth as valid",
            value=False,
            help="When checked, ground truth with size 0 (empty result sets) will be counted as valid",
        )

        # Get available models for this benchmark
        benchmark_info = available_data[selected_kg][selected_benchmark]
        available_models = benchmark_info["models"]

        # Allow selecting multiple models for comparison using checkboxes
        st.sidebar.markdown("---")
        st.sidebar.subheader("Select Models to Compare")

        # Initialize list for currently selected models
        selected_models = []

        # Use shared model selection function
        setup_model_selection(
            available_models=available_models, selected_models_dict=selected_models
        )

        # Filter to only selected models
        model_files = {
            model: available_models[model]
            for model in selected_models
            if model in available_models
        }

        # Main content
        if not model_files:
            st.warning("Please select at least one model for comparison.")
            return

        # Load and process data
        ground_truth, model_outputs, model_eval_data, metrics = load_and_process_data(
            benchmark_info["test_file"],
            model_files,
            restrict_to_common_valid=restrict_to_common,
            empty_target_valid=empty_target_valid,
        )

        # Display metrics with benchmark size
        example_count = len(ground_truth)
        st.subheader(
            f"Performance Metrics for {selected_kg} - {selected_benchmark} ({example_count} examples)"
        )

        # No information banner needed

        # Removed metrics visualization

        # Display metrics table
        # Format the combined predictions column
        combined_predictions = []
        for m in metrics:
            num_outputs = metrics[m]["num_outputs"]
            num_evaluations = metrics[m]["num_evaluations"]
            # Calculate values that might not be directly available in the metrics
            num_without_evaluation = num_outputs - num_evaluations
            num_invalid_outputs = metrics[m]["num_invalid_outputs"]
            num_invalid_evaluations = metrics[m]["num_invalid_evaluations"]

            # Format as: total outputs (missing_evaluations/invalid_evaluations/invalid_outputs)
            combined_predictions.append(
                f"{num_outputs} ({num_without_evaluation}/{num_invalid_evaluations}/{num_invalid_outputs})*"
            )

        metrics_df = pd.DataFrame(
            {
                "Model": list(metrics.keys()),
                "Info*": combined_predictions,
                "Accuracy (%)": [
                    round(metrics[m]["accuracy"] * 100, 1) for m in metrics
                ],  # 1 decimal for percentages
                "Average F1 Score (%)": [
                    round(metrics[m]["f1"] * 100, 1) for m in metrics
                ],  # 1 decimal for percentages
                "Avg. Steps": [
                    round(metrics[m].get("steps", 0), 1) for m in metrics
                ],  # 1 decimal for average steps
                "Avg. Time (sec)": [
                    round(metrics[m].get("time", 0), 3) for m in metrics
                ],  # 3 decimals for time in seconds
            }
        )

        st.dataframe(metrics_df, use_container_width=True)

        # Add explanation for the info column
        empty_ground_truth_text = (
            "" if empty_target_valid else " or those with empty ground truth results"
        )
        st.caption(
            f"* Info format: Outputs (Missing Evaluations/Invalid Evaluations/Invalid Outputs) - 'Outputs' is the total number of model outputs, 'Missing Evaluations' counts outputs without an evaluation, 'Invalid Evaluations' counts evaluations with errors{empty_ground_truth_text}, 'Invalid Outputs' counts model outputs with errors. Note: Accuracy and F1 scores are calculated over all evaluations."
        )

    # Add auto reload option with slider in sidebar (at the bottom)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Auto Reload Settings")
    auto_reload = st.sidebar.checkbox(
        "Enable auto reload",
        value=False,
        help="When checked, the application will automatically reload model outputs and evaluations at the specified interval",
    )
    reload_interval = st.sidebar.slider(
        "Reload interval (seconds)",
        min_value=10,
        max_value=300,
        value=60,
        step=10,
        help="How often to reload model outputs and evaluations",
        disabled=not auto_reload,
    )

    # Set up auto-reload if enabled
    if auto_reload:
        # Show status in sidebar
        with st.sidebar:
            st.caption(
                f"Auto reload enabled. Reloading every {reload_interval} seconds."
            )

        # Initialize auto-refresh component (convert seconds to milliseconds)
        refresh_interval_ms = reload_interval * 1000
        st_autorefresh(interval=refresh_interval_ms, key="model_refresh")


if __name__ == "__main__":
    main()
