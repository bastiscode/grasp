import argparse
import hashlib
import os
import random
import sys
from collections import Counter
from pathlib import Path

import streamlit as st
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_json, load_json, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.apps.shared import (
    display_name_from_file,
    render_messages,
    render_output_panel,
    render_sparql_result,
)
from grasp.configs import KgConfig
from grasp.manager import load_kg_manager
from grasp.utils import is_invalid_output

logger = get_logger("EXPERT APP")

st.set_page_config(page_title="Blind Expert Evaluation", page_icon="🧑‍⚖️", layout="wide")
st.markdown(
    """
    <style>
    div[data-baseweb="select"] > div {
        height: auto !important;
        min-height: 2.5rem;
    }

    div[data-baseweb="select"] span {
        white-space: normal !important;
        overflow: visible !important;
        text-overflow: unset !important;
        word-break: break-word !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Blind expert evaluation app")
    parser.add_argument("input_file", help="JSONL with id/question/sparql")
    parser.add_argument(
        "prediction_files",
        nargs="+",
        help="At least two GRASP prediction JSONL files",
    )
    parser.add_argument(
        "--evaluation",
        required=True,
        help="Path to the evaluation JSON (will be read if it exists, written on save)",
    )
    parser.add_argument(
        "--kg-config",
        default=None,
        help="Optional KgConfig YAML for executing ground-truth SPARQL",
    )
    return parser.parse_args(argv)


@st.cache_resource
def _load_kg_manager_cached(kg_config_path: str):
    cfg = KgConfig(**load_config(kg_config_path))
    return load_kg_manager(cfg)


@st.cache_data
def _load_jsonl_cached(path: str, _mtime: float) -> list:
    return load_jsonl(path)


def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _letter(i: int) -> str:
    return chr(ord("A") + i)


def _evaluation_label(path: str) -> dict[str, str]:
    p = Path(path)
    filename = p.stem
    benchmark = p.parent.name if p.parent.name else "?"
    group = p.parent.parent.name if p.parent.parent.name else "?"

    if p.parent.name == "rank":
        benchmark = p.parent.parent.name if p.parent.parent.name else "?"
        group = p.parent.parent.parent.name if p.parent.parent.parent.name else "?"

    return {"group": group, "benchmark": benchmark, "evaluation": filename}


def _load_evaluation(path: str, prediction_files: list[str]) -> dict:
    if os.path.exists(path):
        try:
            data = load_json(path)
            if data.get("prediction_files") != prediction_files:
                st.warning(
                    "Existing evaluation file references different prediction_files "
                    "than those passed on the command line. Using the CLI list; "
                    "previously-saved per-id indices will still refer to the stored order."
                )
            data.setdefault("prediction_files", prediction_files)
            data.setdefault("evaluations", {})
            return data
        except Exception as e:
            logger.warning(f"Failed to read evaluation file {path}: {e}")

    return {
        "prediction_files": prediction_files,
        "expert_config": {
            "kg_config": None,
            "evaluator": os.environ.get("USER"),
        },
        "evaluations": {},
    }


def _recompute_summary(evaluation_state: dict) -> None:
    prediction_files: list[str] = evaluation_state["prediction_files"]
    evaluations = evaluation_state["evaluations"]

    verdict_dist = Counter(
        ev["verdict"] for ev in evaluations.values() if ev.get("err") is None
    )
    score_sums: dict[int, float] = {}
    score_counts: dict[int, int] = {}
    for ev in evaluations.values():
        if ev.get("err") is not None:
            continue
        scores = ev.get("scores") or {}
        for k, v in scores.items():
            idx = int(k)
            score_sums[idx] = score_sums.get(idx, 0.0) + float(v)
            score_counts[idx] = score_counts.get(idx, 0) + 1

    summary: dict = {}
    total = sum(verdict_dist.values()) or 1
    for idx, count in verdict_dist.most_common():
        key = prediction_files[idx] if idx is not None else "tie"
        entry = {"count": count, "ratio": count / total}
        if idx is not None:
            n = score_counts.get(idx, 0)
            entry["avg_score"] = (score_sums[idx] / n) if n else None
            entry["n_scores"] = n
        summary[key] = entry

    # make sure every prediction file has an entry (avg_score visible even
    # when it never won a verdict)
    for idx, pred_file in enumerate(prediction_files):
        if pred_file in summary:
            continue
        n = score_counts.get(idx, 0)
        summary[pred_file] = {
            "count": 0,
            "ratio": 0.0,
            "avg_score": (score_sums[idx] / n) if n else None,
            "n_scores": n,
        }

    evaluation_state["summary"] = summary


def _save_evaluation(evaluation_state: dict, path: str) -> None:
    _recompute_summary(evaluation_state)
    dump_json(evaluation_state, path)


def _status_marker(evaluations: dict, example_id: str) -> str:
    ev = evaluations.get(example_id)
    if not ev:
        return "○"
    if ev.get("err"):
        return "❌"
    return "✅"


def main() -> None:
    # Streamlit swallows argv[0] as the script name; everything after `--` is ours.
    args = _parse_args(sys.argv[1:])

    prediction_files: list[str] = list(args.prediction_files)
    if len(prediction_files) < 2:
        st.error("Please provide at least two prediction files to compare.")
        return

    # Load input data
    try:
        inputs = _load_jsonl_cached(args.input_file, _mtime(args.input_file))
    except Exception as e:
        st.error(f"Failed to load input file {args.input_file}: {e}")
        return

    input_by_id = {
        row["id"]: row for row in inputs if isinstance(row, dict) and "id" in row
    }
    if not input_by_id:
        st.error("Input file contains no valid rows with an 'id' field.")
        return

    # Load prediction files (cached on mtime)
    pred_by_file: list[dict] = []
    for pf in prediction_files:
        try:
            rows = _load_jsonl_cached(pf, _mtime(pf))
        except Exception as e:
            st.error(f"Failed to load prediction file {pf}: {e}")
            return
        pred_by_file.append(
            {row["id"]: row for row in rows if isinstance(row, dict) and "id" in row}
        )

    # Evaluation state in session (persisted on each save)
    state_key = f"expert_eval::{args.evaluation}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _load_evaluation(
            args.evaluation, prediction_files
        )
    evaluation_state: dict = st.session_state[state_key]
    evaluation_state["expert_config"]["kg_config"] = args.kg_config
    evaluations: dict = evaluation_state["evaluations"]

    # Sidebar: example picker
    st.sidebar.title("Blind Expert Evaluation")
    evaluation_label = _evaluation_label(args.evaluation)
    st.sidebar.markdown(f"**Group:** `{evaluation_label['group']}`")
    st.sidebar.markdown(f"**Benchmark:** `{evaluation_label['benchmark']}`")
    st.sidebar.markdown(f"**Evaluation:** `{evaluation_label['evaluation']}`")
    st.sidebar.caption(f"{len(prediction_files)} candidate file(s)")
    sidebar_judgement_placeholder = st.sidebar.empty()

    example_ids = [
        i for i in input_by_id.keys() if all(i in preds for preds in pred_by_file)
    ]
    if not example_ids:
        st.error("No example id is present in every prediction file.")
        return

    show_only_unrated = st.sidebar.checkbox("Only show unrated", value=False)
    show_judgement_in_sidebar = st.sidebar.checkbox(
        "Show judgement panel in sidebar",
        value=False,
    )
    rate_each = st.sidebar.checkbox(
        "Collect 1-5 scores per candidate",
        value=True,
        help="When enabled, shows sliders inside the verdict box so you can also "
        "rate each candidate individually in addition to picking a winner. "
        "A score of 5 is best, and 1 is worst.",
    )
    id_pool = (
        [i for i in example_ids if not evaluations.get(i)]
        if show_only_unrated
        else example_ids
    )
    if not id_pool:
        st.sidebar.success("All examples have been evaluated 🎉")
        id_pool = example_ids

    def _format_id(i: str) -> str:
        marker = _status_marker(evaluations, i)
        q = input_by_id[i].get("question", "")
        return f"{marker} {i} — {q}"

    if (
        "current_id" not in st.session_state
        or st.session_state["current_id"] not in id_pool
    ):
        st.session_state["current_id"] = id_pool[0]
    if (
        "example_selectbox" not in st.session_state
        or st.session_state["example_selectbox"] not in id_pool
    ):
        st.session_state["example_selectbox"] = st.session_state["current_id"]

    if "pending_selected_id" in st.session_state:
        pending_selected_id = st.session_state.pop("pending_selected_id")
        if pending_selected_id in id_pool:
            st.session_state["current_id"] = pending_selected_id
            st.session_state["example_selectbox"] = pending_selected_id

    selected_id = st.sidebar.selectbox(
        "Example",
        id_pool,
        format_func=_format_id,
        key="example_selectbox",
    )
    st.session_state["current_id"] = selected_id

    # progress
    n_done = sum(
        1 for i in example_ids if evaluations.get(i) and not evaluations[i].get("err")
    )
    st.sidebar.progress(
        n_done / max(1, len(example_ids)), text=f"{n_done}/{len(example_ids)} rated"
    )

    sample = input_by_id[selected_id]
    st.markdown("### Question")
    st.markdown(sample.get("question", "(no question)"))

    # Blind candidate ordering (stable per id)
    perm = list(range(len(prediction_files)))
    random.seed(hashlib.sha256(selected_id.encode()).hexdigest())
    random.shuffle(perm)
    candidate_entries = []  # list[(letter, canonical_idx, output_entry)]
    for letter_i, canonical_idx in enumerate(perm):
        output_entry = pred_by_file[canonical_idx].get(selected_id)
        candidate_entries.append((_letter(letter_i), canonical_idx, output_entry))

    existing = evaluations.get(selected_id) or {}
    has_existing_evaluation = selected_id in evaluations
    letters = [_letter(i) for i in range(len(prediction_files))]
    choice_options = letters + ["Tie"]

    default_choice = "Tie"
    if existing.get("verdict") is not None:
        try:
            canonical_verdict = int(existing["verdict"])
            letter_idx = perm.index(canonical_verdict)
            default_choice = _letter(letter_idx)
        except (ValueError, IndexError):
            default_choice = "Tie"

    def _render_judgement_panel(
        panel, in_sidebar: bool
    ) -> tuple[str, str, dict[str, int], bool, bool, bool]:
        with panel:
            with st.container(border=not in_sidebar):
                with st.form(key=f"verdict_form::{selected_id}"):
                    if in_sidebar:
                        choice = st.segmented_control(
                            "Best candidate",
                            options=choice_options,
                            default=default_choice,
                            selection_mode="single",
                        )
                        explanation = st.text_input(
                            "Notes / explanation (optional)",
                            value=existing.get("explanation") or "",
                            width="stretch",
                        )
                    else:
                        selector_col, explanation_col = st.columns([2, 5])
                        with selector_col:
                            choice = st.segmented_control(
                                "Best candidate",
                                options=choice_options,
                                default=default_choice,
                                selection_mode="single",
                            )
                        with explanation_col:
                            explanation = st.text_input(
                                "Notes / explanation (optional)",
                                value=existing.get("explanation") or "",
                                width="stretch",
                            )

                    letter_scores: dict[str, int] = {}
                    if rate_each:
                        score_cols = st.columns(len(prediction_files))
                        for score_col, (letter, canonical_idx, _) in zip(
                            score_cols, candidate_entries
                        ):
                            prev = None
                            if existing.get("scores"):
                                prev = existing["scores"].get(str(canonical_idx))
                            letter_scores[letter] = score_col.slider(
                                f"Candidate {letter} Score",
                                min_value=1,
                                max_value=5,
                                value=int(prev) if prev is not None else 3,
                                key=f"score::{selected_id}::{letter}",
                            )

                    save_col, save_next_col, clear_col = st.columns(3)
                    with save_col:
                        submitted = st.form_submit_button(
                            "Save", type="primary", use_container_width=True
                        )
                    with save_next_col:
                        submitted_next = st.form_submit_button(
                            "Save & Next", use_container_width=True
                        )
                    with clear_col:
                        submitted_clear = st.form_submit_button(
                            "Clear",
                            use_container_width=True,
                            disabled=not has_existing_evaluation,
                        )

        return (
            choice,
            explanation,
            letter_scores,
            submitted,
            submitted_next,
            submitted_clear,
        )

    if show_judgement_in_sidebar:
        gt_col = st.container()
        (
            choice,
            explanation,
            letter_scores,
            submitted,
            submitted_next,
            submitted_clear,
        ) = _render_judgement_panel(sidebar_judgement_placeholder.container(), in_sidebar=True)
    else:
        # Top row: ground truth (left) and verdict box (right), 50/50.
        gt_col, verdict_col = st.columns(2)
        (
            choice,
            explanation,
            letter_scores,
            submitted,
            submitted_next,
            submitted_clear,
        ) = _render_judgement_panel(verdict_col, in_sidebar=False)

    with gt_col:
        with st.expander("Ground Truth", expanded=False):
            gt_sparql = sample.get("sparql")
            if not gt_sparql:
                st.info("No ground-truth SPARQL available for this example.")
            else:
                st.markdown("**SPARQL**")
                st.code(gt_sparql, language="sparql")
                if args.kg_config:
                    try:
                        manager = _load_kg_manager_cached(args.kg_config)
                    except Exception as e:
                        st.error(
                            f"Failed to load KG manager from {args.kg_config}: {e}"
                        )
                    else:
                        gt_key = f"gt_result::{args.kg_config}::{selected_id}"
                        if gt_key not in st.session_state:
                            try:
                                result = manager.execute_sparql(gt_sparql)
                                st.session_state[gt_key] = {
                                    "ok": True,
                                    "formatted": manager.format_sparql_result(result),
                                }
                            except Exception as e:
                                st.session_state[gt_key] = {
                                    "ok": False,
                                    "error": str(e),
                                }

                        gt = st.session_state[gt_key]
                        if gt["ok"]:
                            st.markdown("**Result**")
                            render_sparql_result(gt["formatted"])
                        else:
                            st.error(
                                f"Failed to execute ground-truth SPARQL: {gt['error']}"
                            )

    # Blind candidate columns (trace expander on top)
    st.markdown("### Candidates")
    columns_per_row = 3
    for i in range(0, len(candidate_entries), columns_per_row):
        chunk = candidate_entries[i : i + columns_per_row]
        cols = st.columns(len(chunk))
        for col, (letter, _canonical_idx, output_entry) in zip(cols, chunk):
            with col.container(border=True):
                st.markdown(f"### Candidate {letter}")
                invalid = output_entry is None or is_invalid_output(output_entry)
                if invalid and output_entry is not None:
                    st.warning("Invalid output.")
                if output_entry and "messages" in output_entry:
                    elapsed = output_entry.get("elapsed")
                    num_steps = len(output_entry["messages"])
                    if isinstance(elapsed, (int, float)):
                        trace_label = f"Trace ({elapsed:.1f} seconds, {num_steps} steps)"
                    else:
                        trace_label = f"Trace ({num_steps} steps)"
                    with st.expander(trace_label, expanded=False):
                        render_messages(output_entry)
                render_output_panel(output_entry, show_answer=False)

    if submitted_clear:
        if has_existing_evaluation:
            del evaluations[selected_id]
            try:
                _save_evaluation(evaluation_state, args.evaluation)
            except Exception as e:
                st.error(f"Failed to clear evaluation: {e}")
            else:
                st.success(f"Cleared evaluation for {selected_id}.")
                st.rerun()
    elif submitted or submitted_next:
        if choice == "Tie":
            canonical_verdict = None
        else:
            letter_idx = letters.index(choice)
            canonical_verdict = perm[letter_idx]

        scores_canonical: dict[str, int] | None = None
        if rate_each:
            scores_canonical = {}
            for letter, canonical_idx, _ in candidate_entries:
                scores_canonical[str(canonical_idx)] = int(letter_scores[letter])

        evaluations[selected_id] = {
            "verdict": canonical_verdict,
            "scores": scores_canonical,
            "explanation": explanation.strip(),
            "err": None,
        }

        try:
            _save_evaluation(evaluation_state, args.evaluation)
        except Exception as e:
            st.error(f"Failed to save evaluation: {e}")
        else:
            st.success(f"Saved evaluation for {selected_id}.")

            if submitted_next:
                try:
                    current_idx = id_pool.index(selected_id)
                except ValueError:
                    current_idx = -1
                if current_idx + 1 < len(id_pool):
                    next_id = id_pool[current_idx + 1]
                else:
                    next_id = id_pool[0]
                st.session_state["pending_selected_id"] = next_id
                st.rerun()

            # reveal identity after save
            with st.expander("Reveal candidate identities", expanded=True):
                for letter, canonical_idx, _ in candidate_entries:
                    st.markdown(
                        f"- **{letter}** → `{display_name_from_file(prediction_files[canonical_idx])}`"
                        f" ({prediction_files[canonical_idx]})"
                    )

    # Always show running summary at the bottom
    if evaluation_state.get("summary"):
        st.markdown("---")
        st.markdown("### Running Summary")
        rows = []
        for pred_file in prediction_files:
            entry = evaluation_state["summary"].get(pred_file, {})
            rows.append(
                {
                    "Model": display_name_from_file(pred_file),
                    "Wins": entry.get("count", 0),
                    "Win Ratio": f"{entry.get('ratio', 0.0):.2%}",
                    "Avg Score": (
                        f"{entry['avg_score']:.2f} (n={entry.get('n_scores', 0)})"
                        if entry.get("avg_score") is not None
                        else "—"
                    ),
                }
            )
        tie = evaluation_state["summary"].get("tie")
        if tie:
            rows.append(
                {
                    "Model": "(tie)",
                    "Wins": tie.get("count", 0),
                    "Win Ratio": f"{tie.get('ratio', 0.0):.2%}",
                    "Avg Score": "—",
                }
            )
        st.table(rows)


if __name__ == "__main__":
    main()
