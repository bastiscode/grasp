"""Blind expert evaluation Streamlit app.

Launch via:
    streamlit run expert.py -- <input_jsonl> <pred_file>... \
        --evaluation <out.json> [--kg-config <yaml>]

Candidates are presented anonymously (A, B, C, ...) in an order that is
deterministic per example id (seeded by hash(id)) so the mapping stays
stable across reloads but is blind to the underlying prediction-file order.

The saved evaluation JSON is compatible with the ranking view in app.py:
    {
      "prediction_files": [...],
      "expert_config": {"kg_config": <path | null>, "evaluator": <str | null>},
      "evaluations": {
         "<id>": {
             "verdict": int | null,            # index into prediction_files
             "scores": {"<pred_idx>": int 1-10, ...} | null,
             "explanation": str,
             "err": null
         }
      },
      "summary": {
         "<pred_file>": {"count": int, "ratio": float,
                          "avg_score": float | null, "n_scores": int},
         "tie": {"count": int, "ratio": float}
      }
    }
"""

import argparse
import hashlib
import os
import sys
from collections import Counter

import streamlit as st
from universal_ml_utils.configuration import load_config
from universal_ml_utils.io import dump_json, load_json, load_jsonl
from universal_ml_utils.logging import get_logger

from grasp.configs import KgConfig
from grasp.manager import load_kg_manager
from grasp.utils import is_invalid_output

from grasp.apps.shared import (
    display_name_from_file,
    render_messages,
    render_output_panel,
    render_sparql_result,
    try_load_model_outputs,
)

logger = get_logger("EXPERT APP")

st.set_page_config(page_title="Blind Expert Evaluation", page_icon="🧑‍⚖️", layout="wide")


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
def _load_jsonl_cached(path: str, mtime: float) -> list:
    return load_jsonl(path)


def _mtime(p: str) -> float:
    try:
        return os.path.getmtime(p)
    except OSError:
        return 0.0


def _candidate_permutation(example_id: str, n: int) -> list[int]:
    """Deterministic per-id permutation of [0..n).

    Stable across restarts so the expert always sees the same letter -> file
    mapping for a given example.
    """
    h = hashlib.sha256(str(example_id).encode("utf-8")).digest()
    # rank each prediction-file index by a per-id hash-derived key
    keys = [(int.from_bytes(h[i * 4 : i * 4 + 4] or b"\x00", "big"), i) for i in range(n)]
    # if n > 16, extend by hashing id+i (unlikely for expert eval, but safe)
    if n > len(h) // 4:
        keys = []
        for i in range(n):
            hh = hashlib.sha256(f"{example_id}::{i}".encode("utf-8")).digest()
            keys.append((int.from_bytes(hh[:8], "big"), i))
    keys.sort()
    return [idx for _, idx in keys]


def _letter(i: int) -> str:
    return chr(ord("A") + i)


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

    input_by_id = {row["id"]: row for row in inputs if isinstance(row, dict) and "id" in row}
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
        pred_by_file.append({row["id"]: row for row in rows if isinstance(row, dict) and "id" in row})

    # Evaluation state in session (persisted on each save)
    state_key = f"expert_eval::{args.evaluation}"
    if state_key not in st.session_state:
        st.session_state[state_key] = _load_evaluation(args.evaluation, prediction_files)
    evaluation_state: dict = st.session_state[state_key]
    evaluation_state["expert_config"]["kg_config"] = args.kg_config
    evaluations: dict = evaluation_state["evaluations"]

    # Sidebar: example picker
    st.sidebar.title("Blind Expert Evaluation")
    st.sidebar.caption(f"Evaluation file: `{args.evaluation}`")
    st.sidebar.caption(f"{len(prediction_files)} candidate file(s)")

    example_ids = [i for i in input_by_id.keys() if all(i in preds for preds in pred_by_file)]
    if not example_ids:
        st.error("No example id is present in every prediction file.")
        return

    show_only_unrated = st.sidebar.checkbox("Only show unrated", value=False)
    rate_each = st.sidebar.checkbox(
        "Collect 1-10 scores per candidate",
        value=False,
        help="When enabled, shows sliders inside the verdict box so you can also "
        "rate each candidate individually in addition to picking a winner.",
    )
    id_pool = [i for i in example_ids if not evaluations.get(i)] if show_only_unrated else example_ids
    if not id_pool:
        st.sidebar.success("All examples have been evaluated 🎉")
        id_pool = example_ids

    def _format_id(i: str) -> str:
        marker = _status_marker(evaluations, i)
        q = input_by_id[i].get("question", "")
        return f"{marker} {i} — {q[:60]}"

    if "current_id" not in st.session_state or st.session_state["current_id"] not in id_pool:
        st.session_state["current_id"] = id_pool[0]

    selected_id = st.sidebar.selectbox(
        "Example",
        id_pool,
        index=id_pool.index(st.session_state["current_id"])
        if st.session_state["current_id"] in id_pool
        else 0,
        format_func=_format_id,
        key="example_selectbox",
    )
    st.session_state["current_id"] = selected_id

    # progress
    n_done = sum(1 for i in example_ids if evaluations.get(i) and not evaluations[i].get("err"))
    st.sidebar.progress(n_done / max(1, len(example_ids)), text=f"{n_done}/{len(example_ids)} rated")

    sample = input_by_id[selected_id]
    st.title("Blind Expert Evaluation")
    st.markdown(f"**Question:** {sample.get('question', '(no question)')}")

    # Blind candidate ordering (stable per id)
    perm = _candidate_permutation(selected_id, len(prediction_files))
    candidate_entries = []  # list[(letter, canonical_idx, output_entry)]
    for letter_i, canonical_idx in enumerate(perm):
        output_entry = pred_by_file[canonical_idx].get(selected_id)
        candidate_entries.append((_letter(letter_i), canonical_idx, output_entry))

    existing = evaluations.get(selected_id) or {}
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

    # Top row: ground truth (left) and verdict box (right), 50/50.
    gt_col, verdict_col = st.columns(2)

    with gt_col:
        st.markdown("### Ground Truth")
        gt_sparql = sample.get("sparql")
        if not gt_sparql:
            st.info("No ground-truth SPARQL available for this example.")
        else:
            with st.container(border=True):
                st.markdown("**SPARQL**")
                st.code(gt_sparql, language="sparql")
                if args.kg_config:
                    try:
                        manager = _load_kg_manager_cached(args.kg_config)
                    except Exception as e:
                        st.error(f"Failed to load KG manager from {args.kg_config}: {e}")
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
                                st.session_state[gt_key] = {"ok": False, "error": str(e)}

                        gt = st.session_state[gt_key]
                        if gt["ok"]:
                            st.markdown("**Result**")
                            render_sparql_result(gt["formatted"])
                        else:
                            st.error(
                                f"Failed to execute ground-truth SPARQL: {gt['error']}"
                            )

    with verdict_col:
        st.markdown("### Your Verdict")
        with st.container(border=True):
            with st.form(key=f"verdict_form::{selected_id}"):
                choice = st.segmented_control(
                    "Best candidate",
                    options=choice_options,
                    default=default_choice,
                    selection_mode="single",
                )

                letter_scores: dict[str, int] = {}
                if rate_each:
                    st.markdown("**Scores (1-10)**")
                    score_cols = st.columns(len(prediction_files))
                    for score_col, (letter, canonical_idx, _) in zip(
                        score_cols, candidate_entries
                    ):
                        prev = None
                        if existing.get("scores"):
                            prev = existing["scores"].get(str(canonical_idx))
                        letter_scores[letter] = score_col.slider(
                            f"Candidate {letter}",
                            min_value=1,
                            max_value=10,
                            value=int(prev) if prev is not None else 5,
                            key=f"score::{selected_id}::{letter}",
                        )

                explanation = st.text_area(
                    "Notes / explanation (optional)",
                    value=existing.get("explanation") or "",
                    height=100,
                )

                submitted = st.form_submit_button(
                    "Save", type="primary", use_container_width=True
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
                    with st.expander("Trace", expanded=False):
                        render_messages(output_entry)
                render_output_panel(output_entry, show_answer=False)

    if submitted:
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
        for idx, pred_file in enumerate(prediction_files):
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


main()
