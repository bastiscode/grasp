import argparse
import csv
import os
from pathlib import Path

from tqdm import tqdm
from universal_ml_utils.io import dump_jsonl, load_jsonl

from grasp.manager.utils import load_data_and_mapping
from grasp.tasks.cea import Annotation, CellAnnotation
from grasp.utils import get_index_dir, is_invalid_model_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    sub_parser = parser.add_subparsers(dest="command")

    create_parser = sub_parser.add_parser(
        "create",
        help="Create JSONL files for GRASP from CSV files",
    )

    create_parser.add_argument(
        "target",
        type=str,
        help="Target CSV file to determine required columns",
    )
    create_parser.add_argument(
        "tables",
        type=str,
        nargs="+",
        help="CSV files to convert",
    )
    create_parser.add_argument(
        "--context-rows",
        type=int,
        help="Number of context rows to include before and after rows to annotate",
    )
    create_parser.add_argument(
        "--max-rows",
        type=int,
        default=8,
        help="Maximum number of rows to annotate per run",
    )
    create_parser.add_argument(
        "--knowledge-graph",
        type=str,
        help="Knowledge graph name, required if ground truth is present in target CSV",
    )

    merge_parser = sub_parser.add_parser(
        "merge",
        help="Merge output from JSONL files into a single CSV file",
    )
    merge_parser.add_argument(
        "target_csv",
        type=str,
        help="Target CSV file to merge results into",
    )
    merge_parser.add_argument(
        "target_jsonl",
        type=str,
        help="Target JSONL file created with the create command",
    )
    merge_parser.add_argument(
        "output_jsonls",
        type=str,
        nargs="+",
        help="JSONL outputs",
    )
    merge_parser.add_argument(
        "output_csv",
        type=str,
        help="Output CSV file",
    )
    merge_parser.add_argument(
        "--only-wdq",
        action="store_true",
        help="Only include entities from Wikidata (wd:Q...)",
    )

    return parser.parse_args()


def create(args: argparse.Namespace):
    columns = {}
    rows = {}
    gts = {}

    data = mapping = None
    if args.knowledge_graph is not None:
        index_dir = get_index_dir(args.knowledge_graph)
        entities_dir = os.path.join(index_dir, "entities")
        data, mapping = load_data_and_mapping(entities_dir)

    with open(args.target) as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) == 3:
                sample, row, col = row
                gt = None
            elif len(row) == 4:
                sample, row, col, gt = row
            else:
                raise ValueError("Invalid target CSV format, expected 3 or 4 columns")

            col = int(col)
            row = int(row) - 1  # exclude header
            if sample not in columns:
                columns[sample] = set()
            columns[sample].add(col)

            if sample not in rows:
                rows[sample] = set()
            rows[sample].add(row)

            if gt is not None:
                assert data is not None and mapping is not None, (
                    "Data file must be provided if ground truth is present"
                )
                assert gt.startswith("http://www.wikidata.org/entity/")
                identifier = f"<{gt}>"
                id = mapping.get(identifier)
                label = None
                if id is not None:
                    label = data.get_val(id, 1)  # first column is always label

                gts[(sample, row, col)] = Annotation(
                    identifier=identifier,
                    entity="wd:" + gt.split("/")[-1],
                    label=label,
                )

    out_file = Path(args.target).with_suffix(".jsonl")

    id = 0
    samples = []
    for sample in tqdm(sorted(args.tables), "Converting tables"):
        table_name = Path(sample).stem

        with open(sample) as f:
            reader = csv.reader(f)

            header = next(reader)
            assert all(h == f"col{c}" for c, h in enumerate(header))
            # rename column names
            header = ["unknown"] * len(header)
            data = list(reader)

            assert all(len(row) == len(header) for row in data), (
                "Inconsistent number of columns"
            )

            if out_file.stem in columns:
                annotate_columns = sorted(columns[out_file.stem])
                if len(annotate_columns) >= len(header):
                    annotate_columns = None
            else:
                annotate_columns = None

            if out_file.stem in rows:
                annotate_rows = sorted(rows[out_file.stem])
                if len(annotate_rows) >= len(data):
                    annotate_rows = None
            else:
                annotate_rows = None

            if args.context_rows is None:
                ctx = len(data)
            else:
                ctx = args.context_rows

            for lower in range(0, len(data), args.max_rows):
                upper = min(lower + args.max_rows, len(data))
                rows_to_annotate = set(range(lower, upper))
                if annotate_rows is not None:
                    rows_to_annotate = rows_to_annotate.intersection(annotate_rows)

                # store before modifying for context
                mx = min(upper + ctx, len(data))
                mn = max(lower - ctx, 0)
                chunk = data[mn:mx]
                rows_to_annotate = sorted(row - mn for row in rows_to_annotate)

                sample = {
                    "name": table_name,
                    # store offset to convert back to original row numbers
                    "offset": mn,
                    "table": {
                        "header": header,
                        "data": chunk,
                        "annotate_columns": annotate_columns,
                        "annotate_rows": rows_to_annotate,
                    },
                }

                if args.knowledge_graph is not None:
                    annotations = []
                    for row in rows_to_annotate:
                        for col in annotate_columns or range(len(header)):
                            annot = gts.get((table_name, row + mn, col))
                            if annot is None:
                                continue

                            annotations.append(
                                {
                                    "row": row,
                                    "column": col,
                                    **annot.model_dump(),
                                }
                            )

                    sample["annotations"] = annotations

                samples.append(sample)
                id += 1

    dump_jsonl(samples, out_file.as_posix())


def merge(args: argparse.Namespace):
    # load target jsonl
    targets = {}
    for i, input in enumerate(load_jsonl(args.target_jsonl)):
        targets[str(i)] = input

    processed = set()
    annots = {}
    for output in args.output_jsonls:
        for out in load_jsonl(output):
            if is_invalid_model_output(out):
                continue

            id = out["id"]
            target = targets[id]
            table = target["name"]
            offset = target["offset"]

            for row in target["table"]["annotate_rows"] or []:
                row += offset
                row += 1
                processed.add((table, row))

            out = out["output"]
            if out is None:
                continue

            for ann in out["annotations"]:
                ann = CellAnnotation(**ann)
                if args.only_wdq and not ann.entity.startswith("wd:Q"):
                    continue

                ann.row += offset
                ann.row += 1  # account for header
                ent = ann.entity.split(":")[-1]
                annots[(table, ann.row, ann.column)] = ent

    # load target csv
    with open(args.target_csv) as f, open(args.output_csv, "w") as out_f:
        reader = csv.reader(f)
        writer = csv.writer(out_f)

        for table, row, col in reader:
            row = int(row)
            col = int(col)
            annot = annots.get((table, row, col), "NIL")
            if (table, row) not in processed:
                annot = "NP"
            writer.writerow([table, row, col, annot])


if __name__ == "__main__":
    args = parse_args()

    if args.command == "create":
        create(args)
    else:
        merge(args)
