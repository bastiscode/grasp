import argparse
import json
import os
import re

from datasets import load_dataset

from grasp.sparql.manager import load_kg_manager
from grasp.utils import Sample


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    bm = parser.add_mutually_exclusive_group(required=True)

    # wikidata
    bm.add_argument("--wikidata-simple-questions", action="store_true")
    bm.add_argument("--lc-quad2-wikidata", action="store_true")
    bm.add_argument("--qald-10", action="store_true")
    bm.add_argument("--qald-7", type=str)
    bm.add_argument("--wwq", type=str)
    bm.add_argument("--instruct-to-sparql", action="store_true")
    bm.add_argument("--spinach", type=str)
    # unused or unimplemented
    bm.add_argument("--mcwq", type=str)
    bm.add_argument("--kqa-pro", type=str)
    bm.add_argument("--qa-wiki", type=str)
    bm.add_argument("--qlever-wikidata", type=str)
    bm.add_argument("--wikidata-query-logs", type=str)
    # data.add_argument("--time-questions", type=str)
    # data.add_argument("--cron-questions", type=str)
    # data.add_argument("--mkqa", type=str)
    # data.add_argument("--mintaka", type=str)

    # freebase
    bm.add_argument("--wqsp", action="store_true")
    bm.add_argument("--cwq", action="store_true")
    bm.add_argument("--freebase-simple-questions", type=str)
    # unused or unimplemented
    bm.add_argument("--cfq", type=str)
    bm.add_argument("--grail-qa", action="store_true")
    # data.add_argument("--30mqa", type=str)
    # data.add_argument("--graph-questions", type=str)

    # dbpedia
    bm.add_argument("--qald-7-dbpedia", type=str)
    bm.add_argument("--qald-9", action="store_true")
    bm.add_argument("--dbpedia-simple-questions", type=str)
    bm.add_argument("--lc-quad1-dbpedia", action="store_true")
    bm.add_argument("--lc-quad2-dbpedia", action="store_true")
    # unused or unimplemented
    # data.add_argument("--mlpq", type=str)
    # data.add_argument("--monument", type=str)

    # dblp
    bm.add_argument("--dblp-quad", action="store_true")

    # orkg
    bm.add_argument("--sci-qa", action="store_true")

    parser.add_argument("--out-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing")
    parser.add_argument("--seed", type=int, default=22)
    return parser.parse_args()


SPLIT_RENAME = {
    "train": "train",
    "test": "test",
    "dev": "val",
    "valid": "val",
    "validation": "val",
}


def clean_sparql_for_wqsp_and_cwq(sparql: str) -> str:
    lines = []
    for line in sparql.splitlines():
        comment = line.find("#")
        if comment != -1:
            line = line[:comment]
        line = line.replace(" OR ", " || ")
        lines.append(line)

    return "\n".join(line for line in lines if line.strip())


def get_qald_sample(data: dict) -> Sample | None:
    # similar to qald 10 above
    sparql = data["query"]["sparql"]
    if isinstance(data["question"], str):
        queries = json.loads(data["question"])
    else:
        queries = data["question"]

    queries = [
        q["string"] for q in queries if q["language"] == "en" and q["string"] != ""
    ]
    if not queries:
        return None

    # take first as the main question
    question = queries[0]
    paraphrases = queries[1:]

    return Sample(
        question=question,
        sparql=sparql,
        paraphrases=paraphrases,
        info={"answers": data["answers"]},
    )


def load_benchmark(args: argparse.Namespace) -> tuple[str, dict[str, list[Sample]]]:
    output = {}
    if args.wikidata_simple_questions:
        kg = "wikidata"
        data = load_dataset("third_party/KGQA-datasets/simple_wikidata_qa")
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                question = item["question"]
                subj = item["answer"]["subject"]
                obj = item["answer"]["object"]
                prop = item["answer"]["predicate"]

                if prop.startswith("R"):
                    subj, obj = obj, subj
                    subj = "x"
                    prop = "P" + prop[1:]
                else:
                    obj = "x"
                prop = "wdt:" + prop

                if subj == "x":
                    subj = "?" + subj
                    obj = "wd:" + obj
                else:
                    obj = "?" + obj
                    subj = "wd:" + subj

                sparql = f"SELECT ?x WHERE {{ {subj} {prop} {obj} . }}"
                samples.append(Sample(question=question, sparql=sparql))
            output[split] = samples

    elif args.lc_quad2_wikidata or args.lc_quad2_dbpedia:
        kg = "wikidata" if args.lc_quad2_wikidata else "dbpedia"
        data = load_dataset("third_party/KGQA-datasets/lcquad_v2", f"lcquad2-{kg}")

        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                question = item["question"]
                if question is None:
                    continue

                sparql = item["sparql"]

                samples.append(
                    Sample(
                        question=question,
                        sparql=sparql,
                        paraphrases=item["paraphrased_question"],
                    )
                )

            output[split] = samples

    elif args.qald_10:
        kg = "wikidata"
        data = load_dataset("third_party/KGQA-datasets/qald/qald-10.py")
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            output[split] = []
            for item in items:
                sample = get_qald_sample(item)
                if sample is None:
                    continue
                output[split].append(sample)

    elif args.qald_7_dbpedia is not None:
        kg = "dbpedia"
        with open(
            os.path.join(args.qald_7_dbpedia, "qald-7-test-multilingual.json"),
            "r",
        ) as inf:
            test = json.load(inf)

        output["test"] = []
        for item in test["questions"]:
            sample = get_qald_sample(item)
            if sample is None:
                continue
            output["test"].append(sample)

    elif args.qald_7 is not None:
        kg = "wikidata"
        with open(
            os.path.join(args.qald_7, "qald-7-train-en-wikidata.json"),
            "r",
        ) as inf:
            train = json.load(inf)

        with open(
            os.path.join(args.qald_7, "qald-7-test-en-wikidata.json"),
            "r",
        ) as inf:
            test = json.load(inf)

        for data, split in [(train, "train"), (test, "test")]:
            output[split] = []
            for item in data["questions"]:
                sample = get_qald_sample(item)
                if sample is None:
                    continue
                output[split].append(sample)

    elif args.mcwq is not None:
        kg = "wikidata"
        with open(os.path.join(args.mcwq, "dataset.json"), "r") as inf:
            train_data = json.load(inf)
        with open(os.path.join(args.mcwq, "gold_test.json"), "r") as inf:
            test_data = json.load(inf)
        for data, split in [(train_data, "train"), (test_data, "test")]:
            samples = []
            for item in data:
                question = item["questionWithBrackets"]
                # sub out brackets
                question = re.sub(r"\[(.+?)\]", lambda m: m.group(1), question)
                # repair some whitespace issues
                # words followed by 's
                question = re.sub(
                    r"(\w+)\s+('s)(?:\s+|$)",
                    lambda m: m.group(1) + m.group(2) + " ",
                    question,
                )
                # punctuation with surrounding spaces
                question = re.sub(
                    r"\s+([,.?!;])(?:\s+|$)", lambda m: m.group(1) + " ", question
                )
                sparql = item["sparql"]
                samples.append(Sample(question=question, sparql=sparql))

            output[split] = samples

    elif args.wwq is not None:
        kg = "wikidata"
        for split in ["train", "dev", "test"]:
            file = os.path.join(args.wwq, f"{split}.json")
            split = SPLIT_RENAME[split]
            with open(file, "r") as inf:
                data = json.load(inf)

            samples = []
            for item in data:
                question = item.pop("utterance")
                sparql = item.pop("sparql")
                id = item.pop("id")
                samples.append(
                    Sample(
                        id=id,
                        question=question,
                        sparql=sparql,
                        info=item,
                    )
                )

            output[split] = samples

    elif args.kqa_pro is not None:
        raise NotImplementedError
        # kg = "wikidata"
        # for split in ["train", "val", "test"]:
        #     file = os.path.join(args.kqa_pro, f"{split}.json")
        #     with open(file, "r") as inf:
        #         data = json.load(inf)
        #
        #     for item in data:
        #         query = item["question"]
        #         sparql = item.get("sparql", "")
        #         samples.append(Sample(query, sparql))
        #     output[split] = samples

    elif args.qa_wiki is not None:
        kg = "wikidata"
        samples = []
        with open(args.qa_wiki, "r") as inf:
            for line in inf:
                line = line.strip()
                sparql, question = line.split("\t")
                samples.append(Sample(question=question, sparql=sparql))
        output["train"] = samples

    elif args.qlever_wikidata is not None:
        kg = "wikidata"
        samples = []
        with open(args.qlever_wikidata, "r") as inf:
            for line in inf:
                line = line.strip()
                question, sparql = line.split("\t")
                samples.append(Sample(question=question, sparql=sparql))
        output["train"] = samples

    elif args.instruct_to_sparql:
        kg = "wikidata"
        full = load_dataset("PaDaS-Lab/Instruct-to-SPARQL", split="full")
        full_ids = set(item["id"] for item in full)
        split_ids = set()
        for split in ["train", "validation", "test"]:
            items = load_dataset("PaDaS-Lab/Instruct-to-SPARQL", split=split)
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                id = item.pop("id")
                split_ids.add(id)

                questions = item.pop("instructions")
                assert questions, f"No instructions found for item {item}"

                # take first as the main question
                question = questions[0]
                paraphrases = questions[1:]
                sparql = item.pop("sparql_query")

                samples.append(
                    Sample(
                        id=str(id),
                        question=question,
                        sparql=sparql,
                        paraphrases=paraphrases,
                        info=item,
                    )
                )

            output[split] = samples

        output["other"] = []
        not_found = full_ids - split_ids
        for item in full:
            if item["id"] not in not_found:
                continue

            # add not found items to train
            # not found usually means the sparql timed out
            # during execution
            questions = item["instructions"]
            assert questions, f"No instructions found for item {item}"

            question = questions[0]
            paraphrases = questions[1:]

            output["other"].append(
                Sample(
                    question=question,
                    sparql=item["sparql_query"],
                    paraphrases=paraphrases,
                )
            )

    elif args.wikidata_query_logs is not None:
        kg = "wikidata"
        samples = []
        with open(args.wikidata_query_logs, "r") as inf:
            for line in inf:
                line = line.rstrip()
                data = json.loads(line)
                if data is None or data["output"]["generation"] is None:
                    continue

                item = data["output"]["generation"]

                question = item.pop("question")
                sparql = item.pop("cleaned_sparql")
                paraphrases = item.pop("paraphrases")

                samples.append(
                    Sample(
                        question=question,
                        sparql=sparql,
                        paraphrases=paraphrases,
                        info=item,
                    )
                )

        output["train"] = samples

    elif args.grail_qa:
        kg = "freebase"
        data = load_dataset("third_party/KGQA-datasets/grail_qa", "grail_qa")

        output["train"] = [
            Sample(question=item["question"], sparql=item["sparql_query"])
            for item in data["train"]
        ]
        output["val"] = [
            Sample(question=item["question"], sparql=item["sparql_query"])
            for item in data["validation"]
        ]

        data = load_dataset("third_party/KGQA-datasets/grail_qa", "grailqa_test_public")
        output["test"] = [
            Sample(question=item["question"], sparql="") for item in data["test"]
        ]

    elif args.wqsp:
        data = load_dataset("third_party/KGQA-datasets/webqsp")
        kg = "freebase"
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                for sparql in item["Parses"]["Sparql"]:
                    samples.append(
                        Sample(
                            question=item["RawQuestion"],
                            sparql=clean_sparql_for_wqsp_and_cwq(sparql),
                        )
                    )
                    if split == "test":
                        # only first for test
                        break

            output[split] = samples

    elif args.cwq:
        data = load_dataset(
            "third_party/KGQA-datasets/complex_web_questions",
            "complex_web_questions",
        )
        kg = "freebase"
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                samples.append(
                    Sample(
                        question=item["question"],
                        sparql=clean_sparql_for_wqsp_and_cwq(item["sparql"]),
                    )
                )
            output[split] = samples

        data = load_dataset(
            "third_party/KGQA-datasets/complex_web_questions",
            "complexwebquestions_test",
        )
        output["test"] = [
            Sample(
                question=item["question"],
                sparql=clean_sparql_for_wqsp_and_cwq(item["sparql"]),
            )
            for item in data["test"]
        ]

    elif args.freebase_simple_questions is not None:

        def to_fb_iri(uri: str) -> str:
            prefix = "www.freebase.com/"
            fb_prefix = "http://rdf.freebase.com/ns/"
            assert uri.startswith(prefix)
            uri = uri[len(prefix) :].replace("/", ".")
            return "<" + fb_prefix + uri + ">"

        kg = "freebase"
        for split in ["train", "valid", "test"]:
            data = os.path.join(
                args.freebase_simple_questions, f"annotated_fb_data_{split}.txt"
            )
            with open(data, "r") as inf:
                lines = inf.readlines()

            split = SPLIT_RENAME[split]
            output[split] = []
            for line in lines:
                subj, prop, obj, question = line.split("\t")
                subj = to_fb_iri(subj)
                prop = to_fb_iri(prop)
                sparql = f"SELECT ?x WHERE {{ {subj} {prop} ?x . }}"
                sample = Sample(
                    question=question.strip(),
                    sparql=sparql,
                    info={"object": to_fb_iri(obj)},
                )
                output[split].append(sample)

    elif args.cfq is not None:
        kg = "freebase"
        split = os.path.join(args.cfq, "splits", "random_split.json")
        dataset = os.path.join(args.cfq, "dataset.json")
        with open(split, "r") as inf:
            split = json.load(inf)

        with open(dataset, "r") as inf:
            data = json.load(inf)

        for split in ["train", "dev", "test"]:
            indices = split[f"{split}Idxs"]
            split = SPLIT_RENAME[split]
            samples = []
            for idx in indices:
                item = data[idx]
                samples.append(
                    Sample(
                        question=item["question"],
                        sparql=item["sparql"],
                    )
                )
            output[split] = samples

    elif args.dbpedia_simple_questions is not None:

        def build_sparql(subj: str, pred_list: list[dict]) -> str:
            assert pred_list
            subj = "<" + subj + ">"
            blocks = []
            for pred in pred_list:
                p = pred["Predicate"]
                fwd = pred["Direction"] == "forward"
                const = pred["Constraint"]
                if fwd:
                    block = f"{subj} <{p}> ?x ."
                else:
                    block = f"?x <{p}> {subj} ."

                if const:
                    block += f" ?x a <{const}> ."

                blocks.append(block)

            blocks = " } UNION { ".join(blocks)
            return f"SELECT DISTINCT ?x WHERE {{ {{ {blocks} }} }}"

        kg = "dbpedia"
        for split in ["train", "valid", "test"]:
            with open(
                os.path.join(args.dbpedia_simple_questions, f"{split}.json"),
                "r",
            ) as inf:
                items = json.load(inf)["Questions"]

            split = SPLIT_RENAME[split]

            output[split] = []
            for item in items:
                sample = Sample(
                    id=item["ID"],
                    question=item["Query"],
                    sparql=build_sparql(item["Subject"], item["PredicateList"]),
                )
                output[split].append(sample)

    elif args.lc_quad1_dbpedia:
        kg = "dbpedia"
        data = load_dataset("third_party/KGQA-datasets/lcquad_v1", "lcquad")
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                samples.append(
                    Sample(
                        question=item["corrected_question"].replace(" ?", "?"),
                        sparql=item["sparql_query"].replace(
                            "COUNT(?uri)",
                            "(COUNT(?uri) AS ?count)",  # fixes count issues
                        ),
                    )
                )
            output[split] = samples

    elif args.qald_9:
        kg = "dbpedia"
        data = load_dataset("third_party/KGQA-datasets/qald/qald-9.py")
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                sample = get_qald_sample(item)
                if sample is None:
                    continue

                # fix some issues with xsd:date
                sample.sparql = re.sub(
                    r"SELECT DISTINCT xsd:date\((.*?)\) ",
                    lambda m: f"SELECT DISTINCT (xsd:date({m.group(1)}) AS ?answer) ",
                    sample.sparql,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                # same for count queries
                sample.sparql = re.sub(
                    r"SELECT \(?COUNT\((.*?)\) AS (.*?)\)? ",
                    lambda m: f"SELECT (COUNT({m.group(1)}) AS {m.group(2)}) ",
                    sample.sparql,
                    flags=re.DOTALL | re.IGNORECASE,
                )
                sample.sparql = re.sub(
                    r"SELECT COUNT\(DISTINCT (.*?) AS (.*?)\) ",
                    lambda m: f"SELECT (COUNT(DISTINCT {m.group(1)}) AS {m.group(2)}) ",
                    sample.sparql,
                    flags=re.DOTALL | re.IGNORECASE,
                )

                samples.append(sample)

            output[split] = samples

    elif args.dblp_quad:
        kg = "dblp"
        data = load_dataset("awalesushil/DBLP-QuAD")
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                id = item.pop("id")
                question = item.pop("question")
                sparql = item.pop("query")
                paraphrase = item.pop("paraphrased_question")
                samples.append(
                    Sample(
                        id=id,
                        question=question["string"],
                        sparql=sparql["sparql"],
                        paraphrases=[paraphrase["string"]],
                        info=item,
                    )
                )

            output[split] = samples

    elif args.sci_qa:
        kg = "orkg"
        data = load_dataset("orkg/SciQA")
        for split, items in data.items():
            split = SPLIT_RENAME[split]
            samples = []
            for item in items:
                samples.append(
                    Sample(
                        question=item["question"]["string"],
                        sparql=item["query"]["sparql"],
                        paraphrases=item["paraphrased_question"],
                    )
                )

            output[split] = samples

    elif args.spinach:
        kg = "wikidata"

        for split, org_split in [("train", "dev"), ("test", "test")]:
            with open(os.path.join(args.spinach, f"{org_split}.json"), "r") as inf:
                data = json.load(inf)

            samples = []
            for item in data:
                question = item.pop("question")
                sparql = item.pop("sparql")
                samples.append(
                    Sample(
                        question=question,
                        sparql=sparql,
                        info=item,
                    )
                )

            output[split] = samples

    else:
        raise RuntimeError("Unknown dataset")

    return kg, output


def prepare(args: argparse.Namespace):
    kg, benchmark = load_benchmark(args)
    num_samples = {s: len(samples) for s, samples in benchmark.items()}
    print(f"Number of raw samples: {num_samples}")

    os.makedirs(args.out_dir, exist_ok=True)

    manager = load_kg_manager(kg)

    for split, samples in benchmark.items():
        out = os.path.join(args.out_dir, f"{split}.jsonl")
        if os.path.exists(out) and not args.overwrite:
            print(f"Skipping {split} split because it already exists")
            continue

        with open(out, "w") as rf:
            for i, sample in enumerate(samples):
                if sample.id is None:
                    sample.id = f"{split}_{i}"

                try:
                    sample.sparql = manager.fix_prefixes(sample.sparql)
                except Exception as e:
                    print(
                        f"Error fixing prefixes in sample {sample.id} "
                        f"with sparql\n{sample.sparql}:\n{e}"
                    )

                rf.write(sample.model_dump_json() + "\n")

        print(f"Got {len(samples):,} {split} samples")


if __name__ == "__main__":
    prepare(parse_args())
