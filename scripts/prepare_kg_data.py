import argparse
import csv
import os
import sys
from typing import Iterator
from urllib.parse import unquote_plus

from search_index import IndexData, Mapping

from grasp.sparql.manager import load_kg_prefixes
from grasp.sparql.sparql import (
    find_longest_prefix,
    get_index_dir,
    is_iri,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kg",
        type=str,
        default=None,
        help="Specify knowledge graph for label fallback",
    )
    parser.add_argument(
        "--add-id-as-synonym",
        action="store_true",
        help="Add object id as synonym if label is not empty, "
        "otherwise we already use it as main label",
    )
    parser.add_argument(
        "--osm-planet-entities",
        action="store_true",
        help="Inputs are OSM Planet entities",
    )
    return parser.parse_args()


def split_iri(iri: str) -> tuple[str, str]:
    if not is_iri(iri):
        return "", iri

    # split iri into prefix and last part after final / or #
    last_hashtag = iri.rfind("#")
    last_slash = iri.rfind("/")
    last = max(last_hashtag, last_slash)
    if last == -1:
        return "", iri[1:-1]
    else:
        return iri[1:last], iri[last + 1 : -1]


def camel_case_split(s: str) -> str:
    # split camelCase into words
    # find uppercase letters
    words = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper() and i > 0 and s[i - 1].islower():
            words.append(s[last:i])
            last = i

    if last < len(s):
        words.append(s[last:])

    return " ".join(words)


PREFIXES = None


def get_object_name_from_id(kg: str, obj_id: str) -> str:
    global PREFIXES
    if PREFIXES is None:
        PREFIXES = load_kg_prefixes(kg)

    pfx = find_longest_prefix(obj_id, PREFIXES)
    if pfx is None:
        # no known prefix, split after final / or # to get objet name
        _, obj_name = split_iri(obj_id)
    else:
        _, long = pfx
        obj_name = obj_id[len(long) : -1]

    # url decode the object name
    return unquote_plus(obj_name)


def get_label_from_id(kg: str, obj_id: str) -> str:
    obj_name = get_object_name_from_id(kg, obj_id)
    label = " ".join(camel_case_split(part) for part in split_at_punctuation(obj_name))
    return label.strip()


# we consider _, -, and . as url punctuation
PUNCTUATION = {"_", "-", "."}


def split_at_punctuation(s: str) -> Iterator[str]:
    start = 0
    for i, c in enumerate(s):
        if c not in PUNCTUATION:
            continue

        yield s[start:i]
        start = i + 1

    if start < len(s):
        yield s[start:]


WD_DATA: IndexData | None = None
WD_MAP: Mapping | None = None


def get_osm_planet_score_from_wikidata_id(wid: str) -> str:
    global WD_DATA, WD_MAP
    if WD_DATA is None or WD_MAP is None:
        index_dir = get_index_dir()
        assert index_dir is not None, "KG_INDEX_DIR environment variable not set"
        data_file = os.path.join(index_dir, "wikidata", "entities", "data.tsv")
        offsets_file = os.path.join(index_dir, "wikidata", "entities", "offsets.bin")
        WD_DATA = IndexData.load(data_file, offsets_file)
        mapping_file = os.path.join(index_dir, "wikidata", "entities", "mapping.bin")
        WD_MAP = Mapping.load(WD_DATA, mapping_file)

    id = WD_MAP.get(wid)
    if id is None:
        return ""

    # score is in the second column
    score = WD_DATA.get_val(id, 1)
    assert score is not None, f"no score for {wid}"
    return score


def clean(s: str) -> str:
    return " ".join(s.split())


if __name__ == "__main__":
    args = parse_args()

    reader = csv.reader(sys.stdin)

    # skip header
    next(reader)

    # write Header
    print("\t".join(["label", "score", "synonyms", "id", "infos"]))

    for row in reader:
        # remove \n and \t from each column
        row = [clean(col) for col in row]

        try:
            label, score, syns, id, infos = row
        except Exception as e:
            print(f"Malformed row: {row}", file=sys.stderr)
            print(f"Error: {e}", file=sys.stderr)
            continue

        # add brackets to id
        id = f"<{id}>"

        if not label:
            # label is empty, try to get it from the object id
            if args.kg is not None:
                label = get_label_from_id(args.kg, id)
            elif syns:
                # use the first synonym as label
                # keep rest of synonyms
                first, *rest = syns.split(";;;")
                syns = ";;;".join(rest)
            else:
                raise ValueError(
                    f"Label is empty and no ID fallback or synonyms provided: {row}"
                )

        elif args.add_id_as_synonym:
            # add id of item to synonyms
            object_name = get_object_name_from_id(args.kg, id)
            if syns:
                syns = f"{syns};;;{object_name}"
            else:
                syns = object_name

        if args.osm_planet_entities:
            # for osm planet entities, score is a wikidata id
            wid = f"<{score}>"
            score = get_osm_planet_score_from_wikidata_id(wid)

        score = "0" if not score else score

        print("\t".join([label, score, syns, id, infos]))
