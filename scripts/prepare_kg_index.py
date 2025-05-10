import argparse
import os
import sys

from search_index import IndexData, Mapping

from grasp.sparql.manager import load_kg_prefixes
from grasp.sparql.sparql import (
    find_longest_prefix,
    get_index_dir,
    is_fq_iri,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    prop_type_group = parser.add_mutually_exclusive_group()
    prop_type_group.add_argument("--dblp-properties", action="store_true")
    prop_type_group.add_argument("--uniprot", action="store_true")
    prop_type_group.add_argument("--osm-planet-entities", action="store_true")
    prop_type_group.add_argument("--osm-planet-properties", action="store_true")
    prop_type_group.add_argument("--imdb-properties", action="store_true")
    return parser.parse_args()


def fix(s: str) -> str:
    return s.replace(r"\n", " ").replace(r"\t", " ").strip()


def format_label(s: str) -> str:
    if s.startswith('"') and s.rfind('"@') != -1:
        # literal with language tag
        return fix(s[1 : s.rfind('"@')])
    elif s.startswith('"') and s.rfind('"^^') != -1:
        # literal with datatype
        return fix(s[1 : s.rfind('"^^')])
    elif s.startswith('"') and s.endswith('"'):
        return fix(s[1:-1])
    else:
        return fix(s)


def split_iri(iri: str) -> tuple[str, str]:
    if not is_fq_iri(iri):
        return "", iri

    # split iri into prefix and last part after final / or #
    last_hashtag = iri.rfind("#")
    last_slash = iri.rfind("/")
    if last_hashtag == -1 and last_slash == -1:
        return iri[1:-1]
    elif last_hashtag > last_slash:
        return iri[1:last_hashtag], iri[last_hashtag + 1 : -1]
    else:
        return iri[1:last_slash], iri[last_slash + 1 : -1]


def camel_case_split(s: str) -> str:
    # split camelCase into words
    # find uppercase letters
    words = []
    last = 0
    for i, c in enumerate(s):
        if c.isupper() and i > last and s[i - 1].islower():
            words.append(s[last:i].lower())
            last = i

    if last < len(s):
        words.append(s[last:].lower())

    return " ".join(words)


PREFIXES = None


def get_label_from_camel_case_id(kg: str, obj_id: str) -> str:
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

    return camel_case_split(obj_name)


WD_DATA: IndexData | None = None
WD_MAP: Mapping | None = None


def get_osm_planet_score_from_wikidata_id(wd_id: str) -> str:
    global WD_DATA, WD_MAP
    if WD_DATA is None or WD_MAP is None:
        index_dir = get_index_dir()
        assert index_dir is not None, "KG_INDEX_DIR environment variable not set"
        data_file = os.path.join(index_dir, "wikidata", "entities", "data.tsv")
        WD_DATA = IndexData(data_file)
        mapping_file = os.path.join(index_dir, "wikidata", "entities", "mapping.bin")
        WD_MAP = Mapping.load(WD_DATA, mapping_file)

    id = WD_MAP.get(wd_id)
    if id is None:
        return ""

    # score is in the second column
    score = WD_DATA.get_val(id, 1)
    assert score is not None, f"no score for {wd_id}"
    return score


if __name__ == "__main__":
    args = parse_args()
    header = next(sys.stdin)
    print("\t".join(field[1:] for field in header.rstrip("\r\n").split("\t")))

    for line in sys.stdin:
        try:
            (label, score, syns, obj_id, infos) = line.rstrip("\r\n").split("\t")
        except Exception as e:
            print(f"Error parsing line '{line}': {e}", file=sys.stderr)
            continue

        assert syns.startswith('"') and syns.endswith('"')
        syns = fix(syns[1:-1])
        assert infos.startswith('"') and infos.endswith('"')
        infos = fix(infos[1:-1])

        if label:
            label = format_label(label)
        else:
            # label is empty, try to get it from the object id
            if args.dblp_properties:
                label = get_label_from_camel_case_id("dblp", obj_id)
            elif args.uniprot:
                label = get_label_from_camel_case_id("uniprot", obj_id)
            elif args.osm_planet_properties:
                label = get_label_from_camel_case_id("osm-planet", obj_id)
            elif args.imdb_properties:
                label = get_label_from_camel_case_id("imdb", obj_id)
            elif syns:
                # use the first synonym as label
                # keep rest of synonyms
                first, *rest = syns.split(";;;")
                label = format_label(first)
                syns = ";;;".join(rest)
            else:
                raise ValueError(
                    "Label is empty and no ID fallback or synonyms provided"
                )

        if args.osm_planet_entities:
            # for osm planet entities, score is a wikidata id
            score = get_osm_planet_score_from_wikidata_id(score)

        score = "0" if score == "" else score

        print("\t".join([label, score, syns, obj_id, infos]))
