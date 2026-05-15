from unittest.mock import Mock

from grasp.build.shapes import (
    ConceptProfile,
    PropertyProfile,
    ShapeConfig,
    assemble_profile,
    cardinality_tag,
    compute_shape,
    emit_pseudo_shex,
    get_rows,
)
from grasp.manager import KgManager
from grasp.shapes import ShapeSample
from grasp.sparql.types import AskResult, SelectResult


def make_manager() -> Mock:
    m = Mock(spec=KgManager)
    m.format_iri.side_effect = lambda iri, **_: iri
    m.get_label.return_value = None
    m.try_get_data.return_value = None
    m.prefixes = {}
    return m


def make_labelled_manager(
    entity_labels: dict[str, str] | None = None,
    property_labels: dict[str, str] | None = None,
) -> Mock:
    m = Mock(spec=KgManager)
    m.format_iri.side_effect = lambda iri, **_: iri
    m.prefixes = {}

    _entity_labels = entity_labels or {}
    _property_labels = property_labels or {}

    def get_label(iri: str, index: str) -> str | None:
        if index == "entities":
            return _entity_labels.get(iri)
        if index == "properties":
            return _property_labels.get(iri)
        return None

    m.get_label.side_effect = get_label
    m.try_get_data.return_value = object()  # non-None signals index exists
    return m


def select(vars: list[str], bindings: list[dict]) -> SelectResult:
    return SelectResult.from_json(
        {"head": {"vars": vars}, "results": {"bindings": bindings}}
    )


def uri(value: str) -> dict:
    return {"type": "uri", "value": value}


def literal(value: str, datatype: str | None = None) -> dict:
    d: dict = {"type": "literal", "value": value}
    if datatype:
        d["datatype"] = datatype
    return d


class TestCardinality:
    def test_mandatory_single(self):
        assert cardinality_tag(0.95, 1.05) == ""

    def test_optional_single(self):
        assert cardinality_tag(0.50, 1.05) == "?"

    def test_mandatory_multi(self):
        assert cardinality_tag(0.95, 2.0) == "+"

    def test_optional_multi(self):
        assert cardinality_tag(0.50, 2.0) == "*"


class TestGetRows:
    def test_select_result(self):
        result = select(
            ["p", "tripleCount"],
            [
                {"p": uri("http://ex.org/p1"), "tripleCount": literal("10")},
                {"p": uri("http://ex.org/p2"), "tripleCount": literal("5")},
            ],
        )
        rows = get_rows(result)
        assert rows == [
            {"p": "http://ex.org/p1", "tripleCount": "10"},
            {"p": "http://ex.org/p2", "tripleCount": "5"},
        ]

    def test_ask_result(self):
        assert get_rows(AskResult(boolean=True)) == []


class TestAssembleProfile:
    def test_basic_profile(self):
        manager = make_manager()
        freq_map = {
            "http://ex.org/name": {"triple_count": 900, "entity_count": 900},
            "http://ex.org/type": {"triple_count": 1000, "entity_count": 1000},
        }
        lit_dtypes = {"http://ex.org/name": ["xsd:string"]}
        range_map = {"http://ex.org/type": ["http://ex.org/Class"]}
        shape_config = ShapeConfig(min_property_coverage=0.01)

        profile = assemble_profile(
            "http://ex.org/Human",
            freq_map,
            lit_dtypes,
            range_map,
            total_entities=1000,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager)

        assert shex == (
            "http://ex.org/Human {\n"
            "  http://ex.org/type [ http://ex.org/Class ] ;\n"
            "  http://ex.org/name xsd:string ;\n"
            "}"
        )

    def test_min_coverage_filters(self):
        manager = make_manager()
        freq_map = {
            "http://ex.org/rare": {"triple_count": 1, "entity_count": 1},
            "http://ex.org/common": {"triple_count": 900, "entity_count": 900},
        }
        shape_config = ShapeConfig(min_property_coverage=0.5)

        profile = assemble_profile(
            "http://ex.org/C",
            freq_map,
            {},
            {},
            total_entities=1000,
            shape_config=shape_config,
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager)

        assert shex == ("http://ex.org/C {\n  http://ex.org/common IRI ;\n}")

    def test_empty_freq_map(self):
        manager = make_manager()
        profile = assemble_profile(
            "http://ex.org/X",
            {},
            {},
            {},
            total_entities=100,
            shape_config=ShapeConfig(),
            manager=manager,
        )
        shex = emit_pseudo_shex(profile, manager)

        assert shex == "http://ex.org/X {\n}"


class TestEmitRetrievalDoc:
    def test_full_output(self):
        manager = make_manager()
        profile = ConceptProfile(
            iri="http://ex.org/Human",
            short_iri="ex:Human",
            total_entities=500,
            properties=[
                PropertyProfile(
                    iri="http://ex.org/name",
                    short_iri="ex:name",
                    triple_count=500,
                    entity_count=500,
                )
            ],
        )
        shex = emit_pseudo_shex(profile, manager)

        assert shex == "ex:Human {\n  ex:name IRI ;\n}"


class TestComputeShape:
    def test_basic(self):
        manager = make_manager()

        freq_result = select(
            ["p", "tripleCount", "entityCount"],
            [
                {
                    "p": uri("http://ex.org/name"),
                    "tripleCount": literal("800"),
                    "entityCount": literal("800"),
                }
            ],
        )
        lit_result = select(
            ["p", "datatype"],
            [
                {
                    "p": uri("http://ex.org/name"),
                    "datatype": uri("http://www.w3.org/2001/XMLSchema#string"),
                }
            ],
        )
        range_result = select(["p", "targetClass"], [])
        total_result = select(["totalEntities"], [{"totalEntities": literal("1000")}])
        manager.execute_sparql.side_effect = [
            freq_result,
            lit_result,
            range_result,
            total_result,
        ]

        doc = compute_shape(
            "http://ex.org/Human", "?instance wdt:P31 {CLASS} .", manager, ShapeConfig()
        )

        assert doc == (
            "http://ex.org/Human {\n"
            "  http://ex.org/name http://www.w3.org/2001/XMLSchema#string ? ;\n"
            "}"
        )

    def test_query_failure_returns_none(self):
        manager = make_manager()
        manager.execute_sparql.side_effect = Exception("timeout")

        assert (
            compute_shape("http://ex.org/X", "?instance a {CLASS} .", manager) is None
        )


def _make_profile(with_target_iris: bool = True) -> ConceptProfile:
    target_iris = ["http://ex.org/Class"] if with_target_iris else []
    return ConceptProfile(
        iri="http://ex.org/Human",
        short_iri="ex:Human",
        total_entities=500,
        properties=[
            PropertyProfile(
                iri="http://ex.org/type",
                short_iri="ex:type",
                triple_count=500,
                entity_count=500,
                target_class_iris=target_iris,
                target_class_short_iris=["ex:Class"] if target_iris else [],
            ),
            PropertyProfile(
                iri="http://ex.org/name",
                short_iri="ex:name",
                triple_count=400,
                entity_count=400,
                literal_datatypes=["xsd:string"],
            ),
        ],
    )


class TestEmitPseudoShexLabelled:
    def test_no_index_falls_back_to_short_iris(self):
        manager = make_manager()
        profile = _make_profile()
        shex = emit_pseudo_shex(profile, manager)
        assert shex == (
            "ex:Human {\n"
            "  ex:type [ ex:Class ] ;\n"
            "  ex:name xsd:string ;\n"
            "}"
        )

    def test_labels_resolved_when_index_available(self):
        manager = make_labelled_manager(
            entity_labels={
                "http://ex.org/Human": "Human",
                "http://ex.org/Class": "MyClass",
            },
            property_labels={
                "http://ex.org/type": "type of",
                "http://ex.org/name": "name",
            },
        )
        profile = _make_profile()
        shex = emit_pseudo_shex(profile, manager)
        assert shex == (
            "Human (ex:Human) {\n"
            "  type of (ex:type) [ MyClass (ex:Class) ] ;\n"
            "  name (ex:name) xsd:string ;\n"
            "}"
        )

    def test_partial_labels_fall_back_to_short_iri(self):
        manager = make_labelled_manager(
            entity_labels={"http://ex.org/Human": "Human"},
        )
        profile = _make_profile()
        shex = emit_pseudo_shex(profile, manager)
        assert shex == (
            "Human (ex:Human) {\n"
            "  ex:type [ ex:Class ] ;\n"
            "  ex:name xsd:string ;\n"
            "}"
        )


class TestShapeSampleQueries:
    def test_label_and_aliases(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            shex="Human (wd:Q5) { ... }",
            label="Human",
            aliases=["person", "human being"],
        )
        assert s.queries() == ["Human", "person", "human being", "wd:Q5"]

    def test_deduplicates(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            shex="wd:Q5 { ... }",
            label="wd:Q5",  # same as short_iri
        )
        assert s.queries() == ["wd:Q5"]

    def test_missing_optional_fields(self):
        s = ShapeSample(iri="http://ex.org/Q5", short_iri="wd:Q5", shex="wd:Q5 { ... }")
        assert s.queries() == ["wd:Q5"]

    def test_backward_compat_extra_kwargs(self):
        s = ShapeSample(
            iri="http://ex.org/Q5",
            short_iri="wd:Q5",
            shex="wd:Q5 { ... }",
            unknown_future_field="ignored",
        )
        assert s.iri == "http://ex.org/Q5"
