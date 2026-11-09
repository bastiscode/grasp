from grasp.sparql.types import (
    INDEX_OBJ_TYPES,
    IRI_OBJ_TYPES,
    ObjType,
    Position,
    iri_obj_types,
)


def test_iri_indices_are_the_standard_non_literal_indices():
    assert set(IRI_OBJ_TYPES) == set(INDEX_OBJ_TYPES) - {ObjType.LITERAL}


def test_iri_lookup_order():
    assert iri_obj_types() == [ObjType.CLASS, ObjType.ENTITY, ObjType.PROPERTY]
    assert iri_obj_types(Position.OBJECT) == iri_obj_types()
    assert iri_obj_types(Position.PROPERTY) == [
        ObjType.PROPERTY,
        ObjType.CLASS,
        ObjType.ENTITY,
    ]
