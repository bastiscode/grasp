# Expanding property paths in witnesses

Status: implemented in `paths.py` (config `expand_paths`, `max_path_length`).
Shown witnesses get one concrete chain per path triple (`via:` line). Over all
candidate witnesses of a row, a path summary counts the subject-object pairs
per shortest chain length and the first intermediate nodes of longer chains
(e.g. `1 step (53), 2 steps (8); ... first via: video album (8)`), with one
query per chain length for all pairs (VALUES inside each UNION branch, which
QLever needs to join it first).

Without expansion, witnesses show a property path only by its two endpoints, e.g.

    Like a Virgin (wd:Q110977503) --path wdt:P31 / wdt:P279 *--> album (wd:Q482994)

so the model cannot see *why* an entity matched (here: which class in the
subclass chain made it an "album"). In a QALD-10 run (Madonna albums, 2026-10-09)
this hid exactly the intermediate class that would have shown the count was
inflated by `P279*`.

SPARQL has no standard way to return the intermediate nodes of a path, so
expansion means extra queries for one concrete chain between the known
endpoints. Everything below is engine-agnostic.

## Fixed-length parts

Sequences, inverses and alternatives have a fixed length and can be rewritten
into plain triple patterns with fresh intermediate variables:

    ?a wdt:P31/wdt:P279 wd:Q482994
    =>  ?a wdt:P31 ?m1 . ?m1 wdt:P279 wd:Q482994

- `^p`: swap subject and object.
- `p1|p2`: UNION, with branch markers as for UNION in the WHERE clause
  (`Planner.walk_branch`), to report which alternative matched.
- `!(a|b)`: `?s ?p ?o FILTER(?p NOT IN (a, b))`, which yields the concrete
  predicate.

## Repetition (`*`, `+`, `?`)

With both endpoints bound from the witness, use iterative deepening: try chain
lengths 0, 1, 2, ... up to a cap (e.g. 4), one `LIMIT 1` query per length, and
stop at the first hit (shortest chain):

    # length 1
    SELECT * WHERE { <s> wdt:P31 ?c0 . ?c0 wdt:P279 ?c1 . FILTER(?c1 = <o>) } LIMIT 1
    # length 2
    SELECT * WHERE { <s> wdt:P31 ?c0 . ?c0 wdt:P279 ?c1 . ?c1 wdt:P279 <o> } LIMIT 1

- Several repeated elements: enumerate their repetition counts by increasing
  total length.
- `?`: lengths 0 and 1.
- No chain within the cap: report "path longer than N steps".

The witness line would then read

    Like a Virgin --instance of--> <class> --subclass of--> album

## Rejected alternatives

- Client-side BFS from the subject (or bidirectional): can blow up on
  high-degree nodes.
- Engine-specific path functions (Virtuoso `TRANSITIVE ... t_step`, Stardog
  `PATHS`, Blazegraph GAS): not portable, QLever has none.

## Where it would go

- `witness.py`: keep the path's parse subtree in `TriplePattern` (not only its
  text), and add a small path AST (sequence / alternative / inverse / modifier).
- `__init__.py`: an `expand_paths` step after `instantiate`, only for path
  triples of witnesses that are shown, with the witness timeouts.
- Cost: at most rows x witnesses x (cap + 1) small queries with bound endpoints.

No changes to core or other tasks are needed.
