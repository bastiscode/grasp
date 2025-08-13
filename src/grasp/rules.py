def general_rules() -> list[str]:
    return [
        "Explain your thought process before and after each step \
and function call.",
        "Do not just use or make up entity or property identifiers \
without verifying their existence in the knowledge graphs first.",
    ]


def task_rules(task: str) -> list[str]:
    if task == "sparql-qa":
        return [
            "Always execute your final SPARQL query before giving an answer to \
make sure it returns the expected results.",
            "The SPARQL query should always return the actual \
identifiers / IRIs of the items in its result. It additionally may return \
labels or other human-readable information, but they are optional and should be \
put within optional clauses unless explicitly requested by the user.",
            "Do not stop early if there are still obvious improvements to be made \
to the SPARQL query. For example, keep refining your SPARQL query if its result \
contains irrelevant items or is missing items you expected.",
            "Do not perform additional computation (e.g. filtering, sorting, calculations) \
on the result of the SPARQL query to determine the answer. All computation should \
be done solely within SPARQL.",
            'For questions with a "True" or "False" answer the SPARQL query \
should be an ASK query.',
            "Do not use 'SERVICE wikibase:label { bd:serviceParam wikibase:language ...' \
in SPARQL queries. It is not SPARQL standard and unsupported by the used QLever \
SPARQL endpoints. Use rdfs:label or similar properties to get labels instead.",
        ]

    elif task == "general-qa":
        return [
            "Your answers preferably should be based on the information available in the \
knowledge graphs. If you do not need them to answer the question, e.g. if \
you know the answer by heart, still try to verify it with the knowledge graphs.",
            "Do not use 'SERVICE wikibase:label { bd:serviceParam wikibase:language ...' \
in SPARQL queries. It is not SPARQL standard and unsupported by the used QLever \
SPARQL endpoints. Use rdfs:label or similar properties to get labels instead.",
        ]

    else:
        raise ValueError(f"Unknown task {task}")
