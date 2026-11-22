from typing import Any

from grasp.manager import KgManager, find_manager
from grasp.model import FunctionDefinition, ToolCallResult
from grasp.tasks.construct import (
    construct_config,
    execute_output,
    execute_output_definition,
    format_queries,
    output_manager,
)
from grasp.tasks.construct.checks import Report, format_report
from grasp.tasks.construct.scratch import Scratch
from grasp.tasks.feedback import FeedbackTask
from grasp.utils import format_enumerate, format_section


def system_information(kg: str, others: list[str], agentic: bool) -> str:
    info = f"""\
You are an independent reviewer of SPARQL CONSTRUCT queries that remodel the \
knowledge graph "{kg}" as a well-formed and semantically meaningful knowledge \
graph. The knowledge graph is typically a direct mapping of structured data to \
RDF. The new knowledge graph should have all of its information, but not the \
form it has there: datatypes, names, keys and references given as strings \
often stem from the format of the original data and not from what the data \
means, so that the knowledge graph has something in a certain way does not \
justify keeping it that way. The queries were run on a sample of the knowledge \
graph, which has some instances of every class together with what they link \
to. You are given the queries and the results of automatic checks of their \
output on the sample."""
    if agentic:
        info += """ \
Query the output on the sample with execute_output, and the knowledge graph \
itself with the other functions. Counts differ between the two, because the \
output only covers the sample."""

    info += """

Review the following aspects:
- Coverage: All information of the knowledge graph is carried over. Verify the \
explanations given for properties without values in the output: a property is \
only redundant if all of its information is in the output in another way, \
compare its values in the knowledge graph with what it is said to duplicate. \
Look into properties with missing values, which can point to values dropped by \
a filter or a failed conversion, and into UNION branches that match nothing, \
which point to wrong patterns.
- Entities: The classes fit what the entities are, and every entity has one \
IRI that is built in the same way in all queries. Compare the instances per \
class in the sample and in the output for lost or merged entities. Things \
that several entities share or that can have properties of their own are \
entities, also if the knowledge graph only has them as strings.
- Links: References to other entities are links to their IRIs and not \
literals, the links resolve to entities in the output, and they point in the \
right direction.
- Values: Literals have the datatypes that fit what they mean, markers for \
missing values are not in the output, and information from property names \
like units is kept.
- Vocabulary: Classes and properties are named after what they mean, are \
described at least by a label, and the same concept is not modeled in two \
ways. For each property, look at some of its triples in the output with the \
labels of subject and object, and verify that its name and direction say what \
the data says."""
    if others:
        info += f"""
- Existing vocabulary: The other knowledge graphs ({", ".join(others)}) \
provide existing classes and properties, and the checks say which ones of the \
output they define. Verify that these are used according to their \
definitions, domains and ranges, and search the other knowledge graphs for \
the classes and properties that the queries define themselves: one that \
duplicates an existing one is an issue."""
    if agentic:
        info += """
- Competency questions, if given: Write and execute a query on the output for \
each of them. A question that cannot be answered, or only with an overly \
complex query, points to a gap in the model. The questions are only examples, \
the model must not be narrowed to them."""

    info += """

Provide your feedback with the give_feedback function. List the issues you \
found, each with its severity (major or minor), the queries it concerns, and \
the evidence for it. Use the status done if there is no major issue, refine \
for specific fixes, and retry if the overall model is wrong. If there is \
previous feedback, first verify whether its issues are resolved, and do not \
raise new minor issues."""
    return info


class ConstructFeedbackTask(FeedbackTask):
    name = "construct-feedback"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.construct_config = construct_config(self.config)
        self.manager = find_manager(self.managers, self.construct_config.kg)
        self.output_manager: KgManager | None = None

    def system_information(self) -> str:
        others = [manager.kg for manager in self.managers if manager != self.manager]
        return system_information(self.manager.kg, others, self.kg_functions)

    def rules(self) -> list[str]:
        return [
            "Do not trust the queries as written. Verify each aspect on the "
            "output and on the knowledge graph, and only report issues you "
            "have verified.",
        ]

    def instructions(self, inputs: list[str], output: dict) -> str:
        sections = [format_section("Queries", format_queries(output["queries"]))]

        if output["report"] is not None:
            report = Report.model_validate(output["report"])
            sections.append(
                format_section(
                    "Checks of the output on the sample",
                    format_report(report),
                )
            )

        questions = self.construct_config.competency_questions
        if questions:
            sections.append(
                format_section("Competency questions", format_enumerate(questions))
            )

        previous = [inpt for inpt in inputs if inpt.startswith("Feedback (status=")]
        if previous:
            sections.append(format_section("Previous feedback", "\n\n".join(previous)))

        return "\n\n".join(sections)

    def setup(self, input: Any) -> str:
        scratch = Scratch(
            self.construct_config.scratch_endpoint,
            self.construct_config.timeout,
        )
        self.output_manager = output_manager(
            scratch,
            self.manager,
            {query["name"]: query["sparql"] for query in input["output"]["queries"]},
        )
        return super().setup(input)

    def function_definitions(self) -> list[FunctionDefinition]:
        fns = super().function_definitions()
        if self.kg_functions:
            # also the output can only be queried with more than one step
            fns.append(execute_output_definition())
        return fns

    def call_function(
        self,
        fn_name: str,
        fn_args: dict,
        known: set[str],
        example_indices: dict | None,
    ) -> str | ToolCallResult:
        if fn_name == "execute_output":
            assert self.output_manager is not None
            return execute_output(self.output_manager, self.config, fn_args)

        return super().call_function(fn_name, fn_args, known, example_indices)
