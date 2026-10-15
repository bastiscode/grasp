import pytest

from grasp.configs import GraspConfig
from grasp.model import Message
from grasp.model.base import Response, ToolCall
from grasp.tasks.sparql_qa import SparqlQaTask
from grasp.utils import FunctionCallException


class FakeManager:
    """Stand-in for KgManager with the real parsers.

    Only fix_prefixes is stubbed to identity -- the normalization under test
    (comments, whitespace) comes from the real SPARQL parser, and check_known
    needs the real iri/literal parser and a prefix map, so those are real too.
    """

    kg = "wikidata"

    def __init__(self):
        from grasp.manager.utils import get_common_sparql_prefixes
        from grasp.sparql.utils import (
            load_iri_and_literal_parser,
            load_sparql_parser,
        )

        self.sparql_parser = load_sparql_parser()
        self.iri_literal_parser = load_iri_and_literal_parser()
        self.prefixes = dict(get_common_sparql_prefixes())
        self.prefixes["wd"] = "http://www.wikidata.org/entity/"

    def fix_prefixes(self, sparql: str) -> str:
        return sparql

    def format_iri(self, iri: str) -> str:
        longest = self.find_longest_prefix(iri)
        if longest is None:
            return f"<{iri}>"
        short, long = longest
        return f"{short}:{iri[len(long):]}"

    def find_longest_prefix(self, iri: str):
        best = None
        for short, long in self.prefixes.items():
            if iri.startswith(long) and (best is None or len(long) > len(best[1])):
                best = (short, long)
        return best


def task(**task_kwargs) -> SparqlQaTask:
    """Task with ONLY the execute gate on, unless overridden.

    know_before_answer is a separate gate with its own tests; leaving it on here
    would make every execute-gate test also depend on IRI bookkeeping.
    """
    config = GraspConfig(model="m")
    config.task_kwargs["sparql-qa"] = {
        "execute_before_answer": True,
        "know_before_answer": False,
        **task_kwargs,
    }
    t = SparqlQaTask(managers=[FakeManager()], config=config)  # type: ignore[list-item]
    return t


Q = "PREFIX wd: <http://www.wikidata.org/entity/> SELECT ?x WHERE { wd:Q1 ?p ?x }"


def executed(t: SparqlQaTask, sparql: str, error: str | None = None) -> None:
    """Append an assistant message with an `execute` call, as the loop would."""
    call = ToolCall(
        id="c", name="execute", args={"kg": "wikidata", "sparql": sparql},
        result=None if error else "Got 1 row", error=error,
    )
    t.messages.append(
        Message.assistant(Response(id="r", message="running it", tool_calls=[call]))
    )


def answer_args(sparql: str = Q) -> dict:
    return {"kg": "wikidata", "sparql": sparql, "answer": "some answer"}


def test_answering_without_executing_is_rejected():
    t = task()
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args(), set(), None)
    assert "has never been executed" in str(e.value)
    # rejected means the loop must NOT stop, so the model gets another round
    assert t.answer_rejected
    assert not t.done("answer")


def test_executing_the_submitted_query_is_accepted():
    t = task()
    executed(t, Q)
    assert t.call_function("answer", answer_args(), set(), None) == "Stopping"
    assert not t.answer_rejected
    assert t.done("answer")


def test_executing_a_different_query_is_rejected():
    t = task()
    other = "SELECT ?y WHERE { ?y ?p ?o }"
    executed(t, other)
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args(), set(), None)
    assert "not the one you just executed" in str(e.value)
    assert t.answer_rejected


def test_only_the_most_recent_execute_counts():
    # the gate is about the model looking at the result before deciding, so an
    # execute of Q that has since been superseded no longer vouches for Q --
    # its result is now buried above a newer one
    t = task()
    executed(t, Q)
    executed(t, "SELECT ?z WHERE { ?z ?p ?o }")
    with pytest.raises(FunctionCallException):
        t.call_function("answer", answer_args(Q), set(), None)


def test_an_intervening_non_execute_call_invalidates_the_execute():
    # strictly positional: the result must be the last thing in the context
    # before the decision, so a search after executing Q pushes Q's result up
    # and Q has to be re-run
    t = task()
    executed(t, Q)
    search = ToolCall(
        id="s", name="search", args={"kg": "wikidata", "query": "Q1"},
        result="some entities", error=None,
    )
    t.messages.append(
        Message.assistant(Response(id="r2", message="checking", tool_calls=[search]))
    )
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args(Q), set(), None)
    # and it must say so: telling a model that DID run the query that it was
    # never executed reads as false and gives it nothing to act on
    msg = str(e.value)
    assert "You did execute" in msg and "another function was called since" in msg


def test_a_client_side_execute_error_does_not_count():
    # the model's own mistake: it never saw a result, and re-running a fixed
    # query is a real path forward, so make it do that
    t = task()
    executed(t, Q, error="Failed to parse SPARQL query: unexpected token")
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args(), set(), None)
    assert "returned an error" in str(e.value)


def test_a_backend_timeout_does_NOT_block_the_answer():
    # not the model's mistake, and re-running usually times out again -- the
    # gate must not trap the episode in a retry loop it cannot escape
    t = task()
    executed(t, Q, error="SPARQL query timed out after 30s")
    assert t.call_function("answer", answer_args(), set(), None) == "Stopping"


def test_a_server_error_does_NOT_block_the_answer():
    t = task()
    executed(t, Q, error="503 Server Error: Service Unavailable")
    assert t.call_function("answer", answer_args(), set(), None) == "Stopping"


def test_a_tolerated_error_still_has_to_be_the_right_query():
    # tolerating backend failures must not become "any timeout unlocks answer"
    t = task()
    executed(t, "SELECT ?z WHERE { ?z ?p ?o }", error="SPARQL query timed out")
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args(Q), set(), None)
    assert "not the one you just executed" in str(e.value)


def test_a_failed_execute_in_between_also_invalidates():
    # same positional rule: whatever went wrong with the last execute, the
    # thing directly above the answer is a different query
    t = task()
    executed(t, Q)
    executed(t, "SELECT ?z WHERE { ?z ?p ?o }", error="timeout")
    with pytest.raises(FunctionCallException):
        t.call_function("answer", answer_args(Q), set(), None)


def test_execute_and_answer_in_the_same_response_is_accepted():
    # the loop dispatches tool calls in order, so by the time answer is checked
    # the execute next to it already has its result -- while answer itself is
    # still undispatched and must not be mistaken for the preceding call
    t = task()
    ran = ToolCall(
        id="e", name="execute", args={"kg": "wikidata", "sparql": Q},
        result="Got 1 row", error=None,
    )
    pending = ToolCall(
        id="a", name="answer", args=answer_args(Q), result=None, error=None
    )
    t.messages.append(
        Message.assistant(
            Response(id="r", message="run and answer", tool_calls=[ran, pending])
        )
    )
    assert t.call_function("answer", answer_args(Q), set(), None) == "Stopping"


def test_comments_and_whitespace_do_not_matter():
    t = task()
    messy = """
        PREFIX wd: <http://www.wikidata.org/entity/>
        # find everything about Q1
        SELECT   ?x
        WHERE {
            wd:Q1 ?p ?x    # any property
        }
    """
    executed(t, messy)
    # submitted as a single tidy line -- must still count as the same query
    assert t.call_function("answer", answer_args(Q), set(), None) == "Stopping"


def test_variable_renaming_is_NOT_tolerated():
    # deliberate: this gate checks "you ran exactly this", not "you ran
    # something equivalent". Renaming variables produces a different query.
    t = task()
    executed(t, Q.replace("?x", "?value"))
    with pytest.raises(FunctionCallException):
        t.call_function("answer", answer_args(Q), set(), None)


def test_executed_queries_do_not_leak_between_episodes():
    # the message list must be per-instance; a class-level default would let
    # one episode's executes satisfy the gate for a different episode
    first = task()
    executed(first, Q)
    second = task()
    assert second.dispatched_tool_calls() == []
    with pytest.raises(FunctionCallException):
        second.call_function("answer", answer_args(), set(), None)


def test_cancel_with_a_best_attempt_is_gated_too():
    # best_attempt earns f1 exactly like an answer, so leaving cancel open lets
    # RL route around the gate instead of learning to execute
    t = task()
    args = {"explanation": "giving up", "best_attempt": {"kg": "wikidata", "sparql": Q}}
    with pytest.raises(FunctionCallException) as e:
        t.call_function("cancel", args, set(), None)
    assert t.answer_rejected
    assert not t.done("cancel")
    # the way out must be spelled out, or a stuck model has nowhere to go
    assert "without a best attempt" in str(e.value)


def test_cancel_without_a_best_attempt_is_always_allowed():
    # the escape hatch that makes gating cancel safe: a model that cannot get
    # its query to run can still stop cleanly, it just earns no f1 for it
    t = task()
    assert t.call_function("cancel", {"explanation": "no idea"}, set(), None) == "Stopping"
    assert not t.answer_rejected
    assert t.done("cancel")


def test_cancel_with_an_executed_best_attempt_is_accepted():
    t = task()
    executed(t, Q)
    args = {"explanation": "close enough", "best_attempt": {"kg": "wikidata", "sparql": Q}}
    assert t.call_function("cancel", args, set(), None) == "Stopping"
    assert t.done("cancel")


def test_unparseable_submitted_query_is_rejected_with_a_parse_message():
    t = task()
    executed(t, Q)
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args("NOT SPARQL AT ALL"), set(), None)
    assert "could not be parsed" in str(e.value)


def test_the_gate_is_opt_in():
    # every other sparql-qa caller (webapp, evals, non-CQD tasks) must be
    # unaffected until a run asks for the gate by name
    config = GraspConfig(model="m")
    config.task_kwargs["sparql-qa"] = {"know_before_answer": False}
    t = SparqlQaTask(managers=[FakeManager()], config=config)  # type: ignore[list-item]
    assert t.call_function("answer", answer_args(), set(), None) == "Stopping"


def test_the_gate_can_be_switched_off():
    t = task(execute_before_answer=False, know_before_answer=False)
    assert t.call_function("answer", answer_args(), set(), None) == "Stopping"


def test_know_before_answer_still_runs_when_execute_gate_is_off():
    # the two gates are independent: turning one off must not disable the other
    t = task(execute_before_answer=False, know_before_answer=True)
    with pytest.raises(FunctionCallException) as e:
        t.call_function("answer", answer_args(), set(), None)
    # wd:Q1 was never seen in a function result, so the KNOWN check is what fires
    assert "without being known" in str(e.value)
