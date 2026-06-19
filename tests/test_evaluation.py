from __future__ import annotations

from fpg import ModelRCAOutput, Scenario, compare_model_to_ground_truth

EVIDENCE = {
    "query": {"language": "sql", "statement": "select 1"},
    "explanation": "observed in telemetry",
}
TIME = {"start": "2026-06-19T00:00:00Z", "end": "2026-06-19T00:01:00Z"}


def _event(node_id: str, subject: str, predicate: str) -> dict[str, object]:
    return {
        "id": node_id,
        "kind": "event",
        "subject": subject,
        "predicate": predicate,
        "time": TIME,
        "grounding": "observed",
        "evidence": [EVIDENCE],
    }


def test_compare_model_to_ground_truth_reports_exact_and_subject_metrics() -> None:
    scenario = Scenario.model_validate(
        {
            "schema_version": "0.1.0",
            "scenario_id": "case-1",
            "testbed": "unit",
            "vocab_version": "core-0.1.0+unit-0.1.0",
            "injections": [
                {
                    "node_id": "seat",
                    "fault_type": "pod_unavailable",
                    "target_entity": "svc:seat",
                    "time": TIME,
                }
            ],
            "graph": {
                "nodes": [
                    _event("seat", "svc:seat", "process_killed"),
                    _event("travel", "svc:travel", "latency_degraded"),
                    _event("ui", "svc:ui", "latency_degraded"),
                ],
                "edges": [
                    {
                        "src": "seat",
                        "dst": "travel",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                    {
                        "src": "travel",
                        "dst": "ui",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                ],
            },
        }
    )
    model = ModelRCAOutput.model_validate(
        {
            "nodes": [
                {
                    "id": "n1",
                    "subject": "svc:seat",
                    "predicate": "process_killed",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
                {
                    "id": "n2",
                    "subject": "svc:travel",
                    "predicate": "error_rate_elevated",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
                {
                    "id": "n3",
                    "subject": "svc:ui",
                    "predicate": "latency_degraded",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
            ],
            "edges": [{"src": "n1", "dst": "n2"}, {"src": "n2", "dst": "n3"}],
            "root_causes": ["n1"],
        }
    )

    comparison = compare_model_to_ground_truth(model, scenario)

    assert comparison.score == 1.0
    assert comparison.root_nodes.f1 == 1.0
    assert comparison.nodes.matched == 2
    assert comparison.nodes.expected == 3
    assert comparison.nodes.predicted == 3
    assert comparison.subjects.f1 == 1.0
    assert comparison.edges.matched == 0
    assert comparison.soft_edges.matched == 0
    assert comparison.subject_edges.matched == 2
    assert comparison.soft_subject_edges.f1 == 1.0
    assert comparison.exact_path_match_hit is False
    assert comparison.subject_path_match_hit is True
    assert comparison.path_reachability_hit is True
    assert comparison.missing_nodes == ["svc:travel|latency_degraded"]
    assert comparison.extra_nodes == ["svc:travel|error_rate_elevated"]


def test_compare_model_to_ground_truth_collapses_gate_edges() -> None:
    scenario = Scenario.model_validate(
        {
            "schema_version": "0.1.0",
            "scenario_id": "case-2",
            "testbed": "unit",
            "vocab_version": "core-0.1.0+unit-0.1.0",
            "injections": [
                {
                    "node_id": "a",
                    "fault_type": "fault_a",
                    "target_entity": "svc:a",
                    "time": TIME,
                },
                {
                    "node_id": "b",
                    "fault_type": "fault_b",
                    "target_entity": "svc:b",
                    "time": TIME,
                },
            ],
            "graph": {
                "nodes": [
                    _event("a", "svc:a", "process_killed"),
                    _event("b", "svc:b", "network_degraded"),
                    {"id": "g", "kind": "gate", "combine": "AND"},
                    _event("c", "svc:c", "latency_degraded"),
                ],
                "edges": [
                    {
                        "src": "a",
                        "dst": "g",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                    {
                        "src": "b",
                        "dst": "g",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                    {
                        "src": "g",
                        "dst": "c",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                ],
            },
        }
    )
    model = ModelRCAOutput.model_validate(
        {
            "nodes": [
                {
                    "id": "n1",
                    "subject": "svc:a",
                    "predicate": "process_killed",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
                {
                    "id": "n2",
                    "subject": "svc:b",
                    "predicate": "network_degraded",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
                {
                    "id": "n3",
                    "subject": "svc:c",
                    "predicate": "latency_degraded",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
            ],
            "edges": [{"src": "n1", "dst": "n3"}, {"src": "n2", "dst": "n3"}],
            "root_causes": ["n1", "n2"],
        }
    )

    comparison = compare_model_to_ground_truth(model, scenario)

    assert comparison.edges.f1 == 1.0
    assert comparison.soft_edges.f1 == 1.0
    assert comparison.soft_subject_edges.f1 == 1.0
    assert comparison.exact_path_match_hit is True
    assert comparison.path_reachability_hit is True
    assert comparison.missing_edges == []
    assert comparison.extra_edges == []


def test_compare_model_to_ground_truth_scores_contracted_edges_softly() -> None:
    scenario = Scenario.model_validate(
        {
            "schema_version": "0.1.0",
            "scenario_id": "case-3",
            "testbed": "unit",
            "vocab_version": "core-0.1.0+unit-0.1.0",
            "injections": [
                {
                    "node_id": "a",
                    "fault_type": "fault_a",
                    "target_entity": "svc:a",
                    "time": TIME,
                }
            ],
            "graph": {
                "nodes": [
                    _event("a", "svc:a", "process_killed"),
                    _event("b", "svc:b", "latency_degraded"),
                    _event("c", "svc:c", "latency_degraded"),
                ],
                "edges": [
                    {
                        "src": "a",
                        "dst": "b",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                    {
                        "src": "b",
                        "dst": "c",
                        "mechanism": "sync_call_blocking",
                        "verification": "consistency-checked",
                    },
                ],
            },
        }
    )
    model = ModelRCAOutput.model_validate(
        {
            "nodes": [
                {
                    "id": "n1",
                    "subject": "svc:a",
                    "predicate": "process_killed",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
                {
                    "id": "n3",
                    "subject": "svc:c",
                    "predicate": "latency_degraded",
                    "time": TIME,
                    "evidence": [EVIDENCE],
                },
            ],
            "edges": [{"src": "n1", "dst": "n3"}],
            "root_causes": ["n1"],
        }
    )

    comparison = compare_model_to_ground_truth(model, scenario)

    assert comparison.edges.matched == 0
    assert comparison.soft_edges.matched == 0.5
    assert comparison.soft_edges.precision == 0.5
    assert comparison.soft_edges.recall == 0.25
    assert comparison.soft_subject_edges.matched == 0.5
    assert comparison.soft_subject_edges.precision == 0.5
    assert comparison.soft_subject_edges.recall == 0.25
    assert comparison.path_reachability_hit is True
