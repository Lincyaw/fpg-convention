"""Set-based comparison between model output and ground-truth FPG graphs."""

from __future__ import annotations

from collections import defaultdict, deque
from enum import Enum
from typing import Any, Mapping

from pydantic import BaseModel, Field

from .model_output import ModelRCAOutput
from .scenario import EventNode, GateNode, Graph, PreconditionNode, Scenario

NodeKey = tuple[str, str]
ExactEdgeKey = tuple[NodeKey, NodeKey]
SubjectEdgeKey = tuple[str, str]
ExactPathKey = tuple[NodeKey, ...]
SubjectPathKey = tuple[str, ...]


class MatchMetrics(BaseModel):
    """Precision/recall/F1 counts for one comparable set."""

    expected: int = Field(ge=0)
    predicted: int = Field(ge=0)
    matched: float = Field(ge=0.0)
    precision: float = Field(ge=0.0, le=1.0)
    recall: float = Field(ge=0.0, le=1.0)
    f1: float = Field(ge=0.0, le=1.0)


class GraphComparison(BaseModel):
    """Comparison result for one ``ModelRCAOutput`` against one ground truth."""

    score: float = Field(
        ge=0.0,
        le=1.0,
        description="Default scalar score: 0.4 root-subject F1 + "
        "0.3 subject F1 + 0.3 soft-subject-edge F1. Predicate-exact "
        "metrics are retained as diagnostics, but do not affect score.",
    )
    root_nodes: MatchMetrics
    root_subjects: MatchMetrics
    nodes: MatchMetrics
    subjects: MatchMetrics
    edges: MatchMetrics
    soft_edges: MatchMetrics
    subject_edges: MatchMetrics
    soft_subject_edges: MatchMetrics
    exact_path_match_hit: bool = False
    subject_path_match_hit: bool = False
    path_reachability_hit: bool = False
    subject_path_reachability_hit: bool = False
    missing_root_nodes: list[str] = Field(default_factory=list)
    extra_root_nodes: list[str] = Field(default_factory=list)
    missing_nodes: list[str] = Field(default_factory=list)
    extra_nodes: list[str] = Field(default_factory=list)
    missing_edges: list[str] = Field(default_factory=list)
    extra_edges: list[str] = Field(default_factory=list)
    missing_subjects: list[str] = Field(default_factory=list)
    extra_subjects: list[str] = Field(default_factory=list)
    missing_subject_edges: list[str] = Field(default_factory=list)
    extra_subject_edges: list[str] = Field(default_factory=list)
    missing_paths: list[str] = Field(default_factory=list)
    extra_paths: list[str] = Field(default_factory=list)
    missing_subject_paths: list[str] = Field(default_factory=list)
    extra_subject_paths: list[str] = Field(default_factory=list)
    missing_path_reachability: list[str] = Field(default_factory=list)
    extra_path_reachability: list[str] = Field(default_factory=list)
    missing_subject_path_reachability: list[str] = Field(default_factory=list)
    extra_subject_path_reachability: list[str] = Field(default_factory=list)


def compare_model_to_ground_truth(
    model_output: ModelRCAOutput | Mapping[str, Any],
    ground_truth: Scenario | Graph | Mapping[str, Any],
    *,
    edge_contraction_gamma: float = 0.5,
) -> GraphComparison:
    """Compare a model-emitted FPG against the corresponding ground truth.

    The comparison is intentionally structural and deterministic:

    - the scalar score ignores predicates and compares affected entities;
    - exact nodes still match on ``(subject, predicate)`` for diagnostics;
    - subject metrics ignore predicates and compare only affected entities;
    - exact edges match on exact endpoint nodes;
    - soft edges give contraction credit when a predicted edge skips
      intermediate ground-truth nodes along a directed path;
    - subject-edge metrics compare only endpoint subjects;
    - exact path hits check whether at least one full root-to-terminal-symptom
      node sequence matches;
    - path reachability hits check whether at least one root/symptom endpoint
      pair is connected in both graphs;
    - model ``root_causes`` are compared against ground-truth source nodes;
    - ground-truth gate nodes are collapsed, because ``ModelRCAOutput`` has
      no gate node type.

    Time-interval overlap and evidence re-execution are deliberately out of
    scope for this helper; callers can layer those verifiers on top.
    """

    if not 0.0 <= edge_contraction_gamma <= 1.0:
        raise ValueError("edge_contraction_gamma must be within [0.0, 1.0]")

    output = _coerce_model_output(model_output)
    graph = _coerce_graph(ground_truth)

    model_node_by_id = {node.id: node for node in output.nodes}

    expected_nodes = _ground_truth_node_keys(graph)
    predicted_nodes = {_model_node_key(node) for node in output.nodes}

    expected_roots = {
        key
        for node in graph.root_causes
        if (key := _ground_truth_node_key(node)) is not None
    }
    predicted_roots = {
        key
        for root_id in output.root_causes
        if (node := model_node_by_id.get(root_id)) is not None
        for key in (_model_node_key(node),)
    }

    expected_edges = _ground_truth_edge_keys(graph)
    predicted_edges = {
        (
            _model_node_key(model_node_by_id[edge.src]),
            _model_node_key(model_node_by_id[edge.dst]),
        )
        for edge in output.edges
    }

    expected_subjects = {subject for subject, _predicate in expected_nodes}
    predicted_subjects = {subject for subject, _predicate in predicted_nodes}
    expected_root_subjects = {subject for subject, _predicate in expected_roots}
    predicted_root_subjects = {subject for subject, _predicate in predicted_roots}
    expected_subject_edges = _subject_edges(expected_edges)
    predicted_subject_edges = _subject_edges(predicted_edges)
    expected_paths = _ground_truth_paths(graph)
    predicted_paths = _model_paths(output)
    expected_subject_paths = _subject_paths(expected_paths)
    predicted_subject_paths = _subject_paths(predicted_paths)
    expected_reachability = _path_reachability(expected_paths)
    predicted_reachability = _path_reachability(predicted_paths)
    expected_subject_reachability = _subject_path_reachability(expected_paths)
    predicted_subject_reachability = _subject_path_reachability(predicted_paths)

    root_metrics = _metrics(predicted_roots, expected_roots)
    root_subject_metrics = _metrics(predicted_root_subjects, expected_root_subjects)
    node_metrics = _metrics(predicted_nodes, expected_nodes)
    subject_metrics = _metrics(predicted_subjects, expected_subjects)
    edge_metrics = _metrics(predicted_edges, expected_edges)
    soft_edge_metrics = _soft_edge_metrics(
        predicted_edges, expected_edges, gamma=edge_contraction_gamma
    )
    subject_edge_metrics = _metrics(predicted_subject_edges, expected_subject_edges)
    soft_subject_edge_metrics = _soft_subject_edge_metrics(
        predicted_subject_edges,
        expected_subject_edges,
        gamma=edge_contraction_gamma,
    )
    score = (
        0.4 * root_subject_metrics.f1
        + 0.3 * subject_metrics.f1
        + 0.3 * soft_subject_edge_metrics.f1
    )

    return GraphComparison(
        score=score,
        root_nodes=root_metrics,
        root_subjects=root_subject_metrics,
        nodes=node_metrics,
        subjects=subject_metrics,
        edges=edge_metrics,
        soft_edges=soft_edge_metrics,
        subject_edges=subject_edge_metrics,
        soft_subject_edges=soft_subject_edge_metrics,
        exact_path_match_hit=bool(predicted_paths & expected_paths),
        subject_path_match_hit=bool(predicted_subject_paths & expected_subject_paths),
        path_reachability_hit=bool(predicted_reachability & expected_reachability),
        subject_path_reachability_hit=bool(
            predicted_subject_reachability & expected_subject_reachability
        ),
        missing_root_nodes=_sorted_node_labels(expected_roots - predicted_roots),
        extra_root_nodes=_sorted_node_labels(predicted_roots - expected_roots),
        missing_nodes=_sorted_node_labels(expected_nodes - predicted_nodes),
        extra_nodes=_sorted_node_labels(predicted_nodes - expected_nodes),
        missing_edges=_sorted_edge_labels(expected_edges - predicted_edges),
        extra_edges=_sorted_edge_labels(predicted_edges - expected_edges),
        missing_subjects=sorted(expected_subjects - predicted_subjects),
        extra_subjects=sorted(predicted_subjects - expected_subjects),
        missing_subject_edges=_sorted_subject_edge_labels(
            expected_subject_edges - predicted_subject_edges
        ),
        extra_subject_edges=_sorted_subject_edge_labels(
            predicted_subject_edges - expected_subject_edges
        ),
        missing_paths=_sorted_path_labels(expected_paths - predicted_paths),
        extra_paths=_sorted_path_labels(predicted_paths - expected_paths),
        missing_subject_paths=_sorted_subject_path_labels(
            expected_subject_paths - predicted_subject_paths
        ),
        extra_subject_paths=_sorted_subject_path_labels(
            predicted_subject_paths - expected_subject_paths
        ),
        missing_path_reachability=_sorted_edge_labels(
            expected_reachability - predicted_reachability
        ),
        extra_path_reachability=_sorted_edge_labels(
            predicted_reachability - expected_reachability
        ),
        missing_subject_path_reachability=_sorted_subject_edge_labels(
            expected_subject_reachability - predicted_subject_reachability
        ),
        extra_subject_path_reachability=_sorted_subject_edge_labels(
            predicted_subject_reachability - expected_subject_reachability
        ),
    )


def _coerce_model_output(value: ModelRCAOutput | Mapping[str, Any]) -> ModelRCAOutput:
    if isinstance(value, ModelRCAOutput):
        return value
    return ModelRCAOutput.model_validate(value)


def _coerce_graph(value: Scenario | Graph | Mapping[str, Any]) -> Graph:
    if isinstance(value, Scenario):
        return value.graph
    if isinstance(value, Graph):
        return value
    if "graph" in value:
        return Scenario.model_validate(value).graph
    return Graph.model_validate(value)


def _ground_truth_node_keys(graph: Graph) -> set[NodeKey]:
    return {
        key for node in graph.nodes if (key := _ground_truth_node_key(node)) is not None
    }


def _ground_truth_node_key(node: object) -> NodeKey | None:
    if isinstance(node, (EventNode, PreconditionNode)):
        return (str(node.subject), _string_value(node.predicate))
    return None


def _model_node_key(node: object) -> NodeKey:
    subject = getattr(node, "subject")
    predicate = getattr(node, "predicate")
    return (str(subject), _string_value(predicate))


def _ground_truth_edge_keys(graph: Graph) -> set[ExactEdgeKey]:
    node_by_id = {node.id: node for node in graph.nodes}
    successors: dict[str, list[str]] = defaultdict(list)
    for edge in graph.edges:
        successors[edge.src].append(edge.dst)

    keys: set[ExactEdgeKey] = set()
    for edge in graph.edges:
        src_key = _ground_truth_node_key(node_by_id[edge.src])
        if src_key is None:
            continue
        for dst_id in _grounded_targets_through_gates(
            edge.dst, node_by_id=node_by_id, successors=successors
        ):
            dst_key = _ground_truth_node_key(node_by_id[dst_id])
            if dst_key is not None:
                keys.add((src_key, dst_key))
    return keys


def _grounded_targets_through_gates(
    node_id: str,
    *,
    node_by_id: Mapping[str, object],
    successors: dict[str, list[str]],
) -> set[str]:
    node = node_by_id[node_id]
    if not isinstance(node, GateNode):
        return {node_id}

    targets: set[str] = set()
    queue: deque[str] = deque(successors[node_id])
    seen: set[str] = {node_id}
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        current_node = node_by_id[current]
        if isinstance(current_node, GateNode):
            queue.extend(successors[current])
        else:
            targets.add(current)
    return targets


def _subject_edges(edges: set[ExactEdgeKey]) -> set[SubjectEdgeKey]:
    return {(src[0], dst[0]) for src, dst in edges}


def _soft_subject_edge_metrics(
    predicted: set[SubjectEdgeKey], expected: set[SubjectEdgeKey], *, gamma: float
) -> MatchMetrics:
    return _soft_edge_metrics(
        _subject_edges_as_node_edges(predicted),
        _subject_edges_as_node_edges(expected),
        gamma=gamma,
    )


def _subject_edges_as_node_edges(edges: set[SubjectEdgeKey]) -> set[ExactEdgeKey]:
    return {((src, ""), (dst, "")) for src, dst in edges}


def _soft_edge_metrics(
    predicted: set[ExactEdgeKey], expected: set[ExactEdgeKey], *, gamma: float
) -> MatchMetrics:
    distances = _shortest_edge_distances(expected)
    matched = 0.0
    for src, dst in predicted:
        distance = distances.get(src, {}).get(dst)
        if distance is None:
            continue
        matched += gamma ** (distance - 1)
    predicted_count = len(predicted)
    expected_count = len(expected)
    precision = (
        matched / predicted_count
        if predicted_count
        else (1.0 if expected_count == 0 else 0.0)
    )
    recall = (
        matched / expected_count
        if expected_count
        else (1.0 if predicted_count == 0 else 0.0)
    )
    precision = min(precision, 1.0)
    recall = min(recall, 1.0)
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MatchMetrics(
        expected=expected_count,
        predicted=predicted_count,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _shortest_edge_distances(
    edges: set[ExactEdgeKey],
) -> dict[NodeKey, dict[NodeKey, int]]:
    successors: dict[NodeKey, list[NodeKey]] = defaultdict(list)
    starts: set[NodeKey] = set()
    for src, dst in edges:
        successors[src].append(dst)
        starts.add(src)

    distances: dict[NodeKey, dict[NodeKey, int]] = {}
    for start in starts:
        distances[start] = {}
        queue: deque[tuple[NodeKey, int]] = deque((dst, 1) for dst in successors[start])
        seen: set[NodeKey] = set()
        while queue:
            node, distance = queue.popleft()
            if node in seen:
                continue
            seen.add(node)
            distances[start][node] = distance
            for successor in successors[node]:
                queue.append((successor, distance + 1))
    return distances


def _ground_truth_paths(graph: Graph) -> set[ExactPathKey]:
    node_by_id = {node.id: node for node in graph.nodes}
    successors: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, int] = defaultdict(int)
    for edge in graph.edges:
        successors[edge.src].append(edge.dst)
        incoming[edge.dst] += 1

    roots = [
        node.id
        for node in graph.root_causes
        if _ground_truth_node_key(node) is not None
    ]
    terminals = {
        node.id
        for node in graph.nodes
        if _ground_truth_node_key(node) is not None
        and incoming[node.id] > 0
        and not _has_grounded_successor(node.id, node_by_id, successors)
        and not bool(getattr(node, "isolated_distractor", False))
    }
    key_by_id = {
        node.id: key
        for node in graph.nodes
        if (key := _ground_truth_node_key(node)) is not None
    }
    return _enumerate_paths(roots, terminals, successors, key_by_id)


def _model_paths(output: ModelRCAOutput) -> set[ExactPathKey]:
    node_by_id = {node.id: node for node in output.nodes}
    successors: dict[str, list[str]] = defaultdict(list)
    incoming: dict[str, int] = defaultdict(int)
    for edge in output.edges:
        successors[edge.src].append(edge.dst)
        incoming[edge.dst] += 1

    roots = [node_id for node_id in output.root_causes if node_id in node_by_id]
    terminals = {
        node.id
        for node in output.nodes
        if incoming[node.id] > 0 and not successors[node.id]
    }
    key_by_id = {node.id: _model_node_key(node) for node in output.nodes}
    return _enumerate_paths(roots, terminals, successors, key_by_id)


def _has_grounded_successor(
    node_id: str,
    node_by_id: Mapping[str, object],
    successors: dict[str, list[str]],
) -> bool:
    queue: deque[str] = deque(successors[node_id])
    seen: set[str] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        current_node = node_by_id[current]
        if _ground_truth_node_key(current_node) is not None:
            return True
        if isinstance(current_node, GateNode):
            queue.extend(successors[current])
    return False


def _enumerate_paths(
    roots: list[str],
    terminals: set[str],
    successors: dict[str, list[str]],
    key_by_id: dict[str, NodeKey],
) -> set[ExactPathKey]:
    paths: set[ExactPathKey] = set()
    for root in roots:
        if root not in key_by_id:
            continue
        _walk_paths(
            root,
            terminals=terminals,
            successors=successors,
            key_by_id=key_by_id,
            current=[],
            seen=set(),
            paths=paths,
        )
    return paths


def _walk_paths(
    node_id: str,
    *,
    terminals: set[str],
    successors: dict[str, list[str]],
    key_by_id: dict[str, NodeKey],
    current: list[NodeKey],
    seen: set[str],
    paths: set[ExactPathKey],
) -> None:
    if node_id in seen:
        return
    key = key_by_id.get(node_id)
    next_current = [*current, key] if key is not None else current
    if node_id in terminals and len(next_current) >= 2:
        paths.add(tuple(next_current))
    next_seen = {*seen, node_id}
    for successor in successors[node_id]:
        _walk_paths(
            successor,
            terminals=terminals,
            successors=successors,
            key_by_id=key_by_id,
            current=next_current,
            seen=next_seen,
            paths=paths,
        )


def _subject_paths(paths: set[ExactPathKey]) -> set[SubjectPathKey]:
    return {tuple(node[0] for node in path) for path in paths}


def _path_reachability(paths: set[ExactPathKey]) -> set[ExactEdgeKey]:
    return {(path[0], path[-1]) for path in paths}


def _subject_path_reachability(paths: set[ExactPathKey]) -> set[SubjectEdgeKey]:
    return {(path[0][0], path[-1][0]) for path in paths}


def _metrics(predicted: set[Any], expected: set[Any]) -> MatchMetrics:
    matched = len(predicted & expected)
    predicted_count = len(predicted)
    expected_count = len(expected)
    precision = (
        matched / predicted_count
        if predicted_count
        else (1.0 if expected_count == 0 else 0.0)
    )
    recall = (
        matched / expected_count
        if expected_count
        else (1.0 if predicted_count == 0 else 0.0)
    )
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return MatchMetrics(
        expected=expected_count,
        predicted=predicted_count,
        matched=matched,
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _string_value(value: object) -> str:
    if isinstance(value, Enum):
        return str(value.value)
    return str(value)


def _node_label(key: NodeKey) -> str:
    subject, predicate = key
    return f"{subject}|{predicate}"


def _edge_label(key: ExactEdgeKey) -> str:
    src, dst = key
    return f"{_node_label(src)} -> {_node_label(dst)}"


def _subject_edge_label(key: SubjectEdgeKey) -> str:
    src, dst = key
    return f"{src} -> {dst}"


def _path_label(key: ExactPathKey) -> str:
    return " -> ".join(_node_label(node) for node in key)


def _subject_path_label(key: SubjectPathKey) -> str:
    return " -> ".join(key)


def _sorted_node_labels(keys: set[NodeKey]) -> list[str]:
    return sorted(_node_label(key) for key in keys)


def _sorted_edge_labels(keys: set[ExactEdgeKey]) -> list[str]:
    return sorted(_edge_label(key) for key in keys)


def _sorted_subject_edge_labels(keys: set[SubjectEdgeKey]) -> list[str]:
    return sorted(_subject_edge_label(key) for key in keys)


def _sorted_path_labels(keys: set[ExactPathKey]) -> list[str]:
    return sorted(_path_label(key) for key in keys)


def _sorted_subject_path_labels(keys: set[SubjectPathKey]) -> list[str]:
    return sorted(_subject_path_label(key) for key in keys)
