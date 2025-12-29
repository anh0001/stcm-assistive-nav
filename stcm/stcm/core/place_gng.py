"""Topological Growing Neural Gas for place graph construction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence

import networkx as nx
import numpy as np


@dataclass
class PlaceGngUpdate:
    winner_id: str
    created: bool
    winner_changed: bool


class PlaceGng:
    """Incremental place discovery using a distance-threshold GNG variant."""

    def __init__(
        self,
        *,
        enabled: bool,
        distance_threshold: float,
        eps_w: float,
        eps_n: float,
        max_edge_age: int,
        semantic_alpha: float,
        semantic_aggregation: str,
        use_second_best_edge: bool,
        use_transition_edges: bool,
        update_semantics_when_empty: bool,
        labels: Sequence[str] | None = None,
        graph: Optional[nx.Graph] = None,
        node_prefix: str = "place",
        logger=None,
    ) -> None:
        self.enabled = bool(enabled)
        self.distance_threshold = float(distance_threshold)
        self.eps_w = float(eps_w)
        self.eps_n = float(eps_n)
        self.max_edge_age = int(max_edge_age)
        self.semantic_alpha = float(semantic_alpha)
        self.semantic_aggregation = str(semantic_aggregation).lower()
        self.use_second_best_edge = bool(use_second_best_edge)
        self.use_transition_edges = bool(use_transition_edges)
        self.update_semantics_when_empty = bool(update_semantics_when_empty)
        self._labels = list(labels) if labels is not None else []
        self._label_set = set(self._labels)
        self.graph = graph if graph is not None else nx.Graph()
        self._node_prefix = node_prefix
        self._node_counter = 0
        self._prev_winner: str | None = None
        self._logger = logger

        if self.semantic_aggregation not in {"max", "sum"}:
            self._log_warning(
                f"Unknown semantic_aggregation '{self.semantic_aggregation}', falling back to 'max'."
            )
            self.semantic_aggregation = "max"

        self._sanitize_graph()

    def seed_from_graph(self, graph: nx.Graph) -> None:
        if not self.enabled:
            return
        self.graph = graph
        self._sanitize_graph()

    def update(
        self,
        position_xy: np.ndarray,
        *,
        labels: Optional[Iterable[str]] = None,
        scores: Optional[Iterable[float]] = None,
    ) -> Optional[PlaceGngUpdate]:
        if not self.enabled:
            return None

        position = self._as_position(position_xy)
        if position is None:
            return None

        created = False
        if self.graph.number_of_nodes() == 0:
            winner_id = self._create_node(position)
            created = True
            second_best_id = None
        else:
            winner_id, second_best_id, winner_dist = self._nearest_nodes(position)
            if self.distance_threshold > 0.0 and winner_dist > self.distance_threshold:
                nearest_before_insert = winner_id
                winner_id = self._create_node(position)
                created = True
                second_best_id = nearest_before_insert
            else:
                self._adapt_winner_and_neighbors(winner_id, position)

        winner_changed = self._prev_winner is not None and self._prev_winner != winner_id

        self._age_edges(winner_id)
        if self.use_second_best_edge and second_best_id is not None and second_best_id != winner_id:
            self._touch_edge(winner_id, second_best_id)
        if self.use_transition_edges and winner_changed:
            self._touch_edge(self._prev_winner, winner_id)

        self._prune_edges()
        self._increment_visits(winner_id)
        self._update_semantics(winner_id, labels, scores)

        self._prev_winner = winner_id
        return PlaceGngUpdate(winner_id=winner_id, created=created, winner_changed=winner_changed)

    def _sanitize_graph(self) -> None:
        if not self.graph:
            self._node_counter = 0
            return

        self._node_counter = self._infer_next_counter()

        for node_id, data in self.graph.nodes(data=True):
            pose = data.get("pose")
            pose_xy = self._as_position(pose)
            if pose_xy is None:
                self._log_warning(f"Place GNG node '{node_id}' missing pose; skipping.")
                continue
            data["pose"] = pose_xy.tolist()
            data.setdefault("id", node_id)
            data.setdefault("visits", 0)
            scores = data.get("scores")
            if not isinstance(scores, dict):
                scores = {}
            if self._labels:
                for label in self._labels:
                    scores.setdefault(label, 0.0)
            data["scores"] = scores
            label = data.get("label")
            if not label and scores:
                data["label"] = self._resolve_label(scores)

        for _, _, edge_data in self.graph.edges(data=True):
            edge_data.setdefault("age", 0)
            edge_data.setdefault("traversals", 0)

    def _infer_next_counter(self) -> int:
        max_idx = -1
        prefix = f"{self._node_prefix}_"
        for node_id in self.graph.nodes:
            if isinstance(node_id, str) and node_id.startswith(prefix):
                suffix = node_id[len(prefix) :]
                if suffix.isdigit():
                    max_idx = max(max_idx, int(suffix))
        if max_idx >= 0:
            return max_idx + 1
        return len(self.graph.nodes)

    def _create_node(self, position: np.ndarray) -> str:
        node_id = self._next_node_id()
        scores = {label: 0.0 for label in self._labels}
        self.graph.add_node(
            node_id,
            id=node_id,
            pose=position.tolist(),
            scores=scores,
            label="",
            visits=0,
        )
        return node_id

    def _next_node_id(self) -> str:
        while True:
            candidate = f"{self._node_prefix}_{self._node_counter}"
            self._node_counter += 1
            if candidate not in self.graph:
                return candidate

    def _nearest_nodes(self, position: np.ndarray) -> tuple[str, Optional[str], float]:
        nodes = list(self.graph.nodes(data=True))
        positions = np.array([node[1]["pose"] for node in nodes], dtype=np.float64)
        dists = np.linalg.norm(positions - position, axis=1)
        winner_idx = int(np.argmin(dists))
        winner_id = nodes[winner_idx][0]
        second_best_id = None
        if len(nodes) > 1:
            second_idx = int(np.argsort(dists)[1])
            second_best_id = nodes[second_idx][0]
        return winner_id, second_best_id, float(dists[winner_idx])

    def _adapt_winner_and_neighbors(self, winner_id: str, position: np.ndarray) -> None:
        winner_data = self.graph.nodes[winner_id]
        winner_pos = np.array(winner_data["pose"], dtype=np.float64)
        winner_pos = winner_pos + self.eps_w * (position - winner_pos)
        winner_data["pose"] = winner_pos.tolist()
        for neighbor in list(self.graph.neighbors(winner_id)):
            neighbor_data = self.graph.nodes[neighbor]
            neighbor_pos = np.array(neighbor_data["pose"], dtype=np.float64)
            neighbor_pos = neighbor_pos + self.eps_n * (position - neighbor_pos)
            neighbor_data["pose"] = neighbor_pos.tolist()

    def _age_edges(self, winner_id: str) -> None:
        if self.max_edge_age <= 0:
            return
        for neighbor in list(self.graph.neighbors(winner_id)):
            edge = self.graph[winner_id][neighbor]
            edge["age"] = int(edge.get("age", 0)) + 1

    def _touch_edge(self, node_a: str, node_b: str) -> None:
        if node_a is None or node_b is None or node_a == node_b:
            return
        if not self.graph.has_edge(node_a, node_b):
            self.graph.add_edge(node_a, node_b, age=0, traversals=0)
        edge = self.graph[node_a][node_b]
        edge["age"] = 0
        edge["traversals"] = int(edge.get("traversals", 0)) + 1

    def _prune_edges(self) -> None:
        if self.max_edge_age <= 0:
            return
        edges_to_remove = []
        for node_a, node_b, edge_data in self.graph.edges(data=True):
            if int(edge_data.get("age", 0)) > self.max_edge_age:
                edges_to_remove.append((node_a, node_b))
        for node_a, node_b in edges_to_remove:
            self.graph.remove_edge(node_a, node_b)

    def _increment_visits(self, winner_id: str) -> None:
        node_data = self.graph.nodes[winner_id]
        node_data["visits"] = int(node_data.get("visits", 0)) + 1

    def _update_semantics(
        self,
        winner_id: str,
        labels: Optional[Iterable[str]],
        scores: Optional[Iterable[float]],
    ) -> None:
        if not self._labels:
            return

        observation = self._build_observation(labels, scores)
        if observation is None and not self.update_semantics_when_empty:
            return

        if observation is None:
            observation = {}

        node_data = self.graph.nodes[winner_id]
        scores_map = node_data.get("scores")
        if not isinstance(scores_map, dict):
            scores_map = {}

        alpha = max(0.0, min(1.0, self.semantic_alpha))
        for label in self._labels:
            prev = float(scores_map.get(label, 0.0))
            obs_val = float(observation.get(label, 0.0))
            scores_map[label] = (1.0 - alpha) * prev + alpha * obs_val

        node_data["scores"] = scores_map
        node_data["label"] = self._resolve_label(scores_map)

    def _build_observation(
        self,
        labels: Optional[Iterable[str]],
        scores: Optional[Iterable[float]],
    ) -> Optional[Dict[str, float]]:
        if labels is None:
            return None
        labels_list = list(labels)
        if not labels_list:
            return None
        scores_list = list(scores) if scores is not None else [1.0] * len(labels_list)

        observation: Dict[str, float] = {}
        for label, score in zip(labels_list, scores_list):
            if label not in self._label_set:
                continue
            score_val = self._clamp_score(score)
            if self.semantic_aggregation == "sum":
                observation[label] = observation.get(label, 0.0) + score_val
            else:
                observation[label] = max(observation.get(label, 0.0), score_val)
        if not observation:
            return None
        return observation

    @staticmethod
    def _resolve_label(scores: Dict[str, float]) -> str:
        if not scores:
            return ""
        best_label = ""
        best_score = None
        for label, score in scores.items():
            if best_score is None or score > best_score:
                best_label = label
                best_score = score
        if best_score is None or best_score <= 0.0:
            return ""
        return best_label

    @staticmethod
    def _clamp_score(score) -> float:
        try:
            value = float(score)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(value):
            return 0.0
        return max(0.0, min(1.0, value))

    @staticmethod
    def _as_position(position_xy) -> Optional[np.ndarray]:
        if position_xy is None:
            return None
        arr = np.asarray(position_xy, dtype=np.float64).reshape(-1)
        if arr.size < 2:
            return None
        return arr[:2]

    def _log_warning(self, message: str) -> None:
        if self._logger:
            self._logger.warning(message)
