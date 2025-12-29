"""Topological Growing Neural Gas for place graph construction."""

from __future__ import annotations

from dataclasses import dataclass
import threading
from typing import Dict, Iterable, Optional, Sequence

import networkx as nx
import numpy as np

try:
    from gng import GNGConfiguration, GrowingNeuralGas
except ImportError:
    GNGConfiguration = None
    GrowingNeuralGas = None


@dataclass
class PlaceGngUpdate:
    winner_id: str
    created: bool
    winner_changed: bool


class PlaceGng:
    """Incremental place discovery using i-GNG for place node adaptation."""

    def __init__(
        self,
        *,
        enabled: bool,
        distance_threshold: float,
        eps_w: float,
        eps_n: float,
        max_edge_age: int,
        gng_max_nodes: int,
        gng_lambda: int,
        gng_alpha: float,
        gng_beta: float,
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
        self._gng_max_nodes = int(gng_max_nodes)
        self._gng_lambda = int(gng_lambda)
        self._gng_alpha = float(gng_alpha)
        self._gng_beta = float(gng_beta)
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
        self._gng = None
        self._gng_node_map: Dict[int, str] = {}
        self._seed_nodes: list[tuple[str, np.ndarray]] = []
        self._seed_mapping_pending = False
        self._gng_ready = False
        self._sample_count = 0
        # Track pose insertions so edges can be formed once i-GNG materializes nodes.
        self._pending_edge_seeds: list[tuple[np.ndarray, Optional[str], Optional[str]]] = []

        if self.semantic_aggregation not in {"max", "sum"}:
            self._log_warning(
                f"Unknown semantic_aggregation '{self.semantic_aggregation}', falling back to 'max'."
            )
            self.semantic_aggregation = "max"

        self._sanitize_graph()
        self._init_gng()

    def seed_from_graph(self, graph: nx.Graph) -> None:
        if not self.enabled:
            return
        self.graph = graph
        self._sanitize_graph()
        self._pending_edge_seeds = []
        self._seed_gng_from_graph()

    def shutdown(self) -> None:
        if self._gng is None:
            return
        try:
            self._gng.terminate()
        except Exception:
            pass

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

        if self._gng is None:
            return None

        pre_winner_id = None
        pre_winner_dist = None
        if self.graph.number_of_nodes() > 0:
            pre_winner_id, _, pre_winner_dist = self._nearest_nodes(position)

        inserted = self._maybe_insert_sample(position, pre_winner_dist)
        current_pending_idx = None
        if inserted:
            current_pending_idx = self._queue_pending_edge(
                position, pre_winner_id, self._prev_winner
            )

        created_ids: set[str] = set()
        nodes = self._snapshot_gng_nodes()
        if nodes is not None:
            _, created_ids = self._sync_graph_from_gng(nodes)

        if self.graph.number_of_nodes() == 0:
            return None

        pending_assignments: dict[int, str] = {}
        if created_ids and self._pending_edge_seeds:
            pending_assignments = self._match_pending_edges(created_ids)

        winner_id, second_best_id, _ = self._nearest_nodes(position)
        created = winner_id in created_ids
        skip_pending_idx = None
        if current_pending_idx is not None:
            assigned_created = pending_assignments.get(current_pending_idx)
            if assigned_created is not None:
                winner_id = assigned_created
                if (
                    pre_winner_id is not None
                    and pre_winner_id in self.graph
                    and pre_winner_id != winner_id
                ):
                    second_best_id = pre_winner_id
                else:
                    second_best_id = None
                created = True
                skip_pending_idx = current_pending_idx

        if self._prev_winner is not None and self._prev_winner not in self.graph:
            self._prev_winner = None
        winner_changed = self._prev_winner is not None and self._prev_winner != winner_id

        self._age_edges(winner_id)
        if self.use_second_best_edge and second_best_id is not None and second_best_id != winner_id:
            self._touch_edge(winner_id, second_best_id)
        if self.use_transition_edges and winner_changed:
            self._touch_edge(self._prev_winner, winner_id)

        self._apply_pending_edges(pending_assignments, skip_pending_idx=skip_pending_idx)
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

    def _init_gng(self) -> None:
        if not self.enabled:
            return
        if GNGConfiguration is None or GrowingNeuralGas is None:
            self._log_error("GNG bindings are unavailable; disabling place_gng.")
            self.enabled = False
            return
        config = self._build_config()
        self._gng = GrowingNeuralGas(config)
        self._seed_gng_from_graph()

    def _build_config(self):
        config = GNGConfiguration()
        config.dim = 2
        if self._gng_max_nodes > 0:
            config.max_nodes = self._gng_max_nodes
        if self._gng_lambda > 0:
            config.lambda_ = self._gng_lambda
        config.grow_on_new_samples = True
        if hasattr(config, "new_node_position_mode"):
            # 1 = place new nodes at the most recent sample instead of error midpoint.
            config.new_node_position_mode = 1
        if self.max_edge_age > 0:
            config.max_age = self.max_edge_age
        if self.eps_w > 0.0:
            config.eps_w = self.eps_w
        if self.eps_n > 0.0:
            config.eps_n = self.eps_n
        if self._gng_alpha > 0.0:
            config.alpha = self._gng_alpha
        if self._gng_beta > 0.0:
            config.beta = self._gng_beta
        config.dataset_type = 2
        return config

    def _seed_gng_from_graph(self) -> None:
        if self._gng is None:
            return
        self._gng_node_map = {}
        self._seed_nodes = []
        self._seed_mapping_pending = False
        if self.graph.number_of_nodes() == 0:
            return
        positions = []
        for node_id, data in self.graph.nodes(data=True):
            pose = data.get("pose")
            pose_xy = self._as_position(pose)
            if pose_xy is None:
                continue
            positions.append(pose_xy)
            self._seed_nodes.append((str(node_id), pose_xy))
        if not positions:
            return
        insert_positions = positions
        if len(positions) == 1:
            insert_positions = [positions[0], positions[0]]
        self._seed_mapping_pending = True
        self._gng.insert(np.stack(insert_positions, axis=0))
        self._sample_count = int(self._gng.server.dataset_size())

    def _maybe_insert_sample(self, position: np.ndarray, winner_dist: Optional[float]) -> bool:
        if self._gng is None:
            return False
        dataset_size = int(self._gng.server.dataset_size())
        if dataset_size < 2:
            bootstrap = dataset_size == 0
            self._insert_sample(position, bootstrap=bootstrap)
            return True
        # Gate insertions so distance_threshold still controls place growth.
        if self.distance_threshold <= 0.0:
            self._insert_sample(position)
            return True
        if winner_dist is None or winner_dist > self.distance_threshold:
            self._insert_sample(position)
            return True
        return False

    def _insert_sample(self, position: np.ndarray, *, bootstrap: bool = False) -> None:
        if self._gng is None:
            return
        self._gng.insert(position[None, :])
        self._sample_count += 1
        if bootstrap:
            self._gng.insert(position[None, :])
            self._sample_count += 1

    def _queue_pending_edge(
        self,
        position: np.ndarray,
        nearest_id: Optional[str],
        prev_winner: Optional[str],
    ) -> int:
        self._pending_edge_seeds.append((position.copy(), nearest_id, prev_winner))
        return len(self._pending_edge_seeds) - 1

    def _match_pending_edges(self, created_ids: set[str]) -> dict[int, str]:
        if not created_ids or not self._pending_edge_seeds:
            return {}
        pairs: list[tuple[float, str, int]] = []
        for created_id in created_ids:
            created_pos = np.asarray(self.graph.nodes[created_id]["pose"], dtype=np.float64)
            for idx, (pending_pos, _, _) in enumerate(self._pending_edge_seeds):
                dist = float(np.linalg.norm(created_pos - pending_pos))
                pairs.append((dist, created_id, idx))
        pairs.sort(key=lambda item: item[0])
        assignments: dict[int, str] = {}
        assigned_created: set[str] = set()
        assigned_pending: set[int] = set()
        for _, created_id, idx in pairs:
            if created_id in assigned_created or idx in assigned_pending:
                continue
            assignments[idx] = created_id
            assigned_created.add(created_id)
            assigned_pending.add(idx)
        return assignments

    def _apply_pending_edges(
        self,
        pending_assignments: dict[int, str],
        *,
        skip_pending_idx: Optional[int] = None,
    ) -> None:
        if not pending_assignments:
            return
        for idx, created_id in pending_assignments.items():
            if idx == skip_pending_idx:
                continue
            _, nearest_id, prev_winner_id = self._pending_edge_seeds[idx]
            if self.use_second_best_edge and nearest_id is not None:
                self._touch_edge(created_id, nearest_id)
            if self.use_transition_edges and prev_winner_id is not None:
                self._touch_edge(prev_winner_id, created_id)
        assigned_indices = set(pending_assignments.keys())
        self._pending_edge_seeds = [
            entry
            for idx, entry in enumerate(self._pending_edge_seeds)
            if idx not in assigned_indices
        ]

    def _snapshot_gng_nodes(self):
        if self._gng is None:
            return None

        pause_event = threading.Event()
        pause_exception = [None]
        resume_event = threading.Event()

        def pause_with_catch():
            try:
                self._gng.pause()
            except Exception as exc:
                pause_exception[0] = exc
            if resume_event.is_set():
                try:
                    self._gng.run()
                except Exception:
                    pass
            pause_event.set()

        pause_thread = threading.Thread(target=pause_with_catch, daemon=True)
        pause_thread.start()

        if not pause_event.wait(timeout=2.0):
            resume_event.set()
            self._log_warning(
                f"Place GNG pause timeout after 2.0s (samples={self._sample_count})."
            )
            return None

        if pause_exception[0] is not None:
            self._log_error(f"Place GNG pause failed: {pause_exception[0]}")
            return None

        try:
            return self._gng.nodes()
        finally:
            self._gng.run()

    def _sync_graph_from_gng(self, nodes) -> tuple[set[str], set[str]]:
        if not nodes:
            return set(), set()
        if self._seed_mapping_pending:
            self._assign_seed_mapping(nodes)
        active_indices = {node.index for node in nodes}
        active_ids: set[str] = set()
        created_ids: set[str] = set()
        for node in nodes:
            node_id = self._gng_node_map.get(node.index)
            if node_id is None:
                node_id = self._create_node(np.asarray(node.position, dtype=np.float64))
                created_ids.add(node_id)
                self._gng_node_map[node.index] = node_id
            data = self.graph.nodes[node_id]
            data["pose"] = np.asarray(node.position, dtype=np.float64).tolist()
            self._ensure_node_fields(node_id)
            active_ids.add(node_id)
        if not self._seed_mapping_pending:
            self._gng_ready = True
        if self._gng_ready:
            self._prune_missing_nodes(active_ids)
            self._prune_missing_mappings(active_indices)
        return active_ids, created_ids

    def _assign_seed_mapping(self, nodes) -> None:
        if not self._seed_mapping_pending or not self._seed_nodes:
            self._seed_mapping_pending = False
            return
        seed_positions = {node_id: pose for node_id, pose in self._seed_nodes}
        pairs = []
        for node in nodes:
            node_pos = np.asarray(node.position, dtype=np.float64)
            for node_id, pose in seed_positions.items():
                dist = float(np.linalg.norm(node_pos - pose))
                pairs.append((dist, node.index, node_id))
        pairs.sort(key=lambda item: item[0])
        assigned_nodes: set[int] = set()
        assigned_seeds: set[str] = set()
        for _, gng_idx, node_id in pairs:
            if gng_idx in assigned_nodes or node_id in assigned_seeds:
                continue
            self._gng_node_map[gng_idx] = node_id
            assigned_nodes.add(gng_idx)
            assigned_seeds.add(node_id)
        self._seed_mapping_pending = False

    def _prune_missing_nodes(self, active_ids: set[str]) -> None:
        prefix = f"{self._node_prefix}_"
        for node_id in list(self.graph.nodes):
            if node_id in active_ids:
                continue
            if isinstance(node_id, str) and node_id.startswith(prefix):
                self.graph.remove_node(node_id)

    def _prune_missing_mappings(self, active_indices: set[int]) -> None:
        for idx in list(self._gng_node_map):
            if idx not in active_indices:
                del self._gng_node_map[idx]

    def _ensure_node_fields(self, node_id: str) -> None:
        data = self.graph.nodes[node_id]
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

    def _age_edges(self, winner_id: str) -> None:
        if self.max_edge_age <= 0:
            return
        for neighbor in list(self.graph.neighbors(winner_id)):
            edge = self.graph[winner_id][neighbor]
            edge["age"] = int(edge.get("age", 0)) + 1

    def _touch_edge(self, node_a: str, node_b: str) -> None:
        if node_a is None or node_b is None or node_a == node_b:
            return
        if node_a not in self.graph or node_b not in self.graph:
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

    def _log_error(self, message: str) -> None:
        if self._logger:
            self._logger.error(message)
