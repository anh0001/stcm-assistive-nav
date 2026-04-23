"""Instance management using GNG clustering with cross-label instance memory."""

from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Any, Dict, Optional

import numpy as np

from .label_calibration import normalize_score

try:
    from gng import GNGConfiguration, GrowingNeuralGas
except ImportError:
    GNGConfiguration = None
    GrowingNeuralGas = None


@dataclass
class InstanceAssignment:
    instance_id: str
    label: str
    centroid: np.ndarray
    stability: float
    committed: bool
    label_votes: dict[str, float]
    last_label_scores: dict[str, float]
    appearance_embedding: np.ndarray | None = None


@dataclass
class ClusterState:
    instance_id: str
    centroid: np.ndarray
    observations: int
    first_seen: float
    last_seen: float
    committed: bool
    label_votes: dict[str, float]
    last_label_scores: dict[str, float]
    appearance_embedding: np.ndarray | None = None


@dataclass
class _LabelState:
    model: "GrowingNeuralGas"
    sample_count: int = 0  # Track insertions to avoid querying cold models


class GngInstanceManager:
    """Tracks stable instances per semantic label using i-GNG."""

    def __init__(
        self,
        *,
        enabled: bool,
        per_label: bool,
        max_nodes: int,
        lambda_: int,
        max_age: int,
        eps_w: float,
        eps_n: float,
        alpha: float,
        beta: float,
        min_observations_to_commit: int,
        cluster_merge_distance: float,
        outlier_gate_meters: float,
        instance_label_voting_enabled: bool,
        cross_label_merge_distance_m: float,
        logger=None,
    ) -> None:
        self.enabled = bool(enabled)
        self._per_label = bool(per_label)
        self._max_nodes = int(max_nodes)
        self._lambda = int(lambda_)
        self._max_age = int(max_age)
        self._eps_w = float(eps_w)
        self._eps_n = float(eps_n)
        self._alpha = float(alpha)
        self._beta = float(beta)
        self._min_obs = int(min_observations_to_commit)
        self._merge_distance = float(cluster_merge_distance)
        self._outlier_gate = float(outlier_gate_meters)
        self._label_voting_enabled = bool(instance_label_voting_enabled)
        self._cross_label_merge_distance = float(cross_label_merge_distance_m)
        self._logger = logger
        self._states: Dict[str, _LabelState] = {}
        self._clusters: Dict[str, ClusterState] = {}
        self._instance_counters: Dict[str, int] = {}
        self._used_instance_ids: set[str] = set()
        self._warned_per_label = False

        if self.enabled and (GNGConfiguration is None or GrowingNeuralGas is None):
            self._log_error("GNG bindings are unavailable; disabling gng_enabled.")
            self.enabled = False

    def shutdown(self) -> None:
        if not self.enabled:
            return
        for state in self._states.values():
            try:
                state.model.terminate()
            except Exception:
                continue

    def seed_from_graph(self, graph) -> None:
        if not self.enabled:
            return

        label_groups: Dict[str, list[tuple[str, np.ndarray]]] = {}
        for node_id, data in graph.nodes(data=True):
            label = data.get("category")
            pose = data.get("pose")
            if not label or pose is None:
                continue
            pose_arr = np.asarray(pose, dtype=np.float64)
            if pose_arr.shape[0] < 3:
                continue
            label_groups.setdefault(label, []).append((str(node_id), pose_arr[:3]))

        for label, entries in label_groups.items():
            state = self._get_state(label)
            if state is None:
                continue
            positions = np.stack([pose for _, pose in entries], axis=0)
            state.model.insert(positions, probabilities=np.ones(len(entries)))
            for node_id, pose in entries:
                instance_id = str(node_id)
                self._used_instance_ids.add(instance_id)
                votes = {label: float(max(1, self._min_obs))}
                self._clusters[instance_id] = ClusterState(
                    instance_id=instance_id,
                    centroid=pose.copy(),
                    observations=max(1, self._min_obs),
                    first_seen=time.time(),
                    last_seen=time.time(),
                    committed=True,
                    label_votes=votes,
                    last_label_scores=dict(votes),
                )

    def update(
        self,
        label: str,
        centroid: np.ndarray,
        confidence: Optional[float] = None,
        stamp=None,
        label_scores: Optional[dict[str, float]] = None,
        appearance_embedding: Optional[np.ndarray] = None,
    ) -> Optional[InstanceAssignment]:
        if not self.enabled:
            return None
        state = self._get_state(label)
        if state is None:
            return None
        centroid_arr = np.asarray(centroid, dtype=np.float64).reshape(3)
        stamp_sec = self._stamp_to_seconds(stamp)

        if self._outlier_gate > 0.0 and self._clusters:
            nearest = min(
                np.linalg.norm(centroid_arr - cluster.centroid)
                for cluster in self._clusters.values()
            )
            if nearest > self._outlier_gate:
                return None

        prob = self._clamp_confidence(confidence)
        label_score_map = self._normalize_label_scores(label, label_scores, prob)
        appearance_arr = self._normalize_embedding(appearance_embedding)
        if prob is None:
            state.model.insert(centroid_arr[None, :])
        else:
            state.model.insert(centroid_arr[None, :], probabilities=[prob])
        
        state.sample_count += 1
        
        # Skip querying GNG until it has warmed up with enough samples
        # The GNG algorithm thread only checks pause requests every lambda iterations
        # If we try to pause before it processes enough data, it will deadlock
        min_samples_for_query = max(self._lambda * 2, 20)  # 2x lambda or 20, whichever is larger
        if state.sample_count < min_samples_for_query:
            # Not enough data yet - fall back to simple distance-based clustering
            cluster_state = self._assign_component(label, centroid_arr, stamp_sec)
            return self._finalize_assignment(
                cluster_state,
                observed_label=label,
                centroid=centroid_arr,
                confidence=prob,
                stamp_sec=stamp_sec,
                label_scores=label_score_map,
                appearance_embedding=appearance_arr,
            )

        # GNG queries must run while the background thread is paused.
        # Use timeout to prevent deadlocks when GNG thread doesn't check pause requests
        
        # Wrapper to call pause with timeout using threading
        pause_event = threading.Event()
        pause_exception = [None]
        
        def pause_with_catch():
            try:
                state.model.pause()
                pause_event.set()
            except Exception as e:
                pause_exception[0] = e
                pause_event.set()
        
        pause_thread = threading.Thread(target=pause_with_catch, daemon=True)
        pause_thread.start()
        
        # Wait up to 2 seconds for pause to complete
        if not pause_event.wait(timeout=2.0):
            if self._logger:
                self._log_error(
                    f"GNG pause timeout after 2.0s for label '{label}' (sample #{state.sample_count}). "
                    f"Falling back to distance-based clustering."
                )
            # Fall back to simple clustering without querying GNG nodes
            cluster_state = self._assign_component(label, centroid_arr, stamp_sec)
            return self._finalize_assignment(
                cluster_state,
                observed_label=label,
                centroid=centroid_arr,
                confidence=prob,
                stamp_sec=stamp_sec,
                label_scores=label_score_map,
                appearance_embedding=appearance_arr,
                force_commit=True,
            )
        
        if pause_exception[0] is not None:
            if self._logger:
                self._log_error(f"GNG pause failed for '{label}': {pause_exception[0]}")
            return None
        
        try:
            nodes = state.model.nodes()
            if not nodes:
                cluster_state = self._assign_component(label, centroid_arr, stamp_sec)
                return self._finalize_assignment(
                    cluster_state,
                    observed_label=label,
                    centroid=centroid_arr,
                    confidence=prob,
                    stamp_sec=stamp_sec,
                    label_scores=label_score_map,
                    appearance_embedding=appearance_arr,
                )

            node_to_component, component_centroids = self._compute_components(nodes)
            try:
                winner_idx = state.model.predict(centroid_arr)
                component_key = node_to_component.get(winner_idx)
            except Exception:
                component_key = None
            if component_key is None:
                component_key = self._nearest_component_key(component_centroids, centroid_arr)
            if component_key is None:
                return None
            component_centroid = component_centroids[component_key]

            cluster_state = self._assign_component(label, component_centroid, stamp_sec)
            return self._finalize_assignment(
                cluster_state,
                observed_label=label,
                centroid=component_centroid,
                confidence=prob,
                stamp_sec=stamp_sec,
                label_scores=label_score_map,
                appearance_embedding=appearance_arr,
            )
        finally:
            state.model.run()

    def _get_state(self, label: str) -> Optional[_LabelState]:
        if not self.enabled:
            return None
        state_key = label
        if not self._per_label:
            if not self._warned_per_label:
                self._log_warning("gng_per_label is false; using a shared GNG model.")
                self._warned_per_label = True
            state_key = "__global__"

        state = self._states.get(state_key)
        if state is None:
            config = self._build_config()
            model = GrowingNeuralGas(config)
            state = _LabelState(model=model)
            self._states[state_key] = state
        return state

    def _build_config(self):
        config = GNGConfiguration()
        config.dim = 3
        if self._max_nodes > 0:
            config.max_nodes = self._max_nodes
        if self._lambda > 0:
            config.lambda_ = self._lambda
        if self._max_age > 0:
            config.max_age = self._max_age
        if self._eps_w > 0.0:
            config.eps_w = self._eps_w
        if self._eps_n > 0.0:
            config.eps_n = self._eps_n
        if self._alpha > 0.0:
            config.alpha = self._alpha
        if self._beta > 0.0:
            config.beta = self._beta
        config.dataset_type = 3
        return config

    def _new_instance_id(self, label: str) -> str:
        counter = self._instance_counters.get(label, 0)
        while True:
            candidate = f"{label}_inst_{counter}"
            if candidate not in self._used_instance_ids:
                self._instance_counters[label] = counter + 1
                self._used_instance_ids.add(candidate)
                return candidate
            counter += 1

    def _assign_component(self, label: str, component_centroid: np.ndarray, stamp_sec: float) -> ClusterState:
        best_id = None
        best_dist = None
        max_distance = self._cross_label_merge_distance if self._label_voting_enabled else self._merge_distance
        for cluster_id, cluster in self._clusters.items():
            if (not self._label_voting_enabled) and self._dominant_label(cluster) != label:
                continue
            dist = np.linalg.norm(component_centroid - cluster.centroid)
            if best_dist is None or dist < best_dist:
                best_id = cluster_id
                best_dist = dist

        if best_id is not None and (max_distance <= 0.0 or best_dist <= max_distance):
            return self._clusters[best_id]

        instance_id = self._new_instance_id(label)
        cluster_state = ClusterState(
            instance_id=instance_id,
            centroid=component_centroid.copy(),
            observations=0,
            first_seen=stamp_sec,
            last_seen=stamp_sec,
            committed=self._min_obs <= 0,
            label_votes={label: 0.0},
            last_label_scores={label: 0.0},
        )
        self._clusters[instance_id] = cluster_state
        return cluster_state

    def _finalize_assignment(
        self,
        cluster_state: ClusterState,
        *,
        observed_label: str,
        centroid: np.ndarray,
        confidence: Optional[float],
        stamp_sec: float,
        label_scores: dict[str, float],
        appearance_embedding: np.ndarray | None,
        force_commit: bool = False,
    ) -> InstanceAssignment:
        cluster_state.observations += 1
        cluster_state.last_seen = stamp_sec
        if cluster_state.observations == 1:
            cluster_state.first_seen = stamp_sec
        cluster_state.centroid = self._blend_centroid(cluster_state.centroid, centroid, confidence)
        self._update_label_memory(cluster_state, observed_label, label_scores)
        cluster_state.appearance_embedding = self._blend_embedding(
            cluster_state.appearance_embedding,
            appearance_embedding,
            confidence,
        )
        if force_commit:
            cluster_state.committed = True
        elif not cluster_state.committed and self._min_obs > 0 and cluster_state.observations >= self._min_obs:
            cluster_state.committed = True

        stability = self._compute_stability(cluster_state)
        return InstanceAssignment(
            instance_id=cluster_state.instance_id,
            label=self._dominant_label(cluster_state),
            centroid=cluster_state.centroid.copy(),
            stability=stability,
            committed=cluster_state.committed or self._min_obs <= 0,
            label_votes=dict(cluster_state.label_votes),
            last_label_scores=dict(cluster_state.last_label_scores),
            appearance_embedding=None if cluster_state.appearance_embedding is None else cluster_state.appearance_embedding.copy(),
        )

    def _update_label_memory(
        self,
        cluster_state: ClusterState,
        observed_label: str,
        label_scores: dict[str, float],
    ) -> None:
        if not label_scores:
            label_scores = {observed_label: 1.0}
        cluster_state.last_label_scores = dict(label_scores)
        for label, score in label_scores.items():
            cluster_state.label_votes[label] = cluster_state.label_votes.get(label, 0.0) + normalize_score(score)
        if observed_label not in cluster_state.label_votes:
            cluster_state.label_votes[observed_label] = 0.0

    @staticmethod
    def _dominant_label(cluster_state: ClusterState) -> str:
        if cluster_state.label_votes:
            return max(cluster_state.label_votes.items(), key=lambda item: item[1])[0]
        if cluster_state.last_label_scores:
            return max(cluster_state.last_label_scores.items(), key=lambda item: item[1])[0]
        return "unknown"

    def _compute_components(self, nodes):
        if not nodes:
            return {}, {}

        parent = {node.index: node.index for node in nodes}

        def find(idx):
            while parent[idx] != idx:
                parent[idx] = parent[parent[idx]]
                idx = parent[idx]
            return idx

        def union(a, b):
            root_a = find(a)
            root_b = find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        for node in nodes:
            for neighbor in node.neighbours:
                if neighbor in parent:
                    union(node.index, neighbor)

        components: Dict[int, list[np.ndarray]] = {}
        node_to_component = {}
        for node in nodes:
            root = find(node.index)
            node_to_component[node.index] = root
            components.setdefault(root, []).append(node.position.astype(np.float64))

        component_centroids = {
            key: np.mean(np.stack(positions, axis=0), axis=0)
            for key, positions in components.items()
        }
        return node_to_component, component_centroids

    @staticmethod
    def _nearest_component_key(component_centroids, centroid: np.ndarray):
        best_key = None
        best_dist = None
        for key, center in component_centroids.items():
            dist = np.linalg.norm(centroid - center)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_key = key
        return best_key

    @staticmethod
    def _stamp_to_seconds(stamp) -> float:
        if stamp is None:
            return time.time()
        if hasattr(stamp, "sec") and hasattr(stamp, "nanosec"):
            return float(stamp.sec) + float(stamp.nanosec) * 1e-9
        if hasattr(stamp, "seconds_nanoseconds"):
            sec, nsec = stamp.seconds_nanoseconds()
            return float(sec) + float(nsec) * 1e-9
        if isinstance(stamp, (float, int)):
            return float(stamp)
        return time.time()

    @staticmethod
    def _blend_centroid(prev: np.ndarray, current: np.ndarray, confidence: Optional[float]) -> np.ndarray:
        if prev is None:
            return current
        weight = 0.4
        if confidence is not None and np.isfinite(confidence):
            weight = max(0.1, min(0.9, float(confidence)))
        return prev * (1.0 - weight) + current * weight

    def _compute_stability(self, cluster: ClusterState) -> float:
        if self._min_obs <= 0:
            return 1.0
        return min(1.0, cluster.observations / float(self._min_obs))

    def _normalize_label_scores(
        self,
        label: str,
        label_scores: Optional[dict[str, float]],
        confidence: Optional[float],
    ) -> dict[str, float]:
        if label_scores:
            return {str(key): normalize_score(value) for key, value in label_scores.items()}
        return {label: normalize_score(confidence if confidence is not None else 1.0)}

    @staticmethod
    def _normalize_embedding(embedding: Optional[np.ndarray]) -> np.ndarray | None:
        if embedding is None:
            return None
        arr = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return None
        norm = np.linalg.norm(arr)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return arr / norm

    @staticmethod
    def _blend_embedding(
        prev: np.ndarray | None,
        current: np.ndarray | None,
        confidence: Optional[float],
    ) -> np.ndarray | None:
        if current is None:
            return prev
        if prev is None:
            return current
        weight = 0.4
        if confidence is not None and np.isfinite(confidence):
            weight = max(0.1, min(0.9, float(confidence)))
        blended = prev * (1.0 - weight) + current * weight
        norm = np.linalg.norm(blended)
        if not np.isfinite(norm) or norm <= 0.0:
            return None
        return blended / norm

    @staticmethod
    def _clamp_confidence(confidence: Optional[float]) -> Optional[float]:
        if confidence is None:
            return None
        try:
            value = float(confidence)
        except (TypeError, ValueError):
            return None
        if not np.isfinite(value):
            return None
        return max(0.0, min(1.0, value))

    def _log_warning(self, message: str) -> None:
        if self._logger:
            self._logger.warning(message)

    def _log_error(self, message: str) -> None:
        if self._logger:
            self._logger.error(message)
