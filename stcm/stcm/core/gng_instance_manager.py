"""Instance management using per-label Growing Neural Gas (GNG) clustering."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Dict, Optional

import numpy as np

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


@dataclass
class ClusterState:
    instance_id: str
    centroid: np.ndarray
    observations: int
    first_seen: float
    last_seen: float
    committed: bool


@dataclass
class _LabelState:
    model: "GrowingNeuralGas"
    clusters: Dict[str, ClusterState]


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
        self._logger = logger
        self._states: Dict[str, _LabelState] = {}
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
                state.clusters[instance_id] = ClusterState(
                    instance_id=instance_id,
                    centroid=pose.copy(),
                    observations=max(1, self._min_obs),
                    first_seen=time.time(),
                    last_seen=time.time(),
                    committed=True,
                )

    def update(
        self,
        label: str,
        centroid: np.ndarray,
        confidence: Optional[float] = None,
        stamp=None,
    ) -> Optional[InstanceAssignment]:
        if not self.enabled:
            return None
        state = self._get_state(label)
        if state is None:
            return None
        centroid_arr = np.asarray(centroid, dtype=np.float64).reshape(3)
        stamp_sec = self._stamp_to_seconds(stamp)

        if self._outlier_gate > 0.0 and state.clusters:
            nearest = min(
                np.linalg.norm(centroid_arr - cluster.centroid)
                for cluster in state.clusters.values()
            )
            if nearest > self._outlier_gate:
                return None

        prob = self._clamp_confidence(confidence)
        if prob is None:
            state.model.insert(centroid_arr[None, :])
        else:
            state.model.insert(centroid_arr[None, :], probabilities=[prob])

        nodes = state.model.nodes()
        if not nodes:
            cluster_state = self._assign_component(label, state, centroid_arr, stamp_sec)
            cluster_state.observations += 1
            cluster_state.last_seen = stamp_sec
            if cluster_state.observations == 1:
                cluster_state.first_seen = stamp_sec
            cluster_state.centroid = self._blend_centroid(cluster_state.centroid, centroid_arr, prob)
            if not cluster_state.committed and self._min_obs > 0:
                if cluster_state.observations >= self._min_obs:
                    cluster_state.committed = True
            stability = self._compute_stability(cluster_state)
            return InstanceAssignment(
                instance_id=cluster_state.instance_id,
                label=label,
                centroid=cluster_state.centroid.copy(),
                stability=stability,
                committed=cluster_state.committed or self._min_obs <= 0,
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

        cluster_state = self._assign_component(label, state, component_centroid, stamp_sec)
        cluster_state.observations += 1
        cluster_state.last_seen = stamp_sec
        if cluster_state.observations == 1:
            cluster_state.first_seen = stamp_sec

        cluster_state.centroid = self._blend_centroid(
            cluster_state.centroid, component_centroid, prob
        )
        if not cluster_state.committed and self._min_obs > 0:
            if cluster_state.observations >= self._min_obs:
                cluster_state.committed = True

        stability = self._compute_stability(cluster_state)
        return InstanceAssignment(
            instance_id=cluster_state.instance_id,
            label=label,
            centroid=cluster_state.centroid.copy(),
            stability=stability,
            committed=cluster_state.committed or self._min_obs <= 0,
        )

    def _get_state(self, label: str) -> Optional[_LabelState]:
        if not self.enabled:
            return None
        state_key = label
        if not self._per_label:
            if not self._warned_per_label:
                self._log_warning(
                    "gng_per_label is false, but global clustering is not implemented; "
                    "falling back to per-label models."
                )
                self._warned_per_label = True
            state_key = label

        state = self._states.get(state_key)
        if state is None:
            config = self._build_config()
            model = GrowingNeuralGas(config)
            state = _LabelState(model=model, clusters={})
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

    def _assign_component(
        self,
        label: str,
        state: _LabelState,
        component_centroid: np.ndarray,
        stamp_sec: float,
    ) -> ClusterState:
        best_id = None
        best_dist = None
        for cluster_id, cluster in state.clusters.items():
            dist = np.linalg.norm(component_centroid - cluster.centroid)
            if best_dist is None or dist < best_dist:
                best_id = cluster_id
                best_dist = dist

        if best_id is not None and (self._merge_distance <= 0.0 or best_dist <= self._merge_distance):
            return state.clusters[best_id]

        instance_id = self._new_instance_id(label)
        cluster_state = ClusterState(
            instance_id=instance_id,
            centroid=component_centroid.copy(),
            observations=0,
            first_seen=stamp_sec,
            last_seen=stamp_sec,
            committed=self._min_obs <= 0,
        )
        state.clusters[instance_id] = cluster_state
        return cluster_state

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
