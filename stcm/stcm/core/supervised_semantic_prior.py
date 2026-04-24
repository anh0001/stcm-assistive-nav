"""Frozen supervised RGB-D semantic priors for STCM proposal fusion."""

from __future__ import annotations

from contextlib import contextmanager
from contextlib import redirect_stdout
from dataclasses import dataclass
import copy
import importlib
import io
from pathlib import Path
import sys
import tarfile
from typing import Iterable

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml


NYU40_NAMES: list[str] = [
    "unlabeled",
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor mat",
    "clothes",
    "ceiling",
    "books",
    "refrigerator",
    "television",
    "paper",
    "towel",
    "shower curtain",
    "box",
    "whiteboard",
    "person",
    "night stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
    "otherstructure",
    "otherfurniture",
    "otherprop",
]


DEFAULT_NYU40_TO_STCM: dict[int, tuple[str, ...]] = {
    5: ("chair",),
    7: ("table", "meeting table set"),
    8: ("door", "elevator sliding door"),
    14: ("desk",),
    29: ("cardboard box", "box"),
    39: ("bench",),
}


@dataclass(frozen=True)
class SemanticPriorPrediction:
    """Single-frame semantic prior output in NYU40 label space."""

    label_map: np.ndarray
    confidence: np.ndarray | None = None
    backend: str = "unknown"


@dataclass(frozen=True)
class PriorCandidate:
    """Fallback object proposal extracted from the semantic prior map."""

    label: str
    mask: np.ndarray
    box_xyxy: tuple[float, float, float, float]
    score: float
    label_scores: dict[str, float]


def normalize_label_key(text: str) -> str:
    tokens = []
    es_suffixes = ("ches", "shes", "sses", "xes", "zes")
    for raw_token in str(text).split():
        token = raw_token.strip(" .,;:()[]{}-_\"'\n\t").lower()
        if not token:
            continue
        if len(token) > 3 and token.endswith("ies"):
            token = token[:-3] + "y"
        elif len(token) > 4 and token.endswith(es_suffixes):
            token = token[:-2]
        elif len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.append(token)
    return " ".join(tokens)


def canonical_prior_map(target_labels: Iterable[str]) -> dict[int, tuple[str, ...]]:
    """Return NYU40 -> target-label map, limited to labels active in this run."""

    lookup = {normalize_label_key(label): str(label) for label in target_labels}
    mapped: dict[int, tuple[str, ...]] = {}
    for nyu_id, candidates in DEFAULT_NYU40_TO_STCM.items():
        labels = []
        for candidate in candidates:
            canonical = lookup.get(normalize_label_key(candidate))
            if canonical is not None:
                labels.append(canonical)
        if labels:
            mapped[int(nyu_id)] = tuple(dict.fromkeys(labels))
    return mapped


def semantic_overlap_scores(
    label_map: np.ndarray,
    mask: np.ndarray,
    prior_map: dict[int, tuple[str, ...]],
) -> dict[str, float]:
    """Compute per-STCM-label prior support inside one proposal mask."""

    mask_arr = np.asarray(mask)
    if mask_arr.ndim == 3 and mask_arr.shape[0] == 1:
        mask_arr = mask_arr[0]
    mask_bool = np.asarray(mask_arr, dtype=bool)
    if mask_bool.shape != label_map.shape or not np.any(mask_bool):
        return {}

    masked_labels = np.asarray(label_map, dtype=np.int32)[mask_bool]
    total = float(masked_labels.size)
    scores: dict[str, float] = {}
    for nyu_id, stcm_labels in prior_map.items():
        frac = float(np.count_nonzero(masked_labels == int(nyu_id))) / total
        if frac <= 0.0:
            continue
        for label in stcm_labels:
            scores[label] = max(scores.get(label, 0.0), frac)
    return scores


def fuse_label_scores_with_semantic_prior(
    *,
    phrases: list[str],
    scores,
    masks,
    prediction: SemanticPriorPrediction | None,
    target_labels: Iterable[str],
    canonicalize,
    label_score_maps: list[dict[str, float]] | None = None,
    agreement_boost: float = 0.35,
    disagreement_penalty: float = 0.45,
    min_agreement: float = 0.08,
) -> tuple[object, list[dict[str, float]] | None]:
    """Blend detector scores with NYU40 semantic-prior mask agreement."""

    if prediction is None:
        return scores, label_score_maps

    label_map = np.asarray(prediction.label_map)
    prior_map = canonical_prior_map(target_labels)
    if not prior_map or label_map.size == 0 or len(phrases) == 0:
        return scores, label_score_maps

    score_values = _scores_to_numpy(scores)
    adjusted = score_values.copy()
    maps = [dict(item or {}) for item in (label_score_maps or [{} for _ in phrases])]
    mask_iter = _masks_to_numpy(masks)

    for idx, (phrase, mask) in enumerate(zip(phrases, mask_iter)):
        canonical_label = canonicalize(phrase)
        prior_scores = semantic_overlap_scores(label_map, mask, prior_map)
        for label, prior_score in prior_scores.items():
            maps[idx][label] = max(float(maps[idx].get(label, 0.0)), float(prior_score))
        if canonical_label is None:
            continue

        mapped_labels = {label for labels in prior_map.values() for label in labels}
        if canonical_label not in mapped_labels:
            if canonical_label not in maps[idx]:
                maps[idx][canonical_label] = float(score_values[idx])
            continue

        agreement = float(prior_scores.get(canonical_label, 0.0))
        if agreement >= min_agreement:
            adjusted[idx] = min(1.0, score_values[idx] * (1.0 + agreement_boost * agreement))
        else:
            adjusted[idx] = max(0.0, score_values[idx] * (1.0 - disagreement_penalty))
        maps[idx][canonical_label] = max(float(maps[idx].get(canonical_label, 0.0)), float(adjusted[idx]))

    return _restore_scores_type(scores, adjusted), maps


def extract_prior_candidates(
    *,
    prediction: SemanticPriorPrediction | None,
    target_labels: Iterable[str],
    existing_labels: Iterable[str],
    min_area_px: int = 400,
    max_area_frac: float = 0.25,
    max_per_label: int = 3,
    score: float = 0.35,
) -> list[PriorCandidate]:
    """Create conservative fallback object proposals from prior connected components."""

    if prediction is None:
        return []
    label_map = np.asarray(prediction.label_map, dtype=np.int32)
    if label_map.ndim != 2 or label_map.size == 0:
        return []

    prior_map = canonical_prior_map(target_labels)
    if not prior_map:
        return []

    existing = {normalize_label_key(label) for label in existing_labels}
    image_area = float(label_map.shape[0] * label_map.shape[1])
    max_area_px = max(float(min_area_px), image_area * float(max_area_frac))
    candidates: list[PriorCandidate] = []

    for nyu_id, labels in prior_map.items():
        binary = (label_map == int(nyu_id)).astype(np.uint8)
        if not np.any(binary):
            continue
        num_labels, components, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        for label in labels:
            if normalize_label_key(label) in existing:
                continue
            label_candidates: list[PriorCandidate] = []
            for comp_id in range(1, num_labels):
                area = int(stats[comp_id, cv2.CC_STAT_AREA])
                if area < int(min_area_px) or area > max_area_px:
                    continue
                x = int(stats[comp_id, cv2.CC_STAT_LEFT])
                y = int(stats[comp_id, cv2.CC_STAT_TOP])
                w = int(stats[comp_id, cv2.CC_STAT_WIDTH])
                h = int(stats[comp_id, cv2.CC_STAT_HEIGHT])
                if w <= 1 or h <= 1:
                    continue
                mask = components == comp_id
                label_candidates.append(
                PriorCandidate(
                    label=label,
                    mask=mask,
                    box_xyxy=(float(x), float(y), float(x + w), float(y + h)),
                    score=float(score),
                    label_scores={label: float(score)},
                )
            )
            label_candidates.sort(key=lambda item: int(item.mask.sum()), reverse=True)
            candidates.extend(label_candidates[: int(max_per_label)])
    return candidates


class SupervisedSemanticPrior:
    """Lazy runtime wrapper around frozen supervised NYUv2 RGB-D models."""

    def __init__(
        self,
        *,
        backend: str,
        nyu_grounded_repo_path: str | Path,
        checkpoint_path: str | Path | None = None,
        experiment_config: str | Path | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.backend = str(backend or "none").strip().lower()
        self.nyu_grounded_repo_path = Path(nyu_grounded_repo_path).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser() if checkpoint_path else None
        self.experiment_config = Path(experiment_config).expanduser() if experiment_config else None
        self.device = torch.device(device)
        self._runtime = None

    @property
    def enabled(self) -> bool:
        return self.backend not in ("", "none", "off", "disabled")

    def predict(self, image_bgr: np.ndarray, depth_m: np.ndarray | None) -> SemanticPriorPrediction | None:
        if not self.enabled:
            return None
        if depth_m is None:
            return None
        if self._runtime is None:
            self._runtime = self._build_runtime()
        return self._runtime.predict(image_bgr=image_bgr, depth_m=depth_m)

    def _build_runtime(self):
        if self.backend == "dformerv2":
            return _DFormerV2Runtime(
                nyu_grounded_repo_path=self.nyu_grounded_repo_path,
                checkpoint_path=self.checkpoint_path,
                experiment_config=self.experiment_config,
                device=self.device,
            )
        if self.backend == "esanet":
            return _ESANetRuntime(
                nyu_grounded_repo_path=self.nyu_grounded_repo_path,
                checkpoint_path=self.checkpoint_path,
                experiment_config=self.experiment_config,
                device=self.device,
            )
        raise ValueError(f"Unsupported semantic_prior_backend: {self.backend}")


class _DFormerV2Runtime:
    def __init__(
        self,
        *,
        nyu_grounded_repo_path: Path,
        checkpoint_path: Path | None,
        experiment_config: Path | None,
        device: torch.device,
    ) -> None:
        cfg = _load_experiment_config(
            nyu_grounded_repo_path,
            experiment_config,
            "dformerv2_b_pretrained_nyu40",
        )
        sup_cfg = cfg["supervised"]
        self.repo_dir = _resolve_external_path(nyu_grounded_repo_path, sup_cfg["repo_dir"])
        self.checkpoint_path = _resolve_checkpoint_path(
            nyu_grounded_repo_path,
            checkpoint_path,
            sup_cfg["checkpoint_path"],
            sup_cfg.get("checkpoint_member"),
        )
        self.config_module = str(sup_cfg["config_module"])
        _require_dependencies(sup_cfg.get("deps", []), "dformerv2")
        _require_path(self.repo_dir, "DFormerV2 repository")
        _require_path(self.checkpoint_path, "DFormerV2 checkpoint")
        self.device = device

        with _prepended_sys_path(self.repo_dir):
            cfg_mod = importlib.import_module(self.config_module)
            config = copy.deepcopy(cfg_mod.C)
            config.pad = False
            from models.builder import EncoderDecoder
            from utils.load_utils import load_pretrain

            import torch.nn as nn

            self.config = config
            self.model = EncoderDecoder(cfg=config, criterion=None, norm_layer=nn.BatchNorm2d)
            load_pretrain(self.model, str(self.checkpoint_path), strict=False)
        self.model.to(self.device).eval()

    def predict(self, *, image_bgr: np.ndarray, depth_m: np.ndarray) -> SemanticPriorPrediction:
        image_bgr = np.asarray(image_bgr, dtype=np.uint8)
        height, width = image_bgr.shape[:2]
        depth_u8 = _depth_to_dformerv2_uint8(depth_m)
        modal_x = cv2.merge([depth_u8, depth_u8, depth_u8])

        rgb = _normalize_image(image_bgr, self.config.norm_mean, self.config.norm_std)
        depth_rgb = _normalize_image(modal_x, np.array([0.48, 0.48, 0.48]), np.array([0.28, 0.28, 0.28]))
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).float()[None].to(self.device)
        )
        depth_tensor = (
            torch.from_numpy(np.ascontiguousarray(depth_rgb.transpose(2, 0, 1))).float()[None].to(self.device)
        )

        with torch.inference_mode():
            logits = _dformerv2_multiscale_logits(
                self.model, image_tensor, depth_tensor, self.config, self.device
            )
            if logits.shape[-2:] != (height, width):
                logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=True)
            probs = logits.softmax(dim=1)
            confidence, pred = probs.max(dim=1)
        label_map = (pred[0].detach().cpu().numpy().astype(np.uint8) + 1)
        return SemanticPriorPrediction(
            label_map=label_map,
            confidence=confidence[0].detach().cpu().numpy().astype(np.float32),
            backend="dformerv2",
        )


class _ESANetRuntime:
    def __init__(
        self,
        *,
        nyu_grounded_repo_path: Path,
        checkpoint_path: Path | None,
        experiment_config: Path | None,
        device: torch.device,
    ) -> None:
        cfg = _load_experiment_config(
            nyu_grounded_repo_path,
            experiment_config,
            "esanet_pretrained_nyu40_fallback",
        )
        sup_cfg = cfg["supervised"]
        self.repo_dir = _resolve_external_path(nyu_grounded_repo_path, sup_cfg["repo_dir"])
        self.checkpoint_path = _resolve_checkpoint_path(
            nyu_grounded_repo_path,
            checkpoint_path,
            sup_cfg["checkpoint_path"],
            sup_cfg.get("checkpoint_member"),
        )
        self.height = int(sup_cfg.get("height", 480))
        self.width = int(sup_cfg.get("width", 640))
        _require_dependencies(sup_cfg.get("deps", []), "esanet")
        _require_path(self.repo_dir, "ESANet repository")
        _require_path(self.checkpoint_path, "ESANet checkpoint")
        self.device = device

        with _isolated_src_import(self.repo_dir):
            from src.build_model import build_model
            from src.preprocessing import get_preprocessor

            class Args:
                pass

            args = Args()
            args.dataset = "nyuv2"
            args.pretrained_on_imagenet = False
            args.height = self.height
            args.width = self.width
            args.valid_full_res = False
            args.batch_size = 1
            args.batch_size_valid = 1
            args.workers = 0
            args.last_ckpt = ""
            args.pretrained_scenenet = ""
            args.modality = "rgbd"
            args.pretrained_dir = str(self.repo_dir / "trained_models" / "imagenet")
            args.encoder = "resnet34"
            args.encoder_block = "NonBottleneck1D"
            args.nr_decoder_blocks = [3]
            args.encoder_depth = None
            args.activation = "relu"
            args.encoder_decoder_fusion = "add"
            args.context_module = "ppm"
            args.channels_decoder = 128
            args.decoder_channels_mode = "decreasing"
            args.fuse_depth_in_rgb_encoder = "SE-add"
            args.upsampling = "learned-3x3-zeropad"
            args.he_init = False
            args.finetune = None
            args.freeze = 0
            self.preprocessor = get_preprocessor(
                height=self.height,
                width=self.width,
                depth_mean=2841.94941272766,
                depth_std=1417.2594281672277,
                depth_mode="refined",
                phase="test",
            )
            with redirect_stdout(io.StringIO()):
                self.model, _ = build_model(args, n_classes=40)
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device).eval()

    def predict(self, *, image_bgr: np.ndarray, depth_m: np.ndarray) -> SemanticPriorPrediction:
        image_rgb = cv2.cvtColor(np.asarray(image_bgr, dtype=np.uint8), cv2.COLOR_BGR2RGB)
        height, width = image_rgb.shape[:2]
        sample = {
            "image": image_rgb,
            "depth": _depth_to_esanet_uint16(depth_m).astype(np.float32),
        }
        sample = self.preprocessor(sample)
        image_tensor = sample["image"][None].to(self.device)
        depth_tensor = sample["depth"][None].to(self.device)
        with torch.inference_mode():
            logits = self.model(image_tensor, depth_tensor)
            logits = F.interpolate(logits, size=(height, width), mode="bilinear", align_corners=False)
            probs = logits.softmax(dim=1)
            confidence, pred = probs.max(dim=1)
        label_map = (pred[0].detach().cpu().numpy().astype(np.uint8) + 1)
        return SemanticPriorPrediction(
            label_map=label_map,
            confidence=confidence[0].detach().cpu().numpy().astype(np.float32),
            backend="esanet",
        )


def _load_experiment_config(repo_path: Path, config_path: Path | None, default_name: str) -> dict:
    path = config_path
    if path is None:
        path = repo_path / "configs" / "experiment" / f"{default_name}.yaml"
    elif not path.is_absolute():
        path = repo_path / path
    _require_path(path, "supervised experiment config")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_external_path(repo_path: Path, path_text: str | Path) -> Path:
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    return repo_path / path


def _resolve_checkpoint_path(
    repo_path: Path,
    explicit_path: Path | None,
    configured_path: str | Path,
    member: str | None = None,
) -> Path:
    archive_or_checkpoint = explicit_path or _resolve_external_path(repo_path, configured_path)
    if member is None:
        return archive_or_checkpoint
    target = archive_or_checkpoint.parent / member
    if target.exists():
        return target
    _require_path(archive_or_checkpoint, "checkpoint archive")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive_or_checkpoint, "r:gz") as tar:
        tar.extract(member, path=archive_or_checkpoint.parent)
    return target


def _require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def _require_dependencies(deps: list[dict[str, str]] | None, backend: str) -> None:
    missing = []
    for dep in deps or []:
        module_name = str(dep.get("module", "")).strip()
        pip_name = str(dep.get("pip", module_name)).strip()
        if module_name and importlib.util.find_spec(module_name) is None:
            missing.append(pip_name or module_name)
    if missing:
        packages = " ".join(missing)
        raise RuntimeError(
            f"Missing Python dependencies for semantic_prior_backend={backend}: {packages}. "
            f"Install them in the STCM Python user base, for example: "
            f"PYTHONUSERBASE=$HOME/.local/stcm_sys_py310 python3 -m pip install --user {packages}"
        )


@contextmanager
def _prepended_sys_path(path: Path):
    path_text = str(path)
    sys.path.insert(0, path_text)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_text)
        except ValueError:
            pass


@contextmanager
def _isolated_src_import(path: Path):
    saved_modules = {
        name: module
        for name, module in sys.modules.items()
        if name == "src" or name.startswith("src.")
    }
    for name in list(saved_modules):
        sys.modules.pop(name, None)
    with _prepended_sys_path(path):
        try:
            yield
        finally:
            for name in [name for name in sys.modules if name == "src" or name.startswith("src.")]:
                sys.modules.pop(name, None)
            sys.modules.update(saved_modules)


def _normalize_image(image: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    image_f = image.astype(np.float64) / 255.0
    return (image_f - mean) / std


def _depth_to_dformerv2_uint8(depth: np.ndarray) -> np.ndarray:
    depth = np.asarray(depth, dtype=np.float32)
    valid = np.isfinite(depth) & (depth > 0.0)
    if not np.any(valid):
        return np.zeros(depth.shape, dtype=np.uint8)
    out = np.zeros(depth.shape, dtype=np.uint8)
    finite = depth[valid]
    d_min = float(finite.min())
    d_max = float(finite.max())
    if abs(d_max - d_min) < 1e-8:
        out[valid] = 255
        return out
    normalized = (depth[valid] - d_min) / (d_max - d_min)
    out[valid] = np.clip(np.round((1.0 - normalized) * 255.0), 0, 255).astype(np.uint8)
    return out


def _depth_to_esanet_uint16(depth: np.ndarray) -> np.ndarray:
    depth_mm = np.nan_to_num(np.asarray(depth, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    depth_mm = np.clip(np.round(depth_mm * 1000.0), 0, np.iinfo(np.uint16).max)
    return depth_mm.astype(np.uint16)


def _dformerv2_multiscale_logits(model, images, modal_xs, config, device):
    _, _, h, w = images.shape
    scaled_logits = torch.zeros(images.shape[0], config.num_classes, h, w, device=device)
    scales = [float(s) for s in config.eval_scale_array]
    flip = bool(config.eval_flip)
    divisor = 0
    for scale in scales:
        new_h = int(np.ceil((scale * h) / 32.0) * 32)
        new_w = int(np.ceil((scale * w) / 32.0) * 32)
        scaled_images = F.interpolate(images, size=(new_h, new_w), mode="bilinear", align_corners=True)
        scaled_modal_xs = F.interpolate(modal_xs, size=(new_h, new_w), mode="bilinear", align_corners=True)
        logits = model(scaled_images, scaled_modal_xs)
        logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True)
        scaled_logits += logits
        divisor += 1
        if flip:
            flip_images = torch.flip(scaled_images, dims=(3,))
            flip_modal_xs = torch.flip(scaled_modal_xs, dims=(3,))
            logits = model(flip_images, flip_modal_xs)
            logits = torch.flip(logits, dims=(3,))
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=True)
            scaled_logits += logits
            divisor += 1
    return scaled_logits / float(max(divisor, 1))


def _scores_to_numpy(scores) -> np.ndarray:
    if scores is None:
        return np.empty((0,), dtype=np.float32)
    if hasattr(scores, "detach"):
        return scores.detach().cpu().numpy().astype(np.float32, copy=True)
    return np.asarray(scores, dtype=np.float32).copy()


def _restore_scores_type(original, values: np.ndarray):
    if original is None:
        return values
    if hasattr(original, "detach"):
        return torch.as_tensor(values, dtype=original.dtype, device=original.device)
    if isinstance(original, np.ndarray):
        return values.astype(original.dtype, copy=False)
    return values.tolist()


def _masks_to_numpy(masks) -> list[np.ndarray]:
    if masks is None:
        return []
    if hasattr(masks, "detach"):
        arr = masks.detach().cpu().numpy()
    else:
        arr = np.asarray(masks)
    return [arr[idx] for idx in range(arr.shape[0])]
