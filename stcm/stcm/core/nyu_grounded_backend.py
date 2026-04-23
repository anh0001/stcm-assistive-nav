"""Adapter for reusing nyu-grounded-rgbd proposal generation inside STCM."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from PIL import Image

from .label_calibration import choose_label, combine_label_scores


PACKAGE_ROOT = Path(__file__).resolve().parents[3]


@dataclass
class ProposalBatch:
    """STCM-ready proposal outputs from the external detector + SAM backend."""

    boxes_xyxy: torch.Tensor
    masks: torch.Tensor
    scores: torch.Tensor
    phrases: list[str]
    label_score_maps: list[dict[str, float]] | None = None
    crop_embeddings: torch.Tensor | None = None


class NyuGroundedRgbdProposalBackend:
    """Wrap the external nyu-grounded-rgbd detector/SAM stack for STCM."""

    def __init__(
        self,
        *,
        repo_path: str | Path,
        prompt_bank_path: str | Path,
        gdino_model_id: str,
        sam_backend: str,
        sam_model_type: str,
        sam_checkpoint: str | Path | None,
        box_threshold: float,
        text_threshold: float,
        label_rerank_enabled: bool = False,
        label_rerank_model: str = "openai/clip-vit-base-patch32",
        label_margin_min: float = 0.1,
        device: str | torch.device = "cuda",
    ) -> None:
        self.repo_path = self._resolve_path(repo_path)
        self.prompt_bank_path = self._resolve_path(prompt_bank_path)
        self.device = torch.device(device)
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        SAMWrapper, PromptChunk, PromptClass, build_prompt, alias_for_label = self._import_external_symbols()
        self._PromptChunk = PromptChunk
        self._PromptClass = PromptClass
        self._build_prompt = build_prompt
        self._alias_for_label = alias_for_label

        self.chunks, self.class_id_to_label, self.per_chunk_thresholds = self._load_prompt_bank()
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self.label_rerank_enabled = bool(label_rerank_enabled)
        self.label_rerank_model = str(label_rerank_model)
        self.label_margin_min = float(label_margin_min)
        self._rerank_unavailable = False
        self._rerank_processor = None
        self._rerank_model = None
        self._alias_text_features: dict[str, torch.Tensor] = {}
        gdino_model_ref = self._resolve_hf_model_ref(gdino_model_id)
        self.processor = AutoProcessor.from_pretrained(gdino_model_ref, local_files_only=True)
        self.gdino = (
            AutoModelForZeroShotObjectDetection.from_pretrained(gdino_model_ref, local_files_only=True)
            .to(self.device)
            .eval()
        )
        self.sam = SAMWrapper(
            backend=sam_backend,
            checkpoint=str(Path(sam_checkpoint).expanduser()) if sam_checkpoint else None,
            model_type=sam_model_type,
            device=self.device,
        )
        if self.label_rerank_enabled:
            self._init_label_reranker()

    @staticmethod
    def _resolve_path(path_text: str | Path) -> Path:
        path = Path(path_text).expanduser()
        if path.is_absolute():
            return path
        cwd_path = Path.cwd() / path
        if cwd_path.exists():
            return cwd_path
        return PACKAGE_ROOT / path

    @staticmethod
    def _resolve_hf_model_ref(model_ref: str | Path) -> str:
        path = Path(str(model_ref)).expanduser()
        if path.exists():
            return str(path)

        model_id = str(model_ref)
        if "/" not in model_id:
            return model_id

        namespace, repo = model_id.split("/", 1)
        hub_root = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{namespace}--{repo}"
        snapshots_dir = hub_root / "snapshots"
        if not snapshots_dir.exists():
            return model_id

        ref_path = hub_root / "refs" / "main"
        if ref_path.exists():
            snapshot = snapshots_dir / ref_path.read_text(encoding="utf-8").strip()
            if snapshot.exists():
                return str(snapshot)

        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return str(snapshots[-1])
        return model_id

    def _import_external_symbols(self):
        repo_str = str(self.repo_path)
        if repo_str not in sys.path:
            sys.path.insert(0, repo_str)

        from src.models.sam_wrapper import SAMWrapper
        from src.prompts.alias_bank import PromptChunk, PromptClass
        from src.prompts.builders import alias_for_label, build_prompt

        return SAMWrapper, PromptChunk, PromptClass, build_prompt, alias_for_label

    def _load_prompt_bank(
        self,
    ) -> tuple[list[Any], dict[int, str], dict[str, tuple[float, float]] | None]:
        with self.prompt_bank_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}

        chunk_specs = payload.get("chunks", {})
        if not chunk_specs:
            raise ValueError(f"Prompt bank has no chunks: {self.prompt_bank_path}")

        chunks = []
        class_id_to_label: dict[int, str] = {}
        per_chunk_thresholds: dict[str, tuple[float, float]] = {}
        self.label_aliases: dict[str, list[str]] = {}
        next_class_id = 1

        for chunk_name, chunk_spec in chunk_specs.items():
            class_entries = chunk_spec.get("classes", [])
            if not class_entries:
                continue
            classes = []
            for entry in class_entries:
                label = str(entry["label"])
                aliases = [str(alias).strip().lower() for alias in entry.get("aliases", []) if str(alias).strip()]
                if not aliases:
                    aliases = [label.lower()]
                classes.append(
                    self._PromptClass(
                        class_id=next_class_id,
                        name=label,
                        aliases=aliases,
                    )
                )
                class_id_to_label[next_class_id] = label
                self.label_aliases[label] = list(dict.fromkeys(aliases))
                next_class_id += 1

            thresholds = chunk_spec.get("thresholds")
            if thresholds is not None:
                per_chunk_thresholds[str(chunk_name)] = (float(thresholds[0]), float(thresholds[1]))

            chunks.append(self._PromptChunk(name=str(chunk_name), classes=classes))

        return chunks, class_id_to_label, (per_chunk_thresholds or None)

    def _init_label_reranker(self) -> None:
        try:
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            self._rerank_unavailable = True
            self.label_rerank_enabled = False
            return

        try:
            model_ref = self._resolve_hf_model_ref(self.label_rerank_model)
            self._rerank_processor = CLIPProcessor.from_pretrained(model_ref, local_files_only=True)
            self._rerank_model = CLIPModel.from_pretrained(model_ref, local_files_only=True).to(self.device).eval()
        except Exception:
            self._rerank_unavailable = True
            self.label_rerank_enabled = False
            return

        texts = []
        owners = []
        for label, aliases in self.label_aliases.items():
            for alias in aliases:
                texts.append(alias)
                owners.append(label)
        with torch.inference_mode():
            inputs = self._rerank_processor(text=texts, return_tensors="pt", padding=True, truncation=True).to(self.device)
            features = self._rerank_model.get_text_features(**inputs)
            features = F.normalize(features, dim=-1)
        grouped: dict[str, list[torch.Tensor]] = {}
        for owner, feature in zip(owners, features):
            grouped.setdefault(owner, []).append(feature.detach())
        self._alias_text_features = {label: torch.stack(label_features, dim=0) for label, label_features in grouped.items()}

    def _encode_images(self, images: list[Image.Image]) -> torch.Tensor | None:
        if (
            not images
            or self._rerank_unavailable
            or self._rerank_processor is None
            or self._rerank_model is None
        ):
            return None
        with torch.inference_mode():
            inputs = self._rerank_processor(images=images, return_tensors="pt").to(self.device)
            features = self._rerank_model.get_image_features(**inputs)
        return F.normalize(features, dim=-1)

    def _label_scores_from_feature(self, feature: torch.Tensor) -> dict[str, float]:
        scores = {}
        for label, text_features in self._alias_text_features.items():
            sims = torch.matmul(text_features, feature)
            scores[label] = float((sims.max().item() + 1.0) * 0.5)
        return scores

    def _rerank_detections(
        self,
        *,
        pil_image: Image.Image,
        boxes_xyxy: np.ndarray,
        scores: np.ndarray,
        phrases: list[str],
    ) -> tuple[np.ndarray, np.ndarray, list[str], list[dict[str, float]], np.ndarray | None]:
        if (
            not self.label_rerank_enabled
            or self._rerank_unavailable
            or self._rerank_processor is None
            or self._rerank_model is None
            or len(phrases) == 0
        ):
            return boxes_xyxy, scores, phrases, [{label: float(score)} for label, score in zip(phrases, scores)], None

        width, height = pil_image.size
        crop_images: list[Image.Image] = []
        rerank_boxes: list[np.ndarray] = []
        rerank_scores: list[float] = []
        rerank_labels: list[str] = []
        rerank_maps: list[dict[str, float]] = []
        rerank_embeddings: list[np.ndarray] = []

        image_features = self._encode_images([pil_image])
        if image_features is None:
            return boxes_xyxy, scores, phrases, [{label: float(score)} for label, score in zip(phrases, scores)], None
        image_priors = self._label_scores_from_feature(image_features[0])

        valid_indices: list[int] = []
        for idx, box in enumerate(boxes_xyxy):
            x0, y0, x1, y1 = [int(round(value)) for value in box]
            x0 = max(0, min(x0, width - 1))
            y0 = max(0, min(y0, height - 1))
            x1 = max(x0 + 1, min(x1, width))
            y1 = max(y0 + 1, min(y1, height))
            if x1 <= x0 or y1 <= y0:
                continue
            valid_indices.append(idx)
            crop_images.append(pil_image.crop((x0, y0, x1, y1)))
        if len(valid_indices) != len(phrases):
            boxes_xyxy = boxes_xyxy[valid_indices]
            scores = scores[valid_indices]
            phrases = [phrases[idx] for idx in valid_indices]

        crop_features = self._encode_images(crop_images)
        if crop_features is None:
            return boxes_xyxy, scores, phrases, [{label: float(score)} for label, score in zip(phrases, scores)], None

        for box, detector_score, raw_label, crop_feature in zip(boxes_xyxy, scores, phrases, crop_features):
            crop_scores = self._label_scores_from_feature(crop_feature)
            combined = combine_label_scores(
                detector_label=raw_label,
                detector_score=float(detector_score),
                crop_scores=crop_scores,
                image_priors=image_priors,
            )
            decision = choose_label(combined, self.label_margin_min)
            if decision.label is None:
                continue
            rerank_boxes.append(box)
            rerank_scores.append(decision.confidence)
            rerank_labels.append(decision.label)
            rerank_maps.append(decision.label_scores)
            rerank_embeddings.append(crop_feature.detach().cpu().numpy())

        if not rerank_labels:
            return (
                np.empty((0, 4), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                [],
                [],
                np.empty((0, crop_features.shape[-1]), dtype=np.float32),
            )
        return (
            np.stack(rerank_boxes, axis=0).astype(np.float32),
            np.asarray(rerank_scores, dtype=np.float32),
            rerank_labels,
            rerank_maps,
            np.stack(rerank_embeddings, axis=0).astype(np.float32),
        )

    def detect_and_segment(self, image_rgb: np.ndarray) -> ProposalBatch:
        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        height, width = image_rgb.shape[:2]
        detections: list[tuple[int, np.ndarray, float]] = []
        pil_image = Image.fromarray(image_rgb)

        for chunk in self.chunks:
            built_prompt = self._build_prompt(chunk)
            if self.per_chunk_thresholds and chunk.name in self.per_chunk_thresholds:
                box_threshold, text_threshold = self.per_chunk_thresholds[chunk.name]
            else:
                box_threshold, text_threshold = self.box_threshold, self.text_threshold

            inputs = self.processor(images=pil_image, text=built_prompt.text, return_tensors="pt").to(self.device)
            outputs = self.gdino(**inputs)
            try:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(height, width)],
                )[0]
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(
                    outputs,
                    inputs.input_ids,
                    box_threshold=box_threshold,
                    text_threshold=text_threshold,
                    target_sizes=[(height, width)],
                )[0]

            boxes = results["boxes"].detach().cpu().numpy()
            scores = results["scores"].detach().cpu().numpy()
            labels = results.get("text_labels")
            if labels is None:
                labels = results.get("labels", [])

            for box_xyxy, score, label_text in zip(boxes, scores, labels):
                class_id = self._alias_for_label(str(label_text), built_prompt.alias_to_class)
                if class_id is None:
                    continue
                detections.append((int(class_id), np.asarray(box_xyxy, dtype=np.float32), float(score)))

        if not detections:
            return ProposalBatch(
                boxes_xyxy=torch.empty((0, 4), dtype=torch.float32),
                masks=torch.empty((0, 1, height, width), dtype=torch.bool),
                scores=torch.empty((0,), dtype=torch.float32),
                phrases=[],
            )

        boxes_xyxy = np.stack([det[1] for det in detections], axis=0).astype(np.float32)
        phrases = [self.class_id_to_label[int(det[0])] for det in detections]
        scores = np.asarray([float(det[2]) for det in detections], dtype=np.float32)
        boxes_xyxy, scores, phrases, label_score_maps, crop_embeddings = self._rerank_detections(
            pil_image=pil_image,
            boxes_xyxy=boxes_xyxy,
            scores=scores,
            phrases=phrases,
        )
        if not len(phrases):
            return ProposalBatch(
                boxes_xyxy=torch.empty((0, 4), dtype=torch.float32),
                masks=torch.empty((0, 1, height, width), dtype=torch.bool),
                scores=torch.empty((0,), dtype=torch.float32),
                phrases=[],
                label_score_maps=[],
                crop_embeddings=torch.empty((0, 0), dtype=torch.float32),
            )

        self.sam.set_image(image_rgb)
        try:
            sam_results = self.sam.predict_boxes(boxes_xyxy)
        except Exception:
            sam_results = [self.sam.predict_box(box) for box in boxes_xyxy]

        mask_stack = np.stack([result.mask for result in sam_results], axis=0)

        return ProposalBatch(
            boxes_xyxy=torch.as_tensor(boxes_xyxy, dtype=torch.float32),
            masks=torch.as_tensor(mask_stack[:, None, :, :], dtype=torch.bool),
            scores=torch.as_tensor(scores, dtype=torch.float32),
            phrases=phrases,
            label_score_maps=label_score_maps,
            crop_embeddings=None if crop_embeddings is None else torch.as_tensor(crop_embeddings, dtype=torch.float32),
        )
