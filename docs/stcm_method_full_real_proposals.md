# STCM Method — `full-real-proposals` Variant

Detailed method doc for STCM as implemented in this repo, mapped to the
JACIII Jc26-0002 paper. Focus: the **`full-real-proposals`** ablation variant
(headline STCM backend in Table 5, denoted "STCM (ours)"). Also explains
GroundingDINO + MobileSAM (SAM) backbone, CLIP rerank, and the
gravity-aligned plane / RANSAC plane consensus used only in the NYUv2-40
public-benchmark variant (STCM-Depth). Closes by listing **what is missing
from the paper** but present in code (or vice-versa).

---

## 1. Where the variant lives

- Config: [configs/experiments/variants/full-real-proposals.yaml](../configs/experiments/variants/full-real-proposals.yaml)
- Scene-specific prompt banks injected via the `*_real` scenario entry in
  [configs/experiments/manifest.yaml](../configs/experiments/manifest.yaml):
  - S1: [stcm/config/meeting_real_prompts.yaml](../stcm/config/meeting_real_prompts.yaml)
  - S2: [stcm/config/livinglab_real_prompts.yaml](../stcm/config/livinglab_real_prompts.yaml)
  - S3: [stcm/config/outdoor_livinglab_real_prompts.yaml](../stcm/config/outdoor_livinglab_real_prompts.yaml)
- Perception backend adapter: [stcm/stcm/core/nyu_grounded_backend.py](../stcm/stcm/core/nyu_grounded_backend.py)
- CLIP rerank + score fusion: [stcm/stcm/core/label_calibration.py](../stcm/stcm/core/label_calibration.py)
- External backbone repo (vendored detector + SAM wrapper + plane code):
  `/home/anhar/codes/nyu-grounded-rgbd/`
- Builder node (ROS 2): [stcm/stcm/nodes/semantic_map_builder.py](../stcm/stcm/nodes/semantic_map_builder.py)
- Place-GNG: [stcm/stcm/core/place_gng.py](../stcm/stcm/core/place_gng.py)
- Instance-GNG manager: [stcm/stcm/core/gng_instance_manager.py](../stcm/stcm/core/gng_instance_manager.py)
- 3D anchoring (voxel-mode pose lift): [stcm/stcm/map_utils.py:159](../stcm/stcm/map_utils.py#L159)

## 2. Variant identity (`full-real-proposals`)

Same backbone as `full-nyu-proposals` (GroundingDINO-base + MobileSAM + CLIP
rerank) but:

- **Prompt bank = scene-specific, not NYU40.** Detect aliases derived from
  raw ground-truth class names + naïve plural + stripped-modifier paraphrases
  (e.g. `meeting table set → {meeting table set, large meeting table,
  conference table, office meeting table}`). HYBRID v7 banks (see
  `meeting_real_prompts.yaml` header) additionally fold in top-2 probe
  winners per class from `scripts/probe/results/probe_meeting.csv` ranked by
  `hit_rate * avg_score`. No NYU40 alias engineering.
- **No per-class threshold tuning.** Single chunk, uniform
  `thresholds: [0.30, 0.30]` (matches v6 best F1). Per-class
  `box_threshold`/`text_threshold` overrides absent.
- **CLIP rerank ON** for fair compare against `full-nyu-proposals`.
- **No semantic prior** (`supervised_semantic_prior` disabled). Distinguishes
  STCM from `full-nyu-proposals-esanet-prior` and `…-dformerv2-prior` variants.

Headline operating point (from config):
```yaml
perception_backend: nyu_grounded_rgbd
nyu_gdino_model_id: IDEA-Research/grounding-dino-base
nyu_sam_backend: mobilesam
nyu_sam_model_type: vit_t
label_rerank_enabled: true
label_rerank_model: openai/clip-vit-base-patch32
label_margin_min: 0.08
box_threshold: 0.30
text_threshold: 0.30
gng_min_observations_to_commit: 3
instance_label_voting_enabled: true
cross_label_merge_distance_m: 0.6
cross_label_merge_min_cosine: 0.25
```

Reported numbers (paper Table 5, "STCM (ours)" row, n=3 replays):

| Scene | F1 | P | R |
|---|---|---|---|
| S1 meeting | 0.449 ± 0.037 | 0.379 | 0.550 |
| S2 living lab | 0.535 ± 0.028 | 0.488 | 0.594 |
| S3 outdoor | 0.456 ± 0.005 | 0.357 | 0.632 |

Macro mean 0.480 (highest among all five backends compared).

---

## 3. Detection backbone — GroundingDINO

Open-vocabulary transformer detector, conditioned on free-form text prompt.
Repo uses HuggingFace `IDEA-Research/grounding-dino-base` (≈233M params),
loaded with `local_files_only=True` and `TRANSFORMERS_OFFLINE=1` for
reproducible reviewer experiments.

### 3.1 Prompt construction
Each prompt-bank chunk → one detection forward pass. Class entries grouped
by `(box_threshold, text_threshold)` pair (lets per-class thresholds coexist
in one chunk; for `full-real-proposals` all share `(0.30, 0.30)`).
`build_prompt(chunk)` (from the external repo) flattens detect-alias lists
into period-separated prompt string `"meeting table set . large meeting
table . chair . armless chair . ..."` and returns `alias_to_class` map for
post-hoc label resolution.

Code path: `NyuGroundedRgbdProposalBackend._build_detection_chunks()` →
`build_prompt(chunk)` → `self.processor(images=pil, text=prompt)` →
`self.gdino(**inputs)` → `processor.post_process_grounded_object_detection`
([nyu_grounded_backend.py:411-432](../stcm/stcm/core/nyu_grounded_backend.py#L411-L432)).

### 3.2 Output decoding
HF post-processor returns boxes (xyxy in image coords), per-box detection
score, and `text_labels` — the substring of the prompt the box matched.
`alias_for_label(label_text, alias_to_class)` resolves matched substring →
canonical class id. Unmatched detections dropped. Each surviving detection:
`(class_id, box_xyxy, score)`.

### 3.3 Thresholds
- `box_threshold` (config 0.30) — minimum decoder confidence to emit a box.
- `text_threshold` (config 0.30) — minimum max-token similarity between box
  query and prompt token span. Sub-threshold matches dropped during
  `post_process_grounded_object_detection`.

Sensitivity sweep (paper Table 2 STCM-nyu40 row, same backbone): F1 plateau
0.385–0.500 across 0.25–0.40, justifying the 0.30 default.

---

## 4. Segmentation — MobileSAM (vit_t)

SAM head used as **box-prompted segmenter**, not as auto mask generator.
MobileSAM swaps the original SAM ViT-H image encoder for a TinyViT (vit_t,
~9.66M params) trained via decoupled distillation, ~60× faster at near-parity
mIoU on standard sets (cited in paper Ref [41]).

Backbone selection at init: `SAMWrapper(backend="mobilesam",
model_type="vit_t", checkpoint=…)`
([nyu_grounded_backend.py:105-110](../stcm/stcm/core/nyu_grounded_backend.py#L105-L110)).

Per-frame flow:
```python
self.sam.set_image(image_rgb)             # encode image once
sam_results = self.sam.predict_boxes(boxes_xyxy)  # batched box prompts
mask_stack = np.stack([r.mask for r in sam_results], axis=0)
```
([nyu_grounded_backend.py:473-479](../stcm/stcm/core/nyu_grounded_backend.py#L473-L479))

`set_image` runs the (single) ViT encode for the full RGB. `predict_boxes`
calls SAM's prompt encoder + mask decoder per box, reusing cached image
embedding (cheap). Each result has `.mask` (HxW bool) and `.score` (mask
quality from SAM IoU head). Falls back to `predict_box` one-by-one if batch
path errors.

Masks returned as `(N, 1, H, W)` bool tensor for downstream LiDAR
association.

---

## 5. Label rerank — CLIP

Open-vocabulary detectors confuse visually-near classes (e.g. `trash bin` vs
`cardboard box`). Rerank stage adjudicates between competing labels using a
separate CLIP encoder (`openai/clip-vit-base-patch32`).

Setup at backend init
([nyu_grounded_backend.py:270-300](../stcm/stcm/core/nyu_grounded_backend.py#L270-L300)):

1. Build per-label alias text features: for every `rerank_aliases` entry in
   the prompt bank, embed via `CLIPModel.get_text_features`, L2-normalise.
   Store as `dict[label → (n_aliases, dim)]`. Per-label score later = `max`
   cosine over its alias bank.
2. CLIP model + processor cached on GPU, eval mode.

Per-detection rerank
([_rerank_detections, nyu_grounded_backend.py:322-403](../stcm/stcm/core/nyu_grounded_backend.py#L322-L403)):

1. **Global image prior.** Encode full image (`get_image_features`),
   normalise, compute `image_priors: dict[label → score]` via max-cosine
   against each label's alias text features. Cheap scene-level
   regulariser — e.g. an outdoor scene gets low prior on `bottle`.
2. **Crop encoding.** For each detection's xyxy box, crop the PIL image
   (clamped to image bounds), batch-encode crops with CLIP.
3. **Per-crop label scores.** For each crop feature, compute max-cosine
   against every label's alias text features → `crop_scores`.
4. **Combine** with detector confidence
   ([label_calibration.py:34-56](../stcm/stcm/core/label_calibration.py#L34-L56)):
   ```
   combined[label] = (norm(crop_scores[label])
                    + norm(image_priors[label])
                    + (detector_score if label == detector_label else 0)) / 3
   ```
   `normalize_score` maps cosine sims [-1, 1] → [0, 1] via `(s+1)/2` then
   clips to [0, 1].
5. **Decide** ([choose_label, label_calibration.py:59-83](../stcm/stcm/core/label_calibration.py#L59-L83)):
   sort labels by score desc; require top1 > 0 **and**
   `top1 - top2 ≥ label_margin_min` (config: 0.08). If margin too small,
   detection is **dropped entirely** (returns `LabelDecision(label=None,…)`)
   — explicit "I don't know" route rather than picking the noisier label.
6. **Outputs.** Surviving detections get the rerank label (often ≠ the GDINO
   label), a confidence (top score post-fusion), and `label_score_maps` (full
   `dict[label → score]` retained for downstream instance-vote logic). Crop
   embeddings also returned and reused as appearance descriptors for the
   instance manager's cross-label cosine gate (Sec. 7).

### What rerank fixes
Replaces noisy GDINO label string (e.g. open-vocab phrase that doesn't match
any canonical class) with a calibrated decision in the **scene's** target
vocabulary. Confidence = combined score, not raw detector score, so
downstream `gng_min_observations_to_commit` thresholds operate on a more
stable signal.

---

## 6. 3D anchoring — image-projected LiDAR (deployed path)

For S1/S2/S3 robot scenarios the deployed path is **not** monocular depth.
It is image-projected LiDAR (`use_projected_lidar: true`). Monocular
DepthAnything fallback is disabled in this variant
(`use_depth_anything_fallback: false`).

Code: [map_utils.py:159 `pose_in_map_frame_from_projected_mode`](../stcm/stcm/map_utils.py#L159).

Per detection mask M:

1. **Read projected cloud.** ROS topic `/lidar_points_projected`. Each point
   carries `(x, y, z)` in `lidar_link` + `(u, v)` pixel coords in the RGB
   frame (precomputed by an upstream projector that uses extrinsics +
   intrinsics).
2. **Filter inside mask.** Round `(u, v)` to integer, gate by image bounds,
   then `mask = segment[v, u] > 0.5`. Keep only LiDAR points whose
   projection falls inside the SAM mask.
3. **Transform to camera frame** via `rt_camera_inv @ rt_cloud`. Drop
   points with `z ≤ 0` or non-finite — those are behind camera or noise.
4. **Transform to base then map.** Apply `rt_cloud` (lidar → base) then
   `rt_base` (base → map). Result: per-point 3D position in map frame.
5. **Voxelise + mode pick.** Discretise XYZ with `voxel_size = 0.15 m`,
   `unique` over voxel indices, pick voxel with **most points**, break ties
   by **lower median depth** (closer to camera = more reliable). Pose =
   centroid of points inside the winning voxel.

This is the "most-frequent 3D point among the associated LiDAR returns" in
paper Sec. 3.4.3, but the paper says "centroid of the bin containing the
largest number of points" — the implementation also tie-breaks on median
depth, which the paper omits.

Failure mode: if mask has no projected-LiDAR hit (sparse return or full
occlusion), the function returns `None` and the detection is **dropped** in
this variant (DepthAnything fallback disabled). Paper Sec. 3.4.3 mentions
the optional monocular fallback but explicitly states it was not used in
on-robot experiments — matches this variant.

---

## 7. Instance-GNG fusion with rerank-aware voting

After 3D anchoring, the builder ingests `(label, pose_map, confidence,
label_score_map, appearance_embedding)` per detection. The instance manager
[GngInstanceManager](../stcm/stcm/core/gng_instance_manager.py) clusters
detections into persistent object instances. Highlights specific to this
variant:

- **Per-label GNG models** (`gng_per_label=true`). Each label gets its own
  `GrowingNeuralGas` with `dim=3` over `(x, y, z)` in map frame.
- **Cross-label merge** (`instance_label_voting_enabled: true`,
  `cross_label_merge_distance_m: 0.6`, `cross_label_merge_min_cosine: 0.25`).
  When a new detection's label differs from an existing cluster's resolved
  label but their centroids are within 0.6 m, the manager checks **cosine
  similarity between CLIP crop embeddings** — if ≥ 0.25 the detection joins
  the existing cluster (and its `label_votes` get updated), instead of
  spawning a duplicate instance under the new label. Prevents the classic
  "same chair seen from two angles, GDINO emits chair / office chair → two
  duplicate nodes" failure.
- **Label-switch hysteresis.** Even with cross-label merge,
  `instance_label_switch_margin: 0.15` + `…_min_observations: 2` requires
  the candidate new label to lead by 0.15 in vote score and be seen ≥ 2
  consecutive times before the resolved label flips. Stops label flicker
  under noisy rerank decisions.
- **Commit gate.** `gng_min_observations_to_commit: 3` — instance must
  accumulate ≥ 3 observations before being committed as a graph node;
  filters single-frame false positives.

Outlier gate (`gng_outlier_gate_meters`) is **disabled** (0.0 m) — same as
all other variants in the paper. Instance GNG winner-take-all dynamics
absorb spurious returns instead.

---

## 8. Place-GNG (trajectory layer)

Independent of instance-GNG. Code: [place_gng.py](../stcm/stcm/core/place_gng.py).
Operates on the 2D base pose `(x, y)` extracted from `rt_base` each frame.

Per-frame update (paraphrased from `update` + `_apply_match`):

1. Query underlying `GrowingNeuralGas` instance for current pose's winner
   prototype index.
2. If pose-to-winner distance exceeds `place_gng_distance_threshold` (config
   defaults), insert new place node. Distance threshold is **per-variant**
   and lives in the manifest scenario block, not in `full-real-proposals.yaml`
   itself.
3. Else adapt winner (`eps_w`) + topological neighbours (`eps_n`).
4. Edge update: refresh edge to second-best winner (if
   `use_second_best_edge`), and refresh edge from previous winner if it
   changed (transition edges).
5. Aggregate per-frame `label_scores` into the winner node's running score
   vector (max-aggregation by default, `semantic_alpha` EMA blend).

Output published as `/semantic_graph/place_graph` MarkerArray and serialised
into the canonical `stcm.json` along with object instances.

---

## 9. NYUv2-40 public benchmark — STCM-Depth, plane consensus

This is the variant evaluated in paper Sec. 5.5 / Table 8. **Not run on
robot bags.** Code path lives outside `stcm/` proper, in
`/home/anhar/codes/nyu-grounded-rgbd/src/pipeline/`. It is reported here
because the paper conflates it with the deployed pipeline; the differences
matter.

### 9.1 Why a separate variant
NYUv2 has only single-view Kinect RGB-D, no LiDAR. The deployed
image-projected-LiDAR anchor (Sec. 6) cannot run. STCM-Depth substitutes
the **registered Kinect depth channel** for image-projected LiDAR as the
geometric anchor that:
- Constrains mask → 3D back-projection (per-pixel pinhole back-project of
  the depth map).
- Feeds the gravity-aligned plane prior used to label
  floor / wall / ceiling pixels.

Detection + SAM + CLIP rerank are unchanged from Sec. 3-5. Prompt bank is
NYU40-grounded, not scene-real.

### 9.2 Depth features pipeline
File: [depth_features.py](../../nyu-grounded-rgbd/src/pipeline/depth_features.py).

```python
backproject(depth, fx, fy, cx, cy) -> points[H,W,3]   # pinhole
compute_normals(points)             -> normals[H,W,3] # cross of central diffs
sobel_edges(depth)                  -> edges[H,W]
up = [0, -1, 0]                                       # camera +Y points down
up_proj = (normals * up).sum(-1)    # ∈ [-1,1] alignment with gravity-up
```

`up_proj > +0.7` → near-horizontal upward surface (floor candidate).
`up_proj < -0.7` → downward-facing (ceiling). `|up_proj| < 0.3` → vertical
(wall).

### 9.3 RANSAC plane consensus
Function: [fit_dominant_planes](../../nyu-grounded-rgbd/src/pipeline/depth_features.py#L79).

Greedy RANSAC over back-projected points:
- 200 iterations × min 4000 inliers × max 3 planes.
- Sample 3 points → fit plane (normal n via cross product, offset
  d = -n·p₀) → count inliers within `dist_thr = 0.04 m`.
- Refine winning plane: SVD over inlier set, take smallest singular vector
  as refined normal (sign-aligned with RANSAC normal).
- Classify role from `n·up` and mean image-y of inliers:
  - `n·up > 0.7` AND mean y > 0.4·H → **floor**
  - `n·up < -0.7` AND mean y < 0.35·H → **ceiling**
  - `|n·up| < 0.3` → **wall**
  - else `other`
- Remove inliers from the candidate pool, repeat for next plane.

### 9.4 Where plane consensus enters rasterisation
File: [semantic_fusion.py](../../nyu-grounded-rgbd/src/pipeline/semantic_fusion.py).

Three-phase fill:
1. **Phase A — instance fill.** GDINO+SAM+CLIP candidates for instance-like
   classes (everything except `{wall, floor, ceiling, otherstructure}`),
   sorted by combined score descending, painted into the semantic map. Each
   pixel keeps the highest-scoring candidate.
2. **Phase B — structural fill.** If `use_ransac_planes=True` → call
   `_ransac_structural_masks(feat)` → take floor / ceiling / wall masks from
   plane fit, assign NYU40 ids `{2, 22, 1}` with confidences `{0.7, 0.65,
   0.5}` to **unassigned** pixels (does not overwrite instance fill).
   Fallback if RANSAC returns nothing: `_geometry_structural_masks(feat)`
   uses pure `up_proj` heuristic (no RANSAC).
3. **Phase C — residual fill.** Remaining zeros filled by next-best
   candidate that covered them, then geometric fallback, then optional
   dense-CLIP residual fill for tail classes.

Plane consensus thus **does not** rerank instance-class GDINO detections.
It **only** decides structural-class pixels. This is the "gravity-aligned
plane prior" claim in paper Sec. 3.4 / Sec. 9.

### 9.5 Reported lift
Paper Table 8: GDINO-tiny + MobileSAM vanilla 26.11 mIoU → STCM-Depth 39.24
mIoU on NYUv2 654-test, all zero-shot. Lift dominated by floor/wall/ceiling
recovery from the plane stage. Still below supervised RGB-D ESANet (50.3
mIoU [49]).

---

## 10. Canonical output — `stcm.json`

Single JSON, written by builder, consumed by both the ROS 2 stack (graph
planning + Nav2 waypoint resolve) and the LLM grounder (in-context
evidence). Schema (paper Sec. 3.6):

```json
{
  "id": "place_12",
  "label": "kitchen",
  "pose": [2.34, -1.12],
  "scores": {"sink": 0.71, "stove": 0.63, "table": 0.22, ...},
  "visits": 47
}
```

Object instances stored in the same JSON under a parallel block with
instance ids, centroids in map frame, resolved labels, label-vote
distribution, and edge associations to nearest place nodes.

---

## 11. What's missing or imprecise in the paper

Cross-checked paper PDF vs code. Items below are present in code but
under-described, or stated in paper without backing implementation, or
present in only one of {deployed, NYU40} pipeline but described as if
shared.

### 11.1 Method gaps in paper (present in code, not explained)

1. **CLIP image-level prior + 3-way fusion.** Paper says "label rerank"
   exists but never writes the formula. Real fusion is `(crop_score +
   image_prior + (detector_score if matching) ) / 3` with
   `score → (s+1)/2` normalisation
   ([label_calibration.py:34-56](../stcm/stcm/core/label_calibration.py#L34-L56)).
   Image prior is a non-trivial regulariser (suppresses out-of-scene
   classes); not mentioned.
2. **Label margin gate.** `label_margin_min: 0.08` drops detections whose
   top-1 minus top-2 score < 0.08
   ([label_calibration.py:59-83](../stcm/stcm/core/label_calibration.py#L59-L83)).
   This is an explicit abstention path; paper does not describe it. It
   materially affects precision/recall tradeoff and the F1 numbers in
   Table 5.
3. **CLIP crop embeddings double as instance-fusion appearance descriptor.**
   Reused for cross-label cosine gate
   (`cross_label_merge_min_cosine: 0.25`). Paper mentions "instance-GNG +
   gating" in Fig. 2 but never says appearance descriptors come from the
   CLIP rerank head — this is the single most important duplicate-
   suppression mechanism in `full-real-proposals` and absent from the
   methodology narrative.
4. **Label-switch hysteresis.** Margin + min-observations on label flip
   (`instance_label_switch_margin: 0.15`,
   `instance_label_switch_min_observations: 2`). Paper does not describe
   instance label transition logic at all.
5. **Voxel pose lift tie-break by median depth.** Paper Sec. 3.4.3 says
   "centroid of bin with most points". Code adds median-depth tie-break
   ([map_utils.py:254](../stcm/stcm/map_utils.py#L254)). Affects stability
   for thin verticals where multiple voxels tie on count.
6. **Outlier gate disabled, not "small".** Paper Table 1 footnote says
   `D_gate = 0.0 m (disabled)`. Code is consistent but the paper still
   refers to "outlier gating" in Sec. 3.5.1 Step 1 as if it runs. In
   `full-real-proposals` it does not.
7. **Per-class `target_label_thresholds` are merge radii, not detection
   confidences.** Paper Sec. 3.4 + Table 1 conflate these. Config explicitly
   distinguishes detection thresholds (`box_threshold`, `text_threshold`)
   from per-class merge radii used inside `is_nearby_in_map` /
   `target_label_thresholds`
   ([full-real-proposals.yaml:48-58](../configs/experiments/variants/full-real-proposals.yaml#L48-L58)).
8. **Prompt-bank chunking + grouping by threshold pair.** Code groups
   classes by `(box_threshold, text_threshold)` into sub-chunks and runs
   one GDINO forward pass per group
   ([nyu_grounded_backend.py:250-268](../stcm/stcm/core/nyu_grounded_backend.py#L250-L268)).
   Paper presents detection as a single forward pass per frame.
9. **Alias systems are separate.** `detect_aliases` (fed to GDINO prompt
   string, optimised for open-set detection recall) vs `rerank_aliases`
   (fed to CLIP text encoder, optimised for crop-text matching). HYBRID v7
   prompt banks use **different alias sets** for the two roles. Paper Sec. 3
   reads as if one alias set is used for both.
10. **HYBRID v7 prompt construction.** Real banks merge legacy text
    prompts (v6 best F1 baseline) ∪ top-2 probe-CSV winners per class. This
    is offline tuning on the test bag — non-trivial, undisclosed in paper.

### 11.2 Plane consensus / RANSAC — only in STCM-Depth, not deployed

11. **Paper Sec. 3.4 reads as if gravity-aligned plane prior runs in the
    deployed fisheye+LiDAR pipeline.** It does not. Code search of
    `stcm/stcm/**` finds zero `plane` / `gravity` references. The plane
    pipeline lives in `nyu-grounded-rgbd/src/pipeline/{depth_features,
    semantic_fusion}.py` and is invoked only for NYUv2 (STCM-Depth /
    STCM-Trans rows of Table 8). Paper Sec. 3.4.5 footnote does mark this
    as "depth-only variant for public RGB-D benchmarking" but Sec. 3.4
    body text and Fig. 2 do not reflect the separation.
12. **RANSAC parameters undisclosed.** `dist_thr=0.04 m`,
    `min_inliers=4000`, `num_iters=200`, `max_planes=3`,
    role thresholds `±0.7` / `±0.3` on `n·up`, image-y gates
    `0.4·H` / `0.35·H`. None reported in paper.
13. **Three-phase rasterisation order.** Paper does not describe Phase A
    (instance) → B (structural plane fill) → C (residual + dense-CLIP).
    Affects mIoU directly via the plane stage's structural recall.

### 11.3 Implementation realities omitted from results discussion

14. **Detection batched per chunk → effective Hz is lower than reported.**
    Multiple chunks per frame are not reflected in Table 9 GroundingDINO
    164 ms p95 figure. For prompt banks with multiple `(box,text)` groups
    actual per-frame detect time is k·164 ms.
15. **CLIP rerank latency missing from Table 9.** Crop encode + alias-text
    matmul + decision sit in `_rerank_detections`, not separately timed.
    Empirically this is a significant fraction of per-detection cost when
    `N_detections × 1` CLIP forwards happen per frame.
16. **`gng_min_observations_to_commit: 3` interacts with bag length.**
    Short scenes (S2 ≈ 153 s, S3 ≈ 170 s) have fewer revisits, so the
    commit gate suppresses objects seen ≤ 2 times. Not flagged as a
    confound in the F1 discussion.
17. **Fallback path on GNG pause timeout.** If the underlying GNG thread
    fails to pause within 2 s, the manager falls back to distance-based
    clustering and **force-commits** the cluster
    ([gng_instance_manager.py:224-243](../stcm/stcm/core/gng_instance_manager.py#L224-L243)).
    Bypasses the `min_obs` gate. Frequency of this path is not reported.
18. **DepthAnything fallback is disabled** in all reported on-robot
    experiments (`use_depth_anything_fallback: false`). Paper says
    "supported only as an optional fallback when projected LiDAR provides
    no points" but never says it was disabled in the reported runs.
    Detections whose mask has zero projected-LiDAR hits are silently
    dropped.
19. **No coverage of the offline-sequential rosbag replay contract.**
    `offline_sequential: true` + `use_sim_time: true` make per-frame timing
    deterministic relative to bag clock; this is the basis for the
    "replicated under box_threshold perturbation" protocol in Sec. 5.1.
    Paper does not document the determinism contract.

### 11.4 Paper claims not implemented (would-be-nice clarity)

20. **"LLM grounder fetches the JSON in full or as a node/object subset
    projection"** (Sec. 3.6). No code path for subset projection exists in
    this variant's grounder; the full `stcm.json` is loaded.
21. **"Robot's 2D position … extracted from the 6-DOF pose"** (Sec. 3.2).
    Place-GNG uses 2D `(x, y)`, but instance-GNG actually runs in **3D**
    over `(x, y, z)` (`config.dim = 3`,
    [gng_instance_manager.py:308](../stcm/stcm/core/gng_instance_manager.py#L308)).
    The Place vs Instance dimensional split is not stated.
22. **NYU40 mIoU recipe.** Table 8 zero-shot row claims 39.24 mIoU for
    STCM-Depth. Paper does not state which depth fill (RANSAC vs
    `up_proj` heuristic), which num_classes (40 vs the 13-collapse path
    via `_map_40_to_13`), or whether `_drop_small_islands` ran. These
    choices each shift mIoU by ≥ 1 point.

---

## 12. End-to-end frame trace (`full-real-proposals`, deployed scenarios)

For a quick mental model, one frame walks through:

```
RGB image (np.uint8, HxWx3)
    ├─► NyuGroundedRgbdProposalBackend.detect_and_segment
    │      ├─ for each (chunk, threshold_pair):
    │      │     prompt = build_prompt(chunk)
    │      │     boxes, scores, labels = GDINO(image, prompt)
    │      │     filter labels via alias_for_label
    │      ├─ _rerank_detections (CLIP)
    │      │     image_priors  = CLIP_global(image)
    │      │     crop_features = CLIP(crop_per_box)
    │      │     combined      = (crop + prior + det)/3
    │      │     decision      = top1, drop if margin < 0.08
    │      └─ SAM.set_image; predict_boxes(boxes) -> masks
    │  → ProposalBatch(boxes, masks, scores, phrases, label_score_maps, crop_embeddings)
    │
    ├─► _filter_to_target_labels (keep only configured target_labels)
    │
    └─► For each (mask, label, score, label_scores, embed):
           pose_map = pose_in_map_frame_from_projected_mode(
                          projected_cloud, rt_cloud, rt_base,
                          segment=mask, voxel_size=0.15)
           if pose_map is None: drop  # no LiDAR hit
           GngInstanceManager.observe(label, pose_map, score,
                                      label_scores, embed)
              ├─ per-label GNG winner + per-label component
              ├─ cross-label merge if dist≤0.6m and cos≥0.25
              ├─ vote update, hysteresis on label switch
              └─ commit when observations ≥ 3
           PlaceGng.update(rt_base[:2,3], label_scores)
              ├─ place winner + adapt
              ├─ edge refresh
              └─ EMA-blend label scores into winner node
```

`stcm.json` re-serialised on each commit; consumed by Nav2 planner and the
LLM grounder.
