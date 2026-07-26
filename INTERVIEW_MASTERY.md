# Adversarial Attacks on ML Models — Complete Interview Mastery Document

> **Project:** Adversarial YOLO — Object Detection with Adversarial Attack Simulation  
> **Repo:** `Adversarial-attacks-on-ML-models`  
> **Author (per README):** Kowshik Thatinati  
> **Purpose of this doc:** Senior-level interview preparation — reverse-engineered from every source file.

---

## ⚠️ CRITICAL: What You Must Know Before Any Interview

**Read this first.** The README and UI claim things the code does not fully deliver. A strong candidate knows both the story *and* the truth.

| Claim (docs/UI) | Actual code reality |
|-----------------|---------------------|
| "Multi-step I-FGSM against YOLO/Faster R-CNN" | `attack_utils.py` (web app) never calls the detector during attack — it pushes bbox pixels toward grey (0.5) |
| "Model gradient-based FGSM" | `attack()` in `attack_utils.py` ignores `calculate_loss()` entirely |
| `attack_mode`: entire_image vs bounding_boxes_only | Parameter accepted but **never used** in `attack()` |
| "Person → car misclassification" in demos | **Post-hoc visual manipulation** in `app.py` lines 212–218 and CLI scripts — not from model output |
| "Confidence drop 40–65%" | Partially real (pixel perturbation may affect model) + **artificially subtracted** in visualization layer |
| CLI uses same attack as web | **No** — CLI imports `attacks.fgsm.FGSMAttack`; web uses `attack_utils.FGSMAttack` |
| Training pipeline | **None** — pretrained weights only |
| COCO mAP evaluation | **None** — count/confidence heuristics only |
| `custom_images/` dataset | Referenced in docs; **not in repo** (likely local/gitignored) |

**Interview-elite move:** Proactively say: *"The demo layer applies visualization post-processing for pedagogical clarity; the core attack module uses region-focused pixel perturbation. I'd refactor to true white-box I-FGSM with end-to-end gradients through the detector loss."*

---

# PHASE 1 — PROJECT RECONSTRUCTION

## 1.1 Actual Project Objective

Build an **educational/research demonstration** that:
1. Runs **object detection** on images using **YOLOv5** (Ultralytics) and **Faster R-CNN** (torchvision ResNet50-FPN v2).
2. Generates **adversarial perturbations** intended to degrade detections (lower confidence, fewer boxes, misclassification).
3. Presents results via **Gradio web UI** and **CLI scripts** with side-by-side visualization.

This is **not** a production security product, **not** a trained-from-scratch detector, and **not** a rigorous robustness benchmark on COCO.

## 1.2 Business Problem

**Stakeholder pain:** Organizations deploying computer vision (autonomous vehicles, surveillance, retail analytics) assume detector outputs are reliable. Adversarial ML shows that **imperceptible input changes can break detection pipelines**, creating liability, safety, and security risk.

**Business value this project demonstrates:**
- Risk awareness for ML product teams
- Need for adversarial robustness testing before deployment
- Architecture comparison (one-stage vs two-stage detectors) under stress
- Prototype for red-team / model evaluation tooling

**Who would care:** ML platform teams, autonomous systems safety engineers, security researchers, compliance auditors evaluating AI systems.

## 1.3 Research Problem

**Core research question:** How vulnerable are standard pretrained object detectors (YOLOv5, Faster R-CNN) to gradient-based adversarial perturbations, and how do architectures differ in robustness?

**Sub-questions the project *attempts* to address:**
- Can FGSM-style attacks reduce detection confidence?
- Does a two-stage detector (FRCNN) behave differently than one-stage (YOLO)?
- Does region-focused vs full-image perturbation matter?

**What the project does NOT rigorously answer:**
- Transferability across models
- Certified robustness bounds
- Physical-world (patch) attacks
- mAP degradation on a held-out dataset

## 1.4 Users

| User type | How they use it |
|-----------|-----------------|
| Student / researcher | Web UI or CLI to explore adversarial CV |
| Presenter / demo | Gradio side-by-side before/after |
| Developer extending attacks | Modular `models/`, `attacks/`, `attack_utils.py` |

No authentication, multi-tenancy, or API consumers — single-user local app.

## 1.5 Inputs

| Input | Format | Notes |
|-------|--------|-------|
| Image | JPG/PNG via Gradio PIL or file path (CLI) | RGB, uint8 |
| Model choice | `"YOLOv5"` or `"Faster R-CNN"` | Dropdown |
| Attack toggle | Boolean | Web only |
| Epsilon | Float 0.01–0.20 (web clamps invalid to 0.05) | Perturbation budget in [0,1] pixel space |
| Attack mode | `entire_image` / `bounding_boxes_only` | **UI only — not implemented in attack** |

**YOLO preprocessing:** Resize to **640×640**, normalize to [0,1], CHW tensor.  
**FRCNN preprocessing:** Native resolution, `ToTensor()` → [0,1]; torchvision handles internal normalization.

## 1.6 Outputs

| Output | Description |
|--------|-------------|
| Original detection plot | Matplotlib image with green boxes, labels, confidence |
| Adversarial detection plot | Same, potentially with modified scores/classes |
| Metrics HTML | Object count, avg confidence, attack success flag |
| CLI PNG | `results/yolo_attack_<name>.jpg` or `faster_rcnn_attack_*` |

**Detection tensor format:** `[N, 6]` = `[x1, y1, x2, y2, confidence, class_id]`

## 1.7 Data Flow (End-to-End)

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ User Image  │────▶│ Preprocess       │────▶│ Detector        │
│ (PIL/path)  │     │ YOLO:640² / FRCNN│     │ YOLO or FRCNN   │
└─────────────┘     └──────────────────┘     └────────┬────────┘
                                                       │
                                                       ▼
                                              ┌─────────────────┐
                                              │ Detections      │
                                              │ [x1,y1,x2,y2,   │
                                              │  conf, cls]     │
                                              └────────┬────────┘
                                                       │
                       ┌───────────────────────────────┘
                       ▼ (if attack enabled)
              ┌─────────────────┐     ┌─────────────────┐
              │ FGSMAttack      │────▶│ Adv image tensor│
              │ (10 iter pixel  │     │ [0,1] clamped   │
              │  perturbation)  │     └────────┬────────┘
              └─────────────────┘              │
                                               ▼
                                      ┌─────────────────┐
                                      │ Re-inference    │
                                      └────────┬────────┘
                                               │
                                               ▼
                              ┌────────────────────────────────┐
                              │ Visual post-process (app/CLI)    │
                              │ Artificial conf↓, class swap   │
                              └────────────────┬───────────────┘
                                               ▼
                              ┌────────────────────────────────┐
                              │ plot_detections / metrics      │
                              └────────────────────────────────┘
```

## 1.8 Architecture

**Pattern:** Modular monolith — Python scripts, no microservices, no database.

```
app.py                    → Gradio UI orchestrator
run_yolo.py               → CLI entry (YOLO path)
run_faster_rcnn.py        → CLI entry (FRCNN path)

model_loader.py           → Lazy model cache, device mgmt (FRCNN path in app)
models/yolo.py            → YOLOModel wrapper (YOLO path in app + CLI)
models/faster_rcnn.py     → FasterRCNNModel wrapper (CLI)

attack_utils.py           → FGSMAttack (WEB — pixel heuristic)
attacks/fgsm.py           → FGSMAttack (CLI — gradient attempt)
attacks/utils.py          → IoU, match_detections

inference_utils.py        → Preprocess, infer, metrics (FRCNN in app)
visualize.py              → Matplotlib bbox plotting (CLI)
```

**Dual-path design in `app.py`:**
- YOLO → `YOLOModel` directly (bug-fix: ensures CLI parity)
- Faster R-CNN → `ModelLoader` + `InferenceUtils`

## 1.9 Training Pipeline

**There is none.**

- YOLOv5: pretrained `yolov5s.pt` (YOLOModel) or `yolov5su.pt`/`yolov5m.pt` (ModelLoader)
- Faster R-CNN: `fasterrcnn_resnet50_fpn_v2(weights='DEFAULT')` — COCO pretrained

No fine-tuning, no custom dataset ingestion, no loss logging, no checkpoints saved by this repo.

## 1.10 Inference Pipeline

**YOLO:**
1. `cv2.imread` → BGR→RGB → resize 640×640
2. Tensor `(1,3,640,640)` on CPU/GPU
3. Convert back to uint8 numpy for Ultralytics API
4. `model(img_np)` → boxes, scores, classes
5. Concatenate to `[N,6]` tensor

**Faster R-CNN:**
1. `cv2.imread` → RGB → `ToTensor()` → `(1,3,H,W)`
2. `model([tensor.squeeze(0)])` in eval mode
3. Filter `scores >= threshold` (0.3 app, 0.5 default in FasterRCNNModel)
4. Stack boxes, scores, labels

---

## 1.11 Project Story (Narrative)

*"I built a research demo that stress-tests two mainstream object detectors — YOLOv5 and Faster R-CNN — against adversarial perturbations. The goal wasn't to ship a product; it was to make model fragility tangible for engineers who've only seen clean validation mAP. You upload a street scene, pick a detector, crank epsilon, and watch confidence collapse and boxes disappear. I unified a Gradio UI and CLI around shared model wrappers, fixed a bunch of integration bugs between Ultralytics and PyTorch autograd, and structured the code so you can swap in PGD or Carlini-Wagner later. Honest caveat: the current attack module uses region-focused pixel optimization rather than full white-box detector gradients — I documented where visualization post-processing enhances the demo effect."*

## 1.12 Technical Story

*"Architecturally it's a thin orchestration layer over Ultralytics and torchvision. Two detector backends share a common detection format `[x1,y1,x2,y2,conf,class]`. The web app deliberately routes YOLO through `YOLOModel` — same class the CLI uses — because ModelLoader had preprocessing mismatches that caused label and coordinate bugs. Attack generation runs 10 micro-steps with step size ε/10, applying signed gradients from a custom loss that pushes bbox regions toward neutral grey plus L2 distance regularization. After attack, we re-run inference. Metrics compare detection count and mean confidence. The legacy `attacks/fgsm.py` path attempts true gradient attacks via negative confidence (YOLO) or training-mode loss (FRCNN) but breaks on YOLO's non-differentiable Ultralytics pipeline — that's why `attack_utils.py` exists as a workaround."*

## 1.13 Business Story

*"Every company shipping vision AI needs to answer: what happens if someone adversarially perturbs our camera feed? This project is a lightweight red-team sandbox — upload an image, see if your off-the-shelf detector fails. For a CTO, the takeaway is twofold: (1) pretrained SOTA models are not robust by default, and (2) one-stage vs two-stage detectors fail differently — YOLO keeps boxes but loses confidence; FRCNN often drops boxes entirely due to score thresholding. That informs whether you invest in adversarial training, input sanitization, ensemble detectors, or human-in-the-loop for safety-critical paths."*

---

# PHASE 2 — MASTER EXPLANATION (Elevator Pitches)

## 30 Seconds

*"I built a demo that attacks YOLO and Faster R-CNN with adversarial noise. You upload a photo, it detects objects, perturbs the image in detected regions, and shows you how confidence and detections degrade — side by side in a web UI. It's for educating teams on why robustness testing matters before deploying vision models."*

## 1 Minute

*"The project evaluates how fragile standard object detectors are. I integrated YOLOv5 through Ultralytics and Faster R-CNN through torchvision, wrapped both behind a common detection format, and built FGSM-style iterative perturbations targeted at bounding box regions. There's a Gradio frontend and CLI tools. When you attack an image, you typically see confidence scores drop and some detections disappear — especially on Faster R-CNN because it threshold-filters low-confidence boxes. I also compared one-stage versus two-stage architectures because they fail in different ways. It's research-educational, not production-hardened, but the modular design makes it easy to plug in stronger attacks like PGD or adversarial training defenses."*

## 3 Minutes

*"Business context: computer vision in autonomous systems and security cameras assumes reliable detections. Adversarial ML shows tiny pixel changes can fool models while looking normal to humans.*

*What I built: a Python framework with three entry points — Gradio app, YOLO CLI, Faster R-CNN CLI. Images flow through preprocessing — YOLO resizes to 640 squared, FRCNN keeps native resolution. We run inference, get boxes in a unified six-column tensor, then optionally run a ten-step perturbation attack with budget epsilon.*

*The attack focuses on detected regions — pushing pixels toward ambiguous values so the detector loses confidence. We re-infer on the adversarial image and compute simple metrics: detection count, average confidence, whether the attack 'succeeded.'*

*Key engineering decisions: separate YOLOModel wrapper used consistently in web and CLI after debugging Ultralytics integration; lazy model loading with GPU fallback; visualization with clipped bounding boxes so labels don't render outside the frame.*

*Key findings from testing: YOLO tends to keep detections but lower confidence; Faster R-CNN often drops from dozens of detections to a handful because scores fall below threshold. That architectural difference matters for anyone choosing detectors for adversarial environments.*

*Limitations I'll own in an interview: no rigorous COCO mAP evaluation, no physical-world attacks, and the attack loss is a simplified region heuristic rather than full end-to-end detector backprop — which I'd prioritize fixing next."*

## 5 Minutes

*(Expand 3-minute version with:)*
- **Stack:** PyTorch 2+, Ultralytics 8+, torchvision, Gradio 4+, OpenCV, Matplotlib
- **Models:** YOLOv5s (80 COCO classes), FRCNN ResNet50-FPN v2 (COCO pretrained)
- **Attack math:** x' = clip(x + (ε/T)·sign(∇_x L), 0, 1) over T=10 steps; L = Σ_regions ||region - 0.5||² + λ||x' - x||²
- **Dual FGSMAttack classes:** legacy gradient version in `attacks/`, pragmatic version in `attack_utils.py`
- **Bug fixes documented in PROJECT_COMPLETION_SUMMARY:** class names, same-image bug, epsilon slider, bbox clipping
- **Metrics logic:** attack success if count drops OR confidence drops with same count
- **Deployment today:** `python app.py` on localhost:7860, no Docker/K8s in repo

## 10 Minutes

*(Full walkthrough for deep-dive rounds — combine all phases below, emphasizing:)*
1. Problem → approach → architecture diagram
2. Live demo flow step-by-step through `app.run_attack()`
3. YOLO vs FRCNN internal differences (anchor-based dense vs RPN+ROI)
4. Why Ultralytics breaks standard FGSM (inference API, no grad through NMS easily)
5. What true white-box object detection attack requires (DAGGER, RPGA, targeted loss on classification + localization)
6. Production gap analysis and roadmap
7. Honest assessment of visualization post-processing in app.py
8. How you'd evaluate properly: mAP@0.5, attack success rate, IoU-matched class flip rate on COCO val

---

# PHASE 3 — FILE BY FILE BREAKDOWN

## Root Level

### `app.py` (~522 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | Primary user-facing Gradio web application |
| **What it does** | Orchestrates upload → detect → attack → visualize → metrics |
| **Interactions** | `ModelLoader`, `InferenceUtils`, `YOLOModel`, `attack_utils.FGSMAttack` |
| **If removed** | No web UI |
| **Key class** | `AdversarialAttackApp` |
| **Key methods** | `run_attack()`, `plot_detections()`, `create_interface()` |
| **Hidden deps** | Temp file for YOLO path; matplotlib Agg backend; **lines 212-218, 252-258 fake visual attack effects** |

### `attack_utils.py` (~165 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | Attack implementation used by **web app** |
| **What it does** | 10-iter region greying + L2 reg; sign gradient step |
| **Interactions** | Called from `app.py`; imports `iou` from attacks.utils (unused in attack) |
| **If removed** | Web attack breaks |
| **Key class** | `FGSMAttack` |
| **Dead code** | `calculate_loss()`, `create_bounding_box_mask()`, `attack_mode` unused |
| **Hidden deps** | Does NOT call `self.model` during `attack()` |

### `inference_utils.py` (~217 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | Unified preprocess/infer/metrics for ModelLoader path |
| **Key class** | `InferenceUtils` |
| **Key methods** | `preprocess_image()`, `run_yolo_inference()`, `run_faster_rcnn_inference()`, `calculate_metrics()` |
| **If removed** | FRCNN web path breaks; YOLO web path unaffected |
| **Hidden deps** | YOLO inference re-converts tensor→numpy because Ultralytics expects uint8 HWC |

### `model_loader.py` (~91 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | Lazy-load and cache models; COCO class name list |
| **Key class** | `ModelLoader` |
| **If removed** | FRCNN in app breaks |
| **Inconsistency** | Loads `yolov5su.pt` but app uses `YOLOModel` with `yolov5s.pt` for YOLO |

### `visualize.py` (~154 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | CLI plotting (`plot_comparison`, `plot_detections`) |
| **If removed** | CLI loses saved figure output |
| **Note** | Duplicates plotting logic in `app.py` (not DRY) |

### `run_yolo.py` (~92 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | CLI demo for YOLO |
| **Uses** | `models.yolo.YOLOModel`, `attacks.FGSMAttack` (legacy), `visualize.plot_comparison` |
| **Hidden behavior** | Lines 48-58: artificial conf reduction + class swap for display |

### `run_faster_rcnn.py` (~109 lines)
| Aspect | Detail |
|--------|--------|
| **Why exists** | CLI demo for FRCNN |
| **Uses** | `FasterRCNNModel`, `attacks.FGSMAttack`, `visualize` |
| **Same visual fakery** | Lines 70-79 |

### `requirements.txt`
| Aspect | Detail |
|--------|--------|
| **Deps** | torch≥2.0, torchvision≥0.15, ultralytics≥8, gradio≥4, numpy, opencv, matplotlib, Pillow |
| **Missing** | pytest, logging framework, docker, fastapi (if API needed) |

### `README.md`, `QUICK_START.md`, `PROJECT_COMPLETION_SUMMARY.md`
| Aspect | Detail |
|--------|--------|
| **Purpose** | User docs; completion summary lists bug fixes and claimed metrics |
| **Caution** | Overstate attack sophistication vs code — know the delta |

### `.gitignore`
| Aspect | Detail |
|--------|--------|
| **Ignores** | `results/`, `*.pt`, `data/`, venv, `.env` |
| **Impact** | Models downloaded at runtime; no bundled test images in repo |

---

## `models/`

### `models/yolo.py` — `YOLOModel`
| Aspect | Detail |
|--------|--------|
| **Why** | Canonical YOLO wrapper — proven working in CLI and app |
| **Methods** | `preprocess()`, `predict()`, `process_image()` |
| **Model weights** | `yolov5s.pt` |
| **If removed** | YOLO CLI + web YOLO path break |
| **Dead code** | `non_max_suppression()`, `xywh2xyxy()` — legacy stubs |

### `models/faster_rcnn.py` — `FasterRCNNModel`
| Aspect | Detail |
|--------|--------|
| **Why** | FRCNN wrapper for CLI |
| **Model** | `fasterrcnn_resnet50_fpn_v2` COCO weights |
| **Default conf** | 0.5 in `process_image` (CLI passes 0.3 via args) |
| **If removed** | FRCNN CLI breaks |

### `models/__init__.py`
Empty package marker.

---

## `attacks/`

### `attacks/fgsm.py` — `FGSMAttack` (LEGACY / CLI)
| Aspect | Detail |
|--------|--------|
| **Why** | Original gradient-based FGSM attempt |
| **YOLO path** | Calls `model.model(adv_image)` + NMS — often fails silently, returns original image |
| **FRCNN path** | Sets model to `.train()`, passes targets, sums detection losses — **valid white-box approach** |
| **Mask** | Applies perturbation only inside bboxes |
| **If removed** | CLI attack breaks |

### `attacks/utils.py`
| Aspect | Detail |
|--------|--------|
| **Functions** | `iou()`, `match_detections()`, `plot_detections()` |
| **Used by** | `fgsm.py`; `iou` imported in `attack_utils.py` but unused |
| **`match_detections`** | **Never called** anywhere — ready for proper eval |

### `attacks/__init__.py`
Exports `FGSMAttack`, `iou`, `match_detections`.

---

# PHASE 4 — EVERY DESIGN DECISION

## D1: YOLOv5 vs Other Detectors (YOLOv8, DETR, SSD)

| | |
|--|--|
| **Chosen** | YOLOv5 via Ultralytics |
| **Why** | Mature ecosystem, easy pretrained weights, fast demo, well-documented |
| **Alternatives** | YOLOv8/v11, DETR, EfficientDet, SSD |
| **Rejected because** | YOLOv8 API changes; DETR slower, harder attack integration; SSD older |
| **Trade-off** | Not latest YOLO; Ultralytics abstraction complicates gradients |
| **Better alternative when** | Need SOTA accuracy → YOLOv8; need transformer research → DETR |

**Follow-ups:** Why v5s not v5m? Why not YOLO-NAS? How does anchor-free change attack surface?

---

## D2: Faster R-CNN as Second Model

| | |
|--|--|
| **Chosen** | `fasterrcnn_resnet50_fpn_v2` torchvision |
| **Why** | Canonical two-stage baseline; native PyTorch — easier grad-based attack in legacy fgsm |
| **Alternatives** | Cascade R-CNN, Mask R-CNN, RetinaNet |
| **Trade-off** | Slow (~5× YOLO); many low-threshold detections inflate "attack success" |
| **Better when** | Need instance segmentation → Mask R-CNN |

**Follow-ups:** Why ResNet50 not ResNet101? Why FPN v2 not v1?

---

## D3: FGSM / I-FGSM vs PGD vs C&W

| | |
|--|--|
| **Chosen** | Multi-step signed gradient (claimed I-FGSM) |
| **Why** | Educational simplicity, fast, famous baseline (Goodfellow 2014) |
| **Alternatives** | PGD (stronger), C&W (optimization), BPDA, patch attacks |
| **Rejected** | PGD/C&W more compute; patch attacks need different pipeline |
| **Trade-off** | Weak attack vs strong defenses; current impl isn't true FGSM on YOLO |
| **Better when** | Robust eval → PGD-100; physical world → Adversarial Patch |

**Follow-ups:** L∞ vs L2 budget? Untargeted vs targeted? Step size α = ε/T optimal?

---

## D4: Two FGSMAttack Implementations

| | |
|--|--|
| **Chosen** | Split: `attacks/fgsm.py` (CLI) + `attack_utils.py` (web) |
| **Why** | Ultralytics YOLO broke gradient flow; workaround for demo stability |
| **Alternatives** | Single unified attack module; Differentiable YOLO wrapper |
| **Trade-off** | Maintenance burden; inconsistent behavior CLI vs web |
| **Better** | One module with backend-specific loss functions |

**Follow-ups:** Which is used where? Did you unify them? Technical debt plan?

---

## D5: Region-Focused vs Full-Image Attack

| | |
|--|--|
| **Chosen (documented)** | Both modes via `attack_mode` radio |
| **Actual** | Always perturbs only inside original detection bboxes (implicit in loss loop) |
| **Why (intent)** | Imperceptibility — change less pixels |
| **Alternatives** | Full image FGSM; sparse patch |
| **Trade-off** | Region attack fails if initial detection misses object |
| **Better** | Implement mask from `create_bounding_box_mask` for `bounding_boxes_only`; full image for stronger attack |

---

## D6: Gradio vs Streamlit vs FastAPI+React

| | |
|--|--|
| **Chosen** | Gradio 4 Blocks API |
| **Why** | Fastest path to upload/slider/image comparison UI |
| **Alternatives** | Streamlit, Flask+React, Jupyter |
| **Trade-off** | Not production-scalable; limited auth/multi-user |
| **Better when** | Production → FastAPI + object storage + job queue |

---

## D7: YOLO Preprocess 640×640 Fixed Resize

| | |
|--|--|
| **Chosen** | Stretch/squash to 640×640 |
| **Why** | YOLOv5 default input |
| **Alternatives** | Letterbox resize (aspect preserve) — Ultralytics default in some paths |
| **Trade-off** | Aspect ratio distortion affects bbox coords |
| **Better** | Letterbox + coord inverse transform |

---

## D8: Dual Code Path in app.py (YOLOModel vs ModelLoader)

| | |
|--|--|
| **Chosen** | YOLO bypasses ModelLoader |
| **Why** | Bug fix — ModelLoader YOLO path caused wrong labels/coords in UI |
| **Trade-off** | Architectural inconsistency |
| **Better** | Fix ModelLoader to match YOLOModel; single path |

---

## D9: Visual Post-Processing of Detections (Conf/Class Manipulation)

| | |
|--|--|
| **Chosen** | Artificially reduce confidence; swap person→car every N detections |
| **Why** | Demo impact when real attack effect weak |
| **Alternatives** | Show raw model output only; tune attack until real effect visible |
| **Trade-off** | **Scientific integrity risk** — must disclose in interviews |
| **Better** | Remove fakery; improve attack; report honest metrics |

---

## D10: No Automated Tests / No COCO Eval

| | |
|--|--|
| **Chosen** | Manual testing only |
| **Why** | Educational scope, time constraints |
| **Trade-off** | Regressions undetected; metrics anecdotal |
| **Better** | pytest + small golden image; COCO subset eval script |

---

## D11: Confidence Thresholds (YOLO 0.35, FRCNN 0.3)

| | |
|--|--|
| **Chosen** | Different defaults per model |
| **Why** | Empirical tuning for demo visuals |
| **Trade-off** | Attack success partially artifact of threshold |
| **Better** | Report PR curves; sweep thresholds |

---

## D12: GPU Auto-Detect with CPU Fallback

| | |
|--|--|
| **Chosen** | `cuda if available else cpu` |
| **Why** | Accessibility |
| **Trade-off** | Slow on CPU for FRCNN |
| **Better** | Explicit `--device` flag everywhere |

---

# PHASE 5 — DEEP MODEL ANALYSIS

## Model A: YOLOv5 (Ultralytics, yolov5s)

### Internal Architecture
1. **Backbone:** CSPDarknet — Cross Stage Partial connections for gradient flow + efficiency
2. **Neck:** PANet — path aggregation multi-scale fusion
3. **Head:** Decoupled detection at 3 scales (P3, P4, P5) — predicts (x,y,w,h,obj,class logits) per anchor
4. **Post-process:** Confidence filter + NMS (non-differentiable)

### Mathematical Intuition
For each anchor cell: predict bounding box offsets + objectness σ(obj) + softmax over 80 classes. Loss at training: box regression (CIoU) + obj BCE + cls BCE.

**Inference:** argmax class × objectness → threshold → NMS suppresses overlapping boxes.

### Inputs / Outputs
| | |
|--|--|
| **Input** | RGB image, effectively 640×640 (project resizes) |
| **Output** | Variable N detections: xyxy pixels, conf ∈ [0,1], cls ∈ {0..79} |

### Training (Pretrained — Not in Project)
Trained on COCO with multi-scale augmentation, mosaic, etc. (Ultralytics recipe).

### Inference in This Project
Tensor → numpy uint8 → `YOLO(img_np)` — **breaks computation graph** for attacks.

### Complexity
- **Time:** O(HW) backbone + O(anchors) head; ~20-30ms GPU for small
- **Memory:** ~14-28M params (yolov5s); ~1-2GB VRAM inference

### Advantages
Fast, single-shot, good real-time baseline

### Disadvantages
NMS breaks differentiability; anchor heuristic; less accurate than two-stage on small objects

### "What If" Chain
| Question | Answer |
|----------|--------|
| Why YOLO not Faster R-CNN for production speed? | YOLO: one forward pass; FRCNN: RPN + per-RoI head |
| Why YOLOv5 not v8? | Project timing/simplicity; v8 unified API |
| Why small not medium? | YOLOModel uses `yolov5s`; faster demo |
| Why not DETR? | DETR: O(N²) attention; slower; different attack (query-based) |
| Attack YOLO vs FRCNN? | YOLO dense predictions harder to target; FRCNN explicit cls loss in train mode |

---

## Model B: Faster R-CNN ResNet50-FPN v2

### Internal Architecture
1. **Backbone:** ResNet50 extracts {C2,C3,C4,C5} feature maps
2. **FPN:** Top-down lateral connections → {P2..P5} multi-scale
3. **RPN:** Slides anchors, predicts objectness + box deltas → proposals
4. **ROI Align:** Bilinear sampling from feature map per proposal (differentiable)
5. **ROI Head:** FC layers → class logits + box refinement
6. **Post-process:** Score threshold + NMS

### Mathematical Intuition
**RPN loss:** L = L_cls + λ L_box on anchors  
**Detection loss:** L = L_cls + L_box on sampled RoIs (foreground/background balance)

**Inference:** proposals → refine → softmax class → filter score > τ

### Inputs / Outputs
| | |
|--|--|
| **Input** | Variable H×W RGB tensor [0,1] |
| **Output** | Dict: boxes (N,4), scores (N,), labels (N,) — label 0 = background unused in output |

### Training in Project
None — uses COCO DEFAULT weights.

### Inference
Eval mode, no grad. ~100-150ms GPU.

### Complexity
- **Time:** O(HW) + O(proposals × head) — heavier than YOLO
- **Memory:** ~44M+ params; 2-4GB VRAM

### Advantages
Higher precision; explicit RoI classification; **train mode exposes sum of losses** — usable for attack in `attacks/fgsm.py`

### Disadvantages
Slow; many false positives at low threshold; two-stage pipeline complex

### "What If" Chain
| Question | Answer |
|----------|--------|
| Why FRCNN v2 not v1? | v2: improved box head, better COCO AP |
| Why ResNet50 not 101? | Speed/VRAM trade-off |
| Why not RetinaNet? | One-stage with focal loss — middle ground; FRCNN classic teaching example |
| Attack via train mode? | Valid trick: `model.train()` + target boxes → backprop input gradient |
| Why detections drop more? | Scores cross below τ=0.3; YOLO keeps low-conf if above 0.25 |

---

## Attack "Model": FGSMAttack (attack_utils.py)

### Not a neural network — optimization procedure

**Loss:** L = 100 · Σ_r mean((r - 0.5)²) + 5 · mean((x' - x)²)

**Update:** x' ← clip(x' + (ε/10)·sign(∇L), 0, 1)

**Intuition:** Grey pixels (0.5) are ambiguous — weaken features detectors rely on.

**Limitation:** Not aligned with detector loss landscape — attack is **heuristic**, not adversarial in the ML-security sense for YOLO path.

---

# PHASE 6 — INTERVIEWER ATTACK MODE

## Challenge 1: Architecture

**Q:** Why two separate attack classes and two YOLO loading paths? Isn't that sloppy?

| Level | Answer |
|-------|--------|
| **Expected** | "There were integration bugs; we used workarounds." |
| **Strong** | "YOLO web path uses YOLOModel for parity with CLI after ModelLoader caused coordinate bugs. Attack split because Ultralytics inference isn't differentiable — attack_utils uses region heuristic while attacks/fgsm keeps gradient attempt for FRCNN. I'd consolidate behind a strategy interface." |
| **Elite** | "Technical debt from autograd boundary at Ultralytics numpy API. I'd introduce `AttackBackend` ABC: `WhiteBoxTorchvision`, `BlackBoxQuery`, `HybridRegion`. Refactor ModelLoader to delegate to YOLOModel. Add contract tests on detection tensor format. Document autograd limitations in ADR." |

---

## Challenge 2: Dataset

**Q:** You claim COCO-trained models but never evaluate on COCO. How do you know anything is statistically valid?

| Level | Answer |
|-------|--------|
| **Expected** | "It's a demo on sample images." |
| **Strong** | "Correct — metrics are qualitative on custom images. Proper eval would run PGD on COCO val, report mAP drop and ASR. I'd use match_detections with IoU>0.5 to pair boxes before/after attack." |
| **Elite** | "Sample size N=1 images is anecdotal. I'd define ASR = fraction of images where matched detection class flips or conf drops >50%. Report confidence intervals on 500-image subset. Separate threshold sensitivity analysis — FRCNN drops are partly threshold artifacts." |

---

## Challenge 3: Metrics

**Q:** Your attack success metric is trivially gameable.

| Level | Answer |
|-------|--------|
| **Expected** | "We check if count or confidence decreased." |
| **Strong** | "`calculate_metrics` uses count drop OR avg conf drop — no IoU matching, no mAP. False positives on FRCNN inflate counts. I'd implement COCO eval + per-box matched degradation." |
| **Elite** | "Also, app.py post-processes detections for display — metrics can reflect manipulated tensors. For production eval I'd freeze `adv_detections_raw` vs `adv_detections_display` and only metricize raw." |

---

## Challenge 4: Model Choice

**Q:** YOLOv5 is outdated. Why should I trust your robustness conclusions?

| Level | Answer |
|-------|--------|
| **Expected** | "It still shows models can be fooled." |
| **Strong** | "Architecture class matters more than version — one-stage vs two-stage. YOLOv5 is representative of deployed real-time detectors. I'd extend to YOLOv8 and open-vocabulary models." |
| **Elite** | "Robustness isn't monotonic with SOTA — newer models may be more or less robust depending on training (adversarial aug). The project is a methodology scaffold, not a benchmark paper." |

---

## Challenge 5: Hyperparameters

**Q:** Why ε=0.15? Why 10 steps? Any ablation?

| Level | Answer |
|-------|--------|
| **Expected** | "Default from README, works visually." |
| **Strong** | "ε in [0,1] pixel space ≈ 38/255 at 0.15 — standard FGSM magnitude. 10 steps with ε/T is I-FGSM heuristic. No ablation in repo — I'd sweep ε ∈ {0.03,0.07,0.15,0.3} and steps ∈ {1,5,10,20}." |
| **Elite** | "L∞ budget should be reported in /255 units for literature comparison. Fixed step α=ε/T doesn't guarantee max perturbation binds — should clip cumulative delta to ε ball (PGD-style)." |

---

## Challenge 6: Scalability

**Q:** This runs one image locally. How does it scale to 1M images?

| Level | Answer |
|-------|--------|
| **Expected** | "It doesn't — it's a demo." |
| **Strong** | "Batch CLI with multiprocessing, GPU queue, S3 in/out. FRCNN bottleneck → use TensorRT or batch YOLO only for screening." |
| **Elite** | "Async job system: Kafka queue, Ray/K8s workers, model server with dynamic batching, result store in Parquet with detection diffs. Attack compute dominates — cache clean inference, parallelize attack per image." |

---

## Challenge 7: Deployment

**Q:** Would you deploy this Gradio app to production?

| Level | Answer |
|-------|--------|
| **Expected** | "No, research only." |
| **Strong** | "Gradio for demos only. Production red-team pipeline needs auth, rate limits, audit logs, sandboxed GPU workers." |
| **Elite** | "Separate concerns: inference microservice (TorchServe/Triton), attack worker pool, results DB. SOC2: no arbitrary code upload without scanning. Model weights pinned by hash." |

---

# PHASE 7 — HIDDEN FOLLOW-UPS (Recursive Chains)

## Chain A: "Why FGSM?"

| Level | Question | Answer |
|-------|----------|--------|
| L1 | Why gradient-based attacks? | Efficient vs random search; exploit model sensitivity direction |
| L2 | Why sign of gradient? | L∞ optimal single-step direction (maximize change per pixel under L∞) |
| L3 | Why multi-step? | Single-step underutilizes ε budget; I-FGSM iteratively applies smaller steps |
| L4 | Complexity per step? | One forward + one backward O(params) — but detection adds NMS discontinuity |
| L5 | Optimize for detection? | Need loss on classification scores + box regression; DAGGER/RPGA papers |
| L6 | Why not BPDA? | For defenses with non-differentiable preprocessing |
| L7 | Why not AutoAttack? | SOTA robustness eval — ensemble of attacks |

## Chain B: "Why YOLO?"

| Level | Question | Answer |
|-------|----------|--------|
| L1 | One-stage vs two-stage? | YOLO: dense prediction; FRCNN: propose then classify |
| L2 | Why anchors? | Prior box shapes reduce search space |
| L3 | Anchor-free YOLO? | v8+ simplifies; attacks target center-ness + distance |
| L4 | NMS role? | Removes duplicates; non-differentiable — attack surface |
| L5 | Differentiable NMS? | Soft-NMS, DIoU-NMS approximations for grad attacks |
| L6 | Why Ultralytics API? | Convenience vs research control |
| L7 | Export to ONNX attack? | Possible but NMS op support varies |

## Chain C: "Why region attack?"

| Level | Question | Answer |
|-------|----------|--------|
| L1 | Sparse vs dense perturbation? | Sparse: imperceptible; dense: stronger |
| L2 | L0 vs L∞ budget? | Region attack ≈ lower L0 (fewer pixels) |
| L3 | Failure mode? | Missed detections → no mask → attack fails |
| L4 | Universal patch? | Region attack is image-specific; patch is universal |
| L5 | Physical realizability? | Digital L∞ ≠ printable patch — needs expectation over transforms |

## Chain D: "Why confidence drops on FRCNN?"

| Level | Question | Answer |
|-------|----------|--------|
| L1 | Softmax calibration? | Scores aren't true probabilities |
| L2 | Threshold effect? | τ=0.3 hides weak detections — recall drop |
| L3 | RPN vs ROI sensitivity? | Perturbation may break proposal quality first |
| L4 | Score vs count metric? | Count drop conflates threshold with attack |
| L5 | Fix? | Report full score distribution pre/post |

## Chain E: "Why Gradio?"

| Level | Question | Answer |
|-------|----------|--------|
| L1 | vs REST API? | Gradio faster for ML demos |
| L2 | Concurrency? | Single process; not for load |
| L3 | Security? | Upload = untrusted input — size limits, malware scan |
| L4 | Scale? | Put nginx + gunicorn-like pattern or migrate to FastAPI |

---

# PHASE 8 — THEORY CONNECTIONS

| Component | ML | DL | CV | NLP | Graph | Stats | Probability | Optimization | MLOps | Distributed | SWE | System Design |
|-----------|----|----|----|----|-------|-------|-------------|--------------|-------|-------------|-----|---------------|
| YOLO inference | ✓ classification | CNN | detection | — | — | threshold | softmax conf | — | model registry | — | wrapper pattern | inference svc |
| FRCNN | two-stage | ResNet+FPN | RoI Align | — | — | NMS | softmax | train-mode loss | versioning | — | modular models | batch infer |
| FGSM | adversarial ML | backprop | pixel perturb | text attacks analog | — | — | — | constrained opt | — | — | strategy pattern | — |
| IoU metric | eval | — | bbox overlap | — | — | — | — | — | monitoring | — | utils | — |
| Gradio UI | — | — | viz | — | — | — | — | — | demo deploy | — | MVC-ish | monolith |
| Epsilon budget | robustness | — | — | — | — | sensitivity | — | L∞ ball | — | — | config | — |
| Preprocess | data pipeline | tensor ops | color space | tokenization analog | — | normalization | — | — | data validation | — | separation | ETL |
| No training | transfer learning | fine-tune analog | — | — | — | — | — | — | CI for models | — | — | — |

**Cross-domain analog:** FGSM on images ↔ FGSM on text embeddings (PromptAttack). Object detection attack ↔ structured output attack (not just cls).

---

# PHASE 9 — FAILURE ANALYSIS

## Failure Point 1: Attack Has No Effect
| | |
|--|--|
| **Root cause** | No detections → attack returns unchanged; or heuristic loss too weak |
| **Detection** | Metrics show FAILED; identical images |
| **Mitigation** | Fallback full-image attack; lower conf threshold for initial detect |
| **Monitoring** | Log `len(detections)` pre-attack |

## Failure Point 2: CUDA OOM
| | |
|--|--|
| **Root cause** | FRCNN on large images + grad mode |
| **Detection** | Exception in logs |
| **Mitigation** | CPU fallback; resize max dimension 1333 (COCO standard) |
| **Recovery** | Retry on CPU; clear cache `torch.cuda.empty_cache()` |

## Failure Point 3: Misleading Demo Metrics
| | |
|--|--|
| **Root cause** | Visual post-processing in app.py |
| **Detection** | User compares to raw CLI without visual layer |
| **Mitigation** | Remove fakery; flag `display_mode` vs `raw` |
| **Monitoring** | Unit test: raw conf unchanged when attack disabled |

## Failure Point 4: Model Download Fails
| | |
|--|--|
| **Root cause** | No network; gitignore excludes `.pt` |
| **Detection** | YOLO load exception |
| **Mitigation** | Bundle weights in artifact store; checksum verify |

## Failure Point 5: Port Conflict (7860)
| | |
|--|--|
| **Root cause** | Another Gradio instance |
| **Detection** | OSError on launch |
| **Mitigation** | Fallback 7861 (implemented) |

## Failure Point 6: Ultralytics Version Break
| | |
|--|--|
| **Root cause** | API change in ultralytics≥8 |
| **Detection** | CI import test fails |
| **Mitigation** | Pin version in requirements.txt |

## Failure Point 7: Adversarial Robustness False Sense
| | |
|--|--|
| **Root cause** | Weak attack → team thinks model is safe |
| **Detection** | Red team with PGD finds failures |
| **Mitigation** | Document attack limitations; use AutoAttack benchmark |

**Recovery strategy (production):** Rollback model version; enable backup detector ensemble; human review queue for low-confidence frames.

---

# PHASE 10 — PRODUCTION READINESS

## Deploy
```text
Current: python app.py (localhost)

Production path:
1. Dockerize: CUDA base + requirements.txt pin
2. FastAPI backend: POST /attack {image, model, epsilon}
3. Triton/TorchServe for model inference
4. Gradio or React frontend → API
5. Secrets via env; no .env in image
```

## Scale
- Horizontal: stateless workers behind load balancer
- GPU: 1 worker ≈ 2-5 img/sec (YOLO), 0.5-1 (FRCNN)
- Queue: SQS/RabbitMQ for batch red-team jobs

## Monitor
- Latency p50/p99 per model
- Attack success rate (properly defined)
- GPU util, OOM rate
- Model version in each response header

## Retrain
- Not applicable today (pretrained only)
- For robust models: adversarial training on COCO with PGD aug

## Version
- Pin `yolov5s.pt` SHA256
- `torchvision` version maps to FRCNN weights
- MLflow model registry

## Rollback
- Blue/green model deployment
- Previous weight artifact in S3 versioned bucket

## Prevent Drift
- Input distribution monitoring (brightness, blur stats)
- Periodic re-run attack suite on golden images
- Alert if avg conf shifts > ε

## Reduce Latency
- TensorRT YOLO
- Half precision FP16
- Skip FRCNN for realtime; YOLO only

## Reduce Cost
- Spot GPU instances for batch
- CPU YOLO for low-priority scans
- Cache clean inference results

---

# PHASE 11 — RESUME DEFENSE

## Ownership Verification Questions

**Q:** Walk me through `attack_utils.FGSMAttack.attack()` line by line.  
**Ideal:** Explain 10-iter loop, region loss toward 0.5, L2 reg, sign grad, clamp — and admit model isn't called.

**Q:** Why did you bypass ModelLoader for YOLO in app.py?  
**Ideal:** Specific bugs — class names, coordinate space 640×640 mismatch, temp file pattern for YOLOModel.process_image.

**Q:** What broke with the original FGSM on YOLO?  
**Ideal:** Ultralytics expects numpy; `model.model(adv_image)` graph differs; NMS non-differentiable; loss often None → returns original image.

**Q:** How long did integration take? What was hardest?  
**Ideal:** Gradio PIL vs cv2 color spaces; bbox clipping; making adversarial image actually differ from original.

## Bluffing Exposure Questions

**Q:** Report mAP drop on COCO val after attack.  
**Ideal:** "We didn't run that — I'd use pycocotools eval with adversarial val set."

**Q:** What's the exact I-FGSM equation you implemented?  
**Ideal:** Honest: iterative sign grad on custom region loss, not standard classifier FGSM on cross-entropy.

**Q:** Show me where attack_mode is implemented.  
**Ideal:** "It's accepted in constructor but not wired — I'd use create_bounding_box_mask to fix."

**Q:** Is person→car from the model or post-processing?  
**Ideal:** **Must admit:** post-processing in app.py lines 216-218 for demo when attack_enabled.

## Shallow Understanding Exposure

**Q:** Difference between RPN and ROI head losses?  
**Ideal:** RPN: anchor objectness + box; ROI: sampled proposal classification + refinement.

**Q:** Why Faster R-CNN drops more detections?  
**Ideal:** Score threshold + softmax sensitivity; not inherently "weaker architecture."

**Q:** L∞ epsilon 0.15 in [0,1] vs [0,255]?  
**Ideal:** Project uses normalized tensors; 0.15 ≈ 38.25/255 pixel max change per step before clamp accumulation.

---

# PHASE 12 — CTO ROUND

## Opening CTO Question
*"You spent engineering time on a demo that fakes some attack results. Why should I fund adversarial robustness work based on this?"*

**Elite answer:** *"This repo is an awareness and architecture prototype, not an audit tool. What it correctly proves: off-the-shelf detectors fail under perturbation, architectures fail differently, and our ML pipeline needs a real eval harness. I'd propose Phase 2: rigorous PGD on 5K COCO images, remove display-layer manipulation, add Triton serving with regression gates blocking deploy if ASR > X%. Cost: 2 engineers × 6 weeks. Risk reduction: prevents shipping vision systems vulnerable to trivial digital attacks."*

## CTO Follow-Ups

| Topic | Question | Strong Response |
|-------|----------|-----------------|
| ROI | Build vs buy robustness tools? | Microsoft Counterfit, CleverHans exist — custom needed for our detector + domain |
| Liability | If our car CV fails under attack? | Document known limits; defense in depth; not sole safety layer |
| Cost | GPU red-team at scale? | $X/month spot; amortize vs recall incident cost |
| Timeline | Production-ready when? | 8-12 weeks with eval harness + CI gates |
| Team | Who maintains? | ML platform + security partnership |
| Risk | Biggest technical lie in project? | Proactively: visualization enhancement — fix first |
| Strategy | One-stage or two-stage for our product? | Latency → YOLO; precision + adversarial? → ensemble |
| Data | Need adversarial training data? | PGD-generated aug on our fleet distribution, not just COCO |
| Compliance | EU AI Act implications? | Robustness testing documentation for high-risk CV |
| Exit | When is good enough? | Defined ASR threshold on domain-specific test suite |

---

# PHASE 13 — PROJECT BIBLE

## Everything I Need To Know About This Project

### One-Sentence Summary
Educational Python framework to detect objects with YOLOv5/Faster R-CNN, apply iterative region-focused perturbations, and visualize detection degradation via Gradio and CLI.

### Architecture (Memorize)
- **Monolithic Python app**
- **Two detectors, two attack modules, dual web paths for YOLO**
- **Detection format:** `[x1,y1,x2,y2,conf,class_id]`
- **No training, no DB, no cloud**

### Dataset
- User-provided images (JPG/PNG)
- Models pretrained on **COCO 80 classes**
- No bundled eval set in repo

### Training
- **None** — uses `yolov5s.pt` and `fasterrcnn_resnet50_fpn_v2` DEFAULT weights

### Inference
- YOLO: 640×640 resize, Ultralytics API
- FRCNN: native res, torchvision list input
- Conf thresholds: YOLO 0.35 (inference_utils), FRCNN 0.3

### Attack
- **Web:** `attack_utils.py` — 10-step region greying, no model grad
- **CLI:** `attacks/fgsm.py` — attempts model grad; works better on FRCNN
- **ε:** 0.01–0.20 web; default 0.15 YOLO CLI, 0.20 FRCNN CLI
- **Display layer artificially modifies conf/class in app + CLI**

### Deployment
- `python app.py` → http://127.0.0.1:7860
- Not production-ready

### Metrics
- Detection count before/after
- Mean confidence before/after
- Attack success = count drop OR conf drop
- **No mAP, no IoU-matched ASR in production code**

### Trade-offs
| Pros | Cons |
|------|------|
| Fast demo | Not rigorous benchmark |
| Two architectures | Inconsistent code paths |
| Modular structure | Two FGSMAttack classes |
| GPU support | Ultralytics blocks true white-box YOLO attack |
| Good viz | Fake demo enhancements |

### Limitations (Say These Aloud)
1. Not true end-to-end FGSM on YOLO
2. `attack_mode` UI not implemented
3. Visual misclassification sometimes faked
4. No statistical eval
5. No physical-world attacks
6. No defenses implemented
7. Model weight inconsistency (s vs su vs m)
8. `match_detections` unused
9. No tests
10. README overclaims vs code

### Future Work (Impress Interviewers)
1. Unified `AttackStrategy` interface
2. True white-box: torchvision YOLO export or DiffYOLO
3. PGD + COCO mAP eval pipeline
4. Implement `match_detections` for ASR
5. Remove visualization fakery
6. Adversarial training defense module
7. Docker + CI + pytest golden files
8. YOLOv8/v11 support
9. Physical patch attack module
10. W&B logging for attack sweeps

### Common Interview Questions
1. What is FGSM? → Single-step L∞ attack using sign(∇_x L)
2. Why object detection attacks are harder? → Multiple outputs, NMS, structured prediction
3. YOLO vs FRCNN? → One-stage speed vs two-stage accuracy
4. What's epsilon? → Perturbation budget in input space
5. How do you preprocess for YOLO? → 640×640, [0,1] tensor
6. Is this production-ready? → No — demo/research
7. How measure attack success properly? → IoU-matched box conf/cls change + mAP
8. Biggest technical challenge? → Differentiable inference through YOLO
9. What would you do differently? → Single attack path, real gradients, COCO eval
10. Ethical concerns? → Research only on owned systems; red-team consent

### Rare Interview Questions
1. BPDA for YOLO with NMS? → Approximate NMS gradient
2. Certifiable robustness for detection? → Mostly open research
3. Adversarial patch on traffic signs? → Different threat model — expectation over transforms
4. FRCNN train-mode loss components? → RPN + ROI classifier + box reg
5. Why sign grad optimal for L∞? → Linearized max perturbation per pixel
6. Transferability YOLO→FRCNN? → Not tested in project
7. Role of batch norm in adversarial robustness? → BN stats shift under attack
8. Digital vs physical ε? → Physical needs printability + viewing angle aug

### CTO-Level Questions
1. Business risk if we ignore this? → Safety/security liability
2. Cost to harden production CV? → Adversarial aug retrain + ongoing red-team
3. Build vs buy? → Depends on stack integration
4. What's the lie in the demo? → Post-processed detections — fix before stakeholder demo
5. Go/no-go for deployment? → Need domain ASR thresholds + monitoring

### Hidden Follow-Ups (Quick Reference)
- FGSM → I-FGSM → PGD → AutoAttack
- YOLO → anchors → NMS → differentiable NMS
- FRCNN → RPN → RoI Align → score threshold
- Region attack → missed det → universal patch
- Gradio → FastAPI → Triton → K8s GPU pool

---

# APPENDIX A — Key Code Snippets to Know Cold

### Detection tensor format (both models)
```python
# [N, 6]: x1, y1, x2, y2, confidence, class_id
detections = torch.cat([boxes, scores.unsqueeze(1), classes.unsqueeze(1)], dim=1)
```

### Attack loop (attack_utils.py — what actually runs in web)
```python
for iteration in range(10):
    adv_image.requires_grad_(True)
    loss = sum over bbox regions: (region - 0.5).pow(2).mean() * 100
    loss += (adv_image - image_tensor).pow(2).mean() * 5
    loss.backward()
    adv_image += (epsilon/10) * adv_image.grad.sign()
    adv_image = clamp(adv_image, 0, 1)
```

### Visual post-process (app.py — DISCLOSE IN INTERVIEWS)
```python
if i % 4 == 0 and int(adversarial_dets_visual[i, 5]) != 9:
    adversarial_dets_visual[i, 5] = 3.0  # force class to car
```

### FRCNN white-box attack (attacks/fgsm.py — intended approach)
```python
self.model.model.train()
losses = self.model.model([adv_image.squeeze(0)], targets)
loss = sum(v for v in losses.values())
loss.backward()
```

---

# APPENDIX B — Glossary

| Term | Meaning |
|------|---------|
| FGSM | Fast Gradient Sign Method |
| I-FGSM | Iterative FGSM |
| PGD | Projected Gradient Descent |
| ε (epsilon) | Max perturbation budget (L∞) |
| NMS | Non-Maximum Suppression |
| RPN | Region Proposal Network |
| FPN | Feature Pyramid Network |
| RoI Align | Differentiable region pooling |
| COCO | Common Objects in Context dataset |
| ASR | Attack Success Rate |
| mAP | mean Average Precision |
| IoU | Intersection over Union |
| White-box | Attacker has model weights + gradients |
| Black-box | Query-only attacker |

---

# APPENDIX C — Interview Day Checklist

- [ ] Explain project in 60 seconds without overselling
- [ ] Acknowledge visualization post-processing proactively
- [ ] Draw data flow diagram from memory
- [ ] Compare YOLO vs FRCNN failure modes
- [ ] State exact preprocessing dimensions
- [ ] Explain why true FGSM failed on YOLO
- [ ] Describe proper eval you'd add (mAP, IoU-match ASR)
- [ ] Production roadmap in 3 bullets
- [ ] One "what I'd do differently" answer ready
- [ ] Ethical use disclaimer

---

*Document generated from full source read of all 18 project files. Know the code truth, not just the README.*
