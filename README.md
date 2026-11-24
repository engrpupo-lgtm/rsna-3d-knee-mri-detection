# Volumetric 3D Knee MRI Abnormality Detection

End-to-end deep learning pipeline for multi-label abnormality classification on 3D musculoskeletal knee MRI volumes (RSNA 2026 Challenge).

## Architecture (`v1.0` Baseline)
- **Backbone:** Dual-branch 2.5D multi-view CNN built on `EfficientNet-B0` (`timm`, PyTorch) fusing multi-planar slice representations.
- **Data Engineering:** Custom DICOM volume preprocessing and cached tensor pipelines (`torch.save` / `.pt` caching) to eliminate disk I/O bottlenecks during training.
- **Supervision:** Hybrid training combining official competition labels with structured pathology targets extracted from radiology reports.

## Benchmarks

| Version | Architecture / Notes | Public Macro-AUC | Status |
| :--- | :--- | :---: | :--- |
| `v1.0-baseline` (`rsna-v8-dual-branch`) | Dual-branch 2.5D EfficientNet-B0 + tensor caching | **0.791** | Verified |
| `v2.0` | Anatomy-guided spatial supervision + `Mix5` label distillation | — | In Progress |

## Repository Structure
- `notebooks/` — Verified competition training and inference notebooks (`rsna-v8-dual-branch.ipynb`).
- `src/` — Modular PyTorch datasets, multi-view model definitions, and training loops (refactoring in progress).
