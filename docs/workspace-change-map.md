# Workspace Change Map / 工作区改动地图

## Purpose / 目的

This file is the entry point for the accumulated workspace changes. It separates versioned implementation work from local datasets, generated results, caches, and external repositories. It does not imply that every experiment is a paper result; method status is tracked in `method-experiment-registry.md`.

本文档是整个工作区累计改动的统一入口，用于区分应版本化的实现、实验配置与文档，以及不应进入 Git 的本地数据、生成结果、缓存和外部仓库。各方法是否有效、是否进入论文，见 `method-experiment-registry.md`。

## Versioned Functional Areas / 应版本化功能域

| Area / 功能域 | Main paths / 主要路径 | Scope / 范围 |
|---|---|---|
| Capture and simulation / 采集与仿真 | `src/rgbd_urdf_mvp/sim`, recorder CLI, recording configs | Multi-view RGB-D recording, articulation excitation, view dropout, GAPartNet/PartNet adapters |
| Masks and tracking / Mask 与跟踪 | `src/rgbd_urdf_mvp/perception`, tracking scripts | Object/part masks, dynamic reseeding, CoTracker/TAPIP3D/hybrid tracks, depth lifting, quality scores |
| Slot segmentation / Slot 分割 | `motion_part_slots.py`, relation training scripts | Learned slots, Hungarian matching, purity/local pair losses, rigid replay, slot-only and staged training |
| Relations and axes / 关系与轴 | `src/rgbd_urdf_mvp/kinematics` | Neural relation heads, analytic fitting, VN/Pivot-B, SO(3) augmentation, child-only and voting estimators |
| Real-data adapters / 真实数据适配 | Arti4D, HOI4D, ArtiPoint and RBO scripts/configs | Conversion, annotation, pseudo labels, audits, train/validation manifests |
| External baselines / 外部方法 | `src/rgbd_urdf_mvp/benchmarks`, baseline scripts/configs | AiM, ReArt, DTA, ArtGS, VideoArtGS, GaussianArt, PARIS and aligned evaluation |
| Evaluation and viewers / 评估与可视化 | evaluation scripts, `baseline_viewer.py`, `object_mask_flow_html.py` | Common-domain metrics, GT/predicted axes, annotation UI, review hubs, paper galleries |
| Paper and reporting / 论文与汇报 | `docs`, reporting scripts, selected `paper_assets` | Method specification, result reconciliation, tables, runtime and qualitative figures |

## Local-Only Areas / 仅本地保留

| Path | Policy / 处理策略 | Reason / 原因 |
|---|---|---|
| `data/` | Ignore, do not delete / 忽略但不删除 | Local datasets and annotations; currently multi-GB |
| `outputs/` | Ignore, do not delete / 忽略但不删除 | Generated checkpoints, metrics and viewers |
| `backups/` | Ignore, do not delete automatically / 忽略且不自动删除 | Large machine backup; ownership and retention must be decided separately |
| `.tmp/`, `tmp/` | Ignore; caches may be deleted / 忽略；可删除缓存 | Reproducible intermediate artifacts |
| `paper_assets/` | Review selectively / 逐项审查 | Contains both source assets and large generated binaries |
| `Hunyuan3D-2`, `Particulate`, `co-tracker` | Keep as submodules / 保持子模块 | External repositories; local dirtiness must not be folded into this repository |
| `*.egg-info`, `__pycache__`, `.DS_Store` | Generated / 生成物 | Packaging or OS/runtime metadata |

## Consistency Issues Found / 已发现一致性问题

1. `current-methods-pipeline.md` contains historical Hybrid-main wording while the latest fixed-ontology validation selects CoTracker-only. Any paper text must state the checkpoint, ontology, and evaluation contract explicitly.
2. Oracle-slot, predicted-slot, relation-only, and slot-only outputs have been mixed in some viewer names. New reports must record both the assignment source and the geometry source.
3. Coordinate-frame rotation and fixed-camera object rotation are different tests and must remain separate.
4. Real-data results from Arti4D, HOI4D, and self-recorded scenes have different depth, mask, camera-pose, and annotation quality. They must not be averaged without a declared common contract.
5. External baselines are object- and metric-aligned but often protocol-specific and use different oracle inputs. Unsupported outputs must remain missing rather than inferred.

## Commit Plan / 分阶段提交计划

1. `tracking-viewer`: capture, masks, tracking, quality, persistent RGB-D map, annotation/viewer UI, and their tests.
2. `slot-relation`: slot objectives, relation heads, SO(3), analytic/VN/child-only/group/voting axis methods, training/evaluation scripts, and tests.
3. `datasets-real`: PartNet/GAPartNet, Arti4D, HOI4D, ArtiPoint/RBO/self-recorded adapters, manifests, annotation tools, and audits.
4. `external-baselines`: AiM, ReArt, DTA, ArtGS, VideoArtGS, GaussianArt, PARIS adapters, aligned metrics, viewers, and tests.
5. `docs-reporting`: method/result registry, paper tables, runtime reports, figures, and selected small reproducible assets.

Each phase should be tested and pushed independently. Local datasets, checkpoints, generated viewers, large PDFs/PPTX files, and submodule working-tree changes are excluded unless explicitly selected.
