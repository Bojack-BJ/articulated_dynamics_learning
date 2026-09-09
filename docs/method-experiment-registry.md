# Method and Experiment Registry / 方法与实验注册表

## Status Vocabulary / 状态定义

- **Main candidate / 主方法候选**: eligible for final model selection under a fixed validation contract.
- **Ablation / 消融**: controlled comparison that changes one declared component.
- **Diagnostic / 诊断**: explains a failure mode but is not a main-table result.
- **Baseline / 基线**: analytic or external comparison with explicit input/oracle assumptions.
- **Rejected / 暂不采用**: tested and currently inferior or invalid under the intended contract.
- **Pending / 待验证**: implemented or planned but not yet evaluated on the aligned held-out set.

## 1. Tracking and Depth / 跟踪与深度

| Method / 方法 | Input and change / 输入与改动 | Evidence / 现有证据 | Status / 状态 |
|---|---|---|---|
| CoTracker-only / 仅 CoTracker | 2D tracks, measured-depth lifting, frozen track tokens | Best fixed-ontology validation assignment loss `0.108469`; controlled test IoU `0.678`, ARI `0.750`, Joint@20 `0.685` | Main candidate / 主方法候选 |
| TAPIP3D-only / 仅 TAPIP3D | Native world-space 3D tracks and TAPIP tokens | Validation assignment loss `0.290504`; controlled test IoU `0.625`, Joint@20 `0.570` | Ablation / 消融 |
| Hybrid tracking / 混合跟踪 | TAPIP3D trajectories plus CoTracker embeddings | Legacy main; validation loss `0.131409`, test IoU `0.669`, Joint@20 `0.659` | Ablation, historical main / 消融、历史主方法 |
| Dynamic reseeding / 动态补点 | Add track queries in newly visible mask regions | Implemented for disocclusion and nested parts | Component / 组件 |
| Windowed and bidirectional tracking / 分窗双向跟踪 | Overlapping temporal windows, forward/backward propagation and stitching | Added to address short temporal coverage; real-data stability remains scene-dependent | Diagnostic component / 诊断组件 |
| Track/timestep quality weighting / 轨迹与时刻质量加权 | Consume `track_quality_score` and `timestep_quality_score` | Implemented in fitting, viewer filtering and training inputs | Component / 组件 |
| Hard depth-spike and ID-switch rejection / 深度尖峰与 ID switch 硬剔除 | Reject discontinuous depth and correspondence jumps | Implemented and tested in robust tracking/fitting tests | Component / 组件 |
| IRLS/Huber segment consensus / IRLS/Huber 分段共识 | Independently fit continuous valid segments, robustly fuse | Implemented; improves outlier resistance but does not repair missing observations | Component / 组件 |
| Persistent RGB-D map / 持久点云地图 | Retain geometry after it leaves the current view | Viewer support implemented; visualization aid, not a source of new correspondences | Viewer diagnostic / 可视化诊断 |
| DA3 depth replacement/fusion / Depth Anything 3 深度替换或融合 | Monocular predicted depth, optional sensor-depth fusion | HOI4D pilot exposed camera/extrinsic drift and scale issues; not a general fix | Rejected as universal fix / 不作为通用方案 |

## 2. Slot Segmentation / Slot 分割

| Method / 方法 | Definition / 定义 | Evidence / 现有证据 | Status / 状态 |
|---|---|---|---|
| Learned slot head / 学习式 slot head | Transformer set encoder/decoder, Hungarian assignment and existence prediction | Core segmentation path | Main component / 主组件 |
| Slot-only fine-tuning / 仅 slot 微调 | Optimize assignment, Dice, pair, existence and rigidity losses; freeze relation head | Used to isolate segmentation adaptation | Ablation/training stage / 消融或训练阶段 |
| Relation-only fine-tuning / 仅 relation 微调 | Freeze slot encoder/decoder; optimize edge/type/axis geometry | Used after slot stabilization | Ablation/training stage / 消融或训练阶段 |
| Full joint fine-tuning / 全量联合微调 | Optimize slot and relation losses together | Real-only and mixed pilots tested; can destabilize slot geometry without careful balancing | Main candidate with safeguards / 需约束的候选 |
| Oracle slots / GT slot assignment | Replace predicted assignment with aligned GT labels while retaining declared geometry path | Diagnostic only; a viewer labeled `oracle` is invalid if it still colors predicted slots | Oracle diagnostic / Oracle 诊断 |
| Local pair/purity loss / 局部同部件与纯度损失 | Encourage nearby, motion-compatible tracks to share a slot and hard boundary negatives to separate | Implemented in slot objective | Component / 组件 |
| Differentiable rigid replay / 可微刚体回放 | Soft slot weights supervise common rigid transforms | Implemented as a training regularizer; not required at inference | Component / 组件 |
| View dropout / 视角丢弃 | Randomly retain one or two of three PartNet views, excluding the least-excited-only case | Best validation checkpoint occurred near epoch 30 in the long slot pretraining pilot | Promising augmentation / 有效增强 |
| Sensor/slot corruption curriculum / 传感器与 slot 污染课程 | Stage depth noise, tracking corruption and imperfect slot assignments before relation training | Several pilots run; gains were limited under current synthetic corruption distribution | Diagnostic, not final / 诊断、非最终方案 |
| Real-only scratch overfit / 真实数据从零过拟合 | Train on a small real subset to test model capacity and label consistency | Tested on Arti4D; sequence fitting possible but cross-object generalization remains weak | Diagnostic / 诊断 |

## 3. Relation, Joint Type and Axis / 关系、关节类型与轴

| Method / 方法 | Evaluation input / 评估输入 | Output / 输出 | Evidence and status / 证据与状态 |
|---|---|---|---|
| Original direct feedforward / 最初直接回归 | Slot token plus pooled canonical descriptors | edge, type, axis, line point | Aligned 32-joint child-oracle axis mean `62.48 deg`; rejected for axis geometry / axis 已淘汰 |
| Full neural relation head / 完整 neural relation head | Predicted slots and learned pair features | directed graph, type, axis, line | Controlled CoTracker result: type `0.885`, axis mean `12.27 deg`, Joint@20 `0.685`; current paper candidate / 主方法候选 |
| Neural type + analytic axis / 神经类型 + 解析轴 | Predicted topology/type, relative-SE(3) fitting | axis and line | Better line localization, lower end-to-end success; baseline/fallback / 基线或回退 |
| Full analytic / 全解析 | Topology proposals plus analytic type and geometry | type, axis and line | Type selection weak (`0.596` in controlled Hybrid study); baseline / 基线 |
| VN + Pivot-B | Equivariant vector-neuron relation geometry and revolute pivot refinement | type, axis and line | Aligned 32-joint oracle geometry mean `16.13 deg`, line `0.0418`; did not beat child analytic / 消融 |
| Child-only analytic / 仅 child 解析拟合 | GT or predicted child tracks, visibility and quality; no parent centroid | prismatic/revolute axis and revolute line | Aligned 32 joints: coverage `30/32`, axis mean `10.70 deg`, line `0.0185`; strong baseline / 强基线 |
| Child-only equivariant feedforward pilot / 仅 child 等变前馈 pilot | Child track vectors and invariant scalar features | type, candidate weights, axis and line | Type `93.75%`, axis mean `41.04 deg`; simple construction fails especially revolute / 暂不采用 |
| Multi-scale group Kabsch proposals / 多尺度小组 Kabsch proposals | Same-track correspondences over several frame strides, quality/membership weights | local `R,t`, rotation-axis and line proposals | Aligned GT-child 32-joint pilot: coverage `81.2%`, axis mean `28.00 deg`, line `0.0152`; strong high-support revolute proposals but weak low-support/prismatic cases / 诊断候选 |
| Track-motion line/circle voting / 单轨迹直线/圆投票 | Independently segmented child trajectories; slot only supplies membership | prismatic PCA votes, revolute circle-normal/center votes, confidence | Aligned GT-child pilot: coverage `87.5%`, axis mean `24.04 deg`; prismatic coverage `100%`, mean `14.47 deg`, but revolute mean `33.61 deg` / 诊断候选 |
| Type-routed proposal fusion / 按类型路由的候选融合 | Track voting for prismatic; group Kabsch for revolute | axis and revolute line | GT-child: coverage `90.6%`, mean `18.83 deg`, penalized mean `25.50 deg`, line `0.0152`; predicted child slot: coverage `87.5%`, mean `17.28 deg`, penalized mean `26.37 deg` / 尚未超过 child-only analytic |

The last two routes deliberately separate assignment from motion inference. Group Kabsch uses local rigid consistency as an auxiliary proposal generator. Track voting treats every continuous trajectory segment as an independent motion hypothesis and uses rigidity only as optional supervision or proposal-consistency evidence.

最后两条路线都将 assignment 与 motion inference 解耦：小组 Kabsch 只产生局部刚体候选；track voting 则让每个连续轨迹片段独立产生直线或圆运动假设，刚体一致性仅作为可选监督或候选一致性证据，不再是推理的硬前提。

## 4. Rotation and Equivariance / 旋转与等变性

| Experiment / 实验 | Meaning / 含义 | Current conclusion / 当前结论 | Status |
|---|---|---|---|
| Coordinate-frame SO(3) rotation / 坐标系 SO(3) 旋转 | Apply the same rigid transform to geometry, tracks and GT axes | Tests representation equivariance only; equivalent to transforming object and camera together, not fixed-camera novel appearance | Required diagnostic / 必要诊断 |
| Fixed-camera object rotation `0/45/90` / 固定相机物体旋转 | Rotate the object while camera observations change | Tests visibility, appearance and self-occlusion shift; angles are object yaw, not coordinate labels | External/generalization experiment / 泛化实验 |
| Yaw/random-axis/Haar augmentation | Rotate explicit geometry during training | Mild random-axis `<=15 deg` was least harmful; Haar collapsed to broad `60--70 deg` errors | Diagnostic; aggressive version rejected / 诊断 |
| Exact equivariance unit audit / 严格等变单测 | Compare transformed predictions under known `Q,b` | Implemented for relation geometry modules | Required test / 必测 |

## 5. Training Data and Adaptation / 训练数据与适配

| Dataset or regime / 数据或训练方式 | Use / 用途 | Known limitation / 已知限制 | Status |
|---|---|---|---|
| PartNet-Mobility three-view | Large synthetic pretraining and controlled held-out evaluation | Fused views and clean depth differ from occluded real RGB-D | Primary pretraining / 主预训练 |
| PartNet view-dropout | Simulate partial observation while retaining sufficient motion excitation | Does not reproduce all real sensor artifacts | Recommended pretraining augmentation / 推荐增强 |
| Arti4D manually/pseudo annotated | Real interaction fine-tuning and held-out-object validation | Small object count, mask/track variability, pseudo-axis noise | Usable curated real set / 可用真实集 |
| HOI4D selected subset | Real egocentric articulation, CAD/pose annotations and RGB-D | Depth/camera drift, incomplete masks, unstable tracks and questionable transformed axes | Audit first; only curated sequences trainable / 需筛选 |
| Self-recorded real scenes | Final real validation and possible object-held-out adaptation | Physical-instance leakage must be prevented | Primary real validation / 真实验证 |
| Balanced PartNet + real | Mix clean broad synthetic data with oversampled curated real sequences | Sampling ratio and label quality strongly affect slot stability | Pending final selection / 待拍板 |

## 6. External Baselines / 外部方法

| Method | Protocol / 输入协议 | Outputs evaluated / 已评估输出 | Current use / 当前用途 |
|---|---|---|---|
| AiM | Method-specific multi-view reconstruction/trajectory protocol | segmentation, reconstruction diagnostics, native axis where available | External baseline; reproduction sensitivity documented |
| ReArt | Native 4D point cloud | segmentation and native kinematics | External baseline |
| DTA | Two-state multi-view RGB-D, GT part count | segmentation and selected axis hypotheses | External baseline |
| ArtGS | Two-state multi-view RGB-D, GT part count | segmentation, axis, object-rotation runs | External baseline |
| VideoArtGS | Continuous monocular video, oracle joint metadata | segmentation and available kinematics | External baseline with explicit oracle label |
| GaussianArt | Two-state multi-view, semantic/motion initialization | segmentation and available kinematics | External baseline with explicit oracle label |
| PARIS | Two-state RGB, mainly two-part objects | segmentation and native articulation when successful | Scope-limited external baseline |

External results are object-aligned and metric-aligned but protocol-specific. Canonical-axis and rotation tables must show missing values when a method does not expose a valid axis line or when the run failed.

## 7. Immediate Aligned Tests / 下一步统一测试

1. The aligned 32-joint GT-child and predicted-child-slot proposal tests are complete. Neither raw proposal method nor the type-routed hybrid replaces child-only analytic (`30/32` coverage, `10.70 deg` mean, `0.0185` line).
2. If proposal learning continues, train only invariant confidence/calibration over existing world-vector candidates. Preserve the type routing and use child-only analytic as an explicit missing-proposal fallback.
3. Evaluate any learned calibration on a separate held-out split; do not select thresholds and report the same 32-joint pilot as a final test result.
4. Run coordinate SO(3) equivariance first; only then run fixed-camera object rotation for the surviving learned method.
5. On real data, report per-dataset and per-sequence results before any aggregate, with mask, depth, camera-pose and track-quality gates recorded.
