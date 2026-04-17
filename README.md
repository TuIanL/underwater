# Swim Pose

`Swim Pose` 是一个面向游泳视频分析的 Python 项目。当前主线聚焦于“以单条 stitched 合成视频作为输入的蛙泳 2D 关键点定位”，并已经补齐了数据清单、抽帧、人工标注、训练、推理、评估、可视化这一整条本地实验链路。自 2026-04-17 起，默认监督基线已经切换为“YOLOv8-Pose 2D 强基线 + 水下域适应 + 时序后处理”；原有自研 heatmap、Phase 1 SupCon 和 bridge 路线保留为研究型 baseline。

## 截至 2026-04-17 的开发状态

- 主线流程已打通：视频清单 -> 审计 -> 抽帧 -> 种子帧选择 -> 标注 -> 标注审计 -> 训练 -> 推理 -> 评估 -> 浏览器可视化
- 已实现本地桌面标注 GUI 和浏览器标注 UI，支持 18 点关键点标注、可见性设置、状态切换和审计
- 已实现默认 YOLO 监督训练、legacy heatmap 监督训练、半监督训练、带时序约束的半监督训练、伪标签生成
- 已实现单模型预测查看器，支持骨架叠加、逐帧浏览、序列播放、置信度过滤、指标联动展示
- 已实现 YOLO pose 数据集导出、18 点 schema 适配、raw/filtered 双轨导出，以及时序稳定性评估
- 已实现 Phase 1 视频级 SupCon 预训练，以及通过 `bridge` 配置把视频教师特征蒸馏到 legacy 2D heatmap 学生模型
- 当前仍处于早期实验验证阶段，结论主要用于验证流程和方向，不代表已经获得稳定可泛化的模型能力

## 当前仓库快照

### 1. 定位主线数据

- `data/manifests/clips.csv`：当前有 1 个定位用 stitched 蛙泳 clip
- `data/manifests/unlabeled_index.csv`：当前抽出了 1276 张 stitched 帧
- `data/manifests/seed_frames.csv`：当前选出了 12 张种子帧
- `data/annotations/seed/`：当前共有 12 个标注 JSON
- `reports/annotation-audit.json`：审计结果为 6 个 `labeled`、6 个 `no_swimmer`、0 个 warning、0 个 validation error
- `data/manifests/annotation_index.csv`：当前只有 6 条可用于监督训练的已标注样本
- `data/manifests/splits/train.csv`：当前有 6 条训练样本
- `data/manifests/splits/val.csv`：当前为空
- `data/manifests/splits/test.csv`：当前为空

### 2. SupCon 视频预训练数据

- `data/manifests/supcon_videos.csv`：当前有 12 条视频索引
- 其中 11 条为 `valid`，1 条为 `mixed_stroke`
- 当前有效泳姿分布为：蛙泳 3 条、仰泳 3 条、自由泳 3 条、蝶泳 2 条

### 3. 基线实验产物

- `artifacts/baselines/localization_baseline/` 下已经保存了一份基线包
- 当前基线包中选中的模型是 `supervised`
- `comparison.json` 显示目前监督基线优于两个半监督变体
- `metrics.json` 中当前监督基线的初步结果为：
  - `mean_normalized_error = 0.4096`
  - `pck@0.05 = 0.0808`
  - `pck@0.10 = 0.1919`
- 上述指标只来自单个 athlete/session 的 6 张已标注帧，不能代表 held-out athlete 或 held-out session 的泛化能力

## 已实现功能

### 数据与路径管理

- 支持从视频目录自动生成定位 manifest：`manifest init`
- 支持补充视频元数据并审计 manifest：`manifest audit`
- 支持把旧的相对路径 manifest 迁移为仓库相对路径或绝对路径：`manifest migrate-paths`
- 支持构建 Phase 1 SupCon 视频索引，并过滤 `mixed_stroke`、异常命名和异常目录结构
- 已统一“仓库内管理路径按 repo root 解析、外部源路径按输入来源解析”的规则，减少不同工作目录下的路径混乱

### 标注与数据集构建

- 已固定 18 点关键点定义：`nose`、`neck`、肩、肘、腕、髋、膝、踝、脚跟、脚尖
- 已实现三档可见性语义：
  - `2`：直接可见
  - `1`：不可见但可推断
  - `0`：不可见且不可可靠推断
- 已实现标注模板生成、JSON 校验、种子帧标注脚手架生成
- 已实现桌面 Tk 标注 GUI：`annotations gui`
- 已实现浏览器标注界面：`annotations web`
- 已实现标注审计：`annotations audit`
- 已实现标注索引构建，只把 `frame_status = labeled` 的样本写入训练索引
- 已支持 `pending`、`review`、`labeled`、`no_swimmer` 四种帧状态

### 训练能力

- 已实现默认 YOLO 2D 关键点监督训练：`train supervised --config configs/supervised.toml`
- 已实现 project-owned YOLO pose 数据集导出：`dataset export-yolo-pose`
- 已保留 legacy heatmap 监督训练：`train supervised --config configs/supervised_legacy.toml`
- 已实现半监督训练（legacy heatmap research baseline）：`train semisupervised`
- 已实现带时序平滑约束的半监督训练配置：`configs/semi_supervised_temporal.toml`
- 已实现 Phase 1 视频级 SupCon 预训练：`train supcon`
- 当前默认 2D 主干配置为 `yolov8s-pose.pt`
- 当前默认视频预训练主干配置为 `r2plus1d_18`
- 已实现 bridge 蒸馏训练（legacy research baseline）：
  - 冻结视频教师编码器
  - 从时间邻域帧构建 clip
  - 将 2D 学生的 pooled feature 通过 projector 对齐到视频教师特征

### 推理、评估与可视化

- 已实现预测导出：`predict`
- 预测结果会输出每个关键点的：
  - `x / y`
  - `confidence`
  - `visibility`
- 当启用时序后处理时，同一条预测 JSONL 还会额外输出 `filtered_points`
- 已实现评估：`evaluate`
- 当前评估会输出：
  - `mean_normalized_error`
  - `pck@0.05`
  - `pck@0.10`
  - `visible_mean_error`
  - `occluded_mean_error`
  - raw `temporal_jitter`（基于 midpoint residual）
  - `temporal_stability.raw / temporal_stability.filtered`
  - `per_joint` 分项指标
- 已实现预测浏览器：`predictions web`
- 预测浏览器当前支持：
  - 单个预测 JSONL 文件浏览
  - 关键点骨架叠加
  - 按 clip 过滤
  - 序列播放
  - 置信度阈值控制
  - 每个关键点的坐标、置信度、可见性明细
  - 可选展示整体指标和逐关节指标
- 已实现伪标签生成：`pseudolabel generate`
- 伪标签生成支持可选消费 `filtered_points`：`pseudolabel generate --use-filtered`

## 当前实验结论

- 当前定位主线已经不是“只有标注工具”，而是已经具备完整的实验闭环
- 当前监督基线优于两个半监督变体，说明在极小标注集条件下，伪标签与时序约束还没有带来稳定收益
- 当前最难的关节主要集中在下肢远端，尤其是 `ankle / heel / toe`
- `reports/failure-modes.md` 记录了当前最主要的问题来源：
  - stitched 水线拼接导致头肩腕等位置不稳定
  - 飞溅和遮挡使脚踝、脚跟、脚尖置信度下降
  - 左右肢体在侧视重叠时容易发生身份混淆
  - stitched 视频上的时序抖动仍然较高

## 仓库结构

- `src/swim_pose/`：核心实现与 CLI 入口
- `configs/`：默认 YOLO baseline、legacy heatmap research baseline、半监督、SupCon、bridge 等实验配置
- `docs/`：标注规范和研究说明
- `data/manifests/`：clip 清单、抽帧索引、标注索引、数据划分、SupCon 视频索引
- `data/templates/`：标注模板
- `reports/`：标注审计和失败模式记录
- `artifacts/`：基线包、checkpoint、预测结果、报告等实验产物
- `tests/`：路径、manifest 迁移、SupCon 流程、bridge 流程等 smoke test
- `openspec/`：需求规格与已归档变更记录

## 环境与安装

项目要求：

- Python `>= 3.11`
- 推荐使用 `uv`

安装依赖：

```bash
uv sync --locked
```

如果本机的 `uv` 缓存目录有权限问题，可以临时这样运行：

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync --locked
```

说明：

- 文档中的仓库内路径默认都以仓库根目录为基准
- 大部分命令在仓库根目录或仓库子目录下都可以运行
- 当前依赖中已包含 `ultralytics==8.4.26`
- 当前 `pyproject.toml` 和 `uv.lock` 里还没有声明 `pytest`，所以仓库虽然已经有 `tests/`，但还没有切到 `uv run pytest` 这一套标准回归入口

## 数据约定

定位主线当前以 stitched 视频作为唯一必需输入。推荐视频目录如下：

```text
data/raw/videos/
  athlete01/
    session01/
      trial01_stitched.mp4
```

如果还保留原始机位视频，可以作为补充元数据保留：

```text
data/raw/videos/
  athlete01/
    session01/
      trial01_above.mp4
      trial01_under.mp4
      trial01_stitched.mp4
```

支持的命名后缀：

- `_above`
- `_under`
- `_stitched`

如果文件名没有视角后缀，工具默认把它当作 `stitched`

## 推荐工作流

### 1. 生成并审计定位 manifest

```bash
uv run -- swim-pose manifest init \
  --video-root data/raw/videos \
  --output data/manifests/clips.csv

uv run -- swim-pose manifest audit \
  --manifest data/manifests/clips.csv \
  --output data/manifests/clips.audit.csv
```

### 2. 抽帧并建立未标注索引

```bash
uv run -- swim-pose frames extract \
  --manifest data/manifests/clips.audit.csv \
  --output-root data/frames \
  --index-output data/manifests/unlabeled_index.csv \
  --views stitched \
  --every-nth 5
```

### 3. 选择种子帧并生成标注文件

```bash
uv run -- swim-pose seed select \
  --manifest data/manifests/clips.audit.csv \
  --output data/manifests/seed_frames.csv \
  --source-view stitched

uv run -- swim-pose annotations scaffold \
  --seed-csv data/manifests/seed_frames.csv \
  --frame-root data/frames \
  --output-root data/annotations/seed
```

### 4. 进行人工标注

桌面 GUI：

```bash
uv run -- swim-pose annotations gui \
  --annotation-root data/annotations/seed \
  --frame-root data/frames
```

浏览器标注界面：

```bash
uv run -- swim-pose annotations web \
  --annotation-root data/annotations/seed \
  --frame-root data/frames
```

如果 macOS 上 Tk GUI 不稳定，也可以直接运行：

```bash
./scripts/launch_annotation_gui.sh
```

### 5. 审计标注并生成训练索引

```bash
uv run -- swim-pose annotations audit \
  --annotation-root data/annotations/seed \
  --output reports/annotation-audit.json

uv run -- swim-pose annotations index \
  --annotation-root data/annotations/seed \
  --output data/manifests/annotation_index.csv
```

### 6. 划分数据并训练默认 YOLO 2D 基线

```bash
uv run -- swim-pose dataset split \
  --index data/manifests/annotation_index.csv \
  --output-dir data/manifests/splits

uv run -- swim-pose train supervised \
  --config configs/supervised.toml
```

如果你想复现实验期的 legacy heatmap baseline，可使用：

```bash
uv run -- swim-pose train supervised \
  --config configs/supervised_legacy.toml
```

### 7. 推理、评估、查看结果

```bash
uv run -- swim-pose predict \
  --config configs/supervised.toml \
  --checkpoint artifacts/checkpoints/supervised_yolo/best.pt \
  --index data/manifests/splits/train.csv \
  --output artifacts/predictions/supervised_yolo_train_predictions.jsonl

uv run -- swim-pose evaluate \
  --predictions artifacts/predictions/supervised_yolo_train_predictions.jsonl \
  --annotations data/manifests/annotation_index.csv \
  --output artifacts/reports/supervised_yolo_train_eval.json

uv run -- swim-pose predictions web \
  --predictions artifacts/predictions/supervised_yolo_train_predictions.jsonl \
  --frame-root data/frames \
  --report artifacts/reports/supervised_yolo_train_eval.json
```

如果配置里启用了 `postprocess`，预测 JSONL 会同时包含 raw `points` 与 `filtered_points`；也可以通过 `postprocess.filtered_output` 额外写出纯 filtered sidecar artifact。

## 进阶实验入口

### 半监督（legacy heatmap research baseline）

```bash
uv run -- swim-pose pseudolabel generate \
  --predictions artifacts/predictions/supervised_yolo_train_predictions.jsonl \
  --output artifacts/predictions/pseudolabels.jsonl

uv run -- swim-pose train semisupervised \
  --config configs/semi_supervised.toml
```

如果你希望伪标签直接消费滤波后的轨迹，可使用：

```bash
uv run -- swim-pose pseudolabel generate \
  --predictions artifacts/predictions/supervised_yolo_train_predictions.jsonl \
  --output artifacts/predictions/pseudolabels.filtered.jsonl \
  --use-filtered
```

带时序约束的半监督可使用：

```bash
uv run -- swim-pose train semisupervised \
  --config configs/semi_supervised_temporal.toml
```

### Phase 1 SupCon 视频预训练

```bash
uv run -- swim-pose dataset build-video-index \
  --video-root data/raw/videos \
  --output data/manifests/supcon_videos.csv

uv run -- swim-pose train supcon \
  --config configs/supcon.toml
```

### Bridge：视频特征迁移到 legacy 2D heatmap 定位

```bash
uv run -- swim-pose train supervised \
  --config configs/supervised_bridge.toml
```

这个配置会额外：

- 加载 `artifacts/checkpoints/supcon/best.pt` 作为视频教师
- 从标注帧邻域构造时间 clip
- 用 feature distillation 约束 2D 学生特征

## 现有测试覆盖

当前仓库中已经有以下 smoke test / 单元测试文件：

- `tests/test_pathing.py`
- `tests/test_manifest_migration.py`
- `tests/test_supcon_pipeline.py`
- `tests/test_localization_bridge.py`

它们主要覆盖：

- 仓库路径解析
- 旧 manifest 路径迁移
- SupCon 视频索引构建与训练 smoke test
- bridge 数据集与监督训练 smoke test

但目前项目依赖尚未正式纳入 `pytest`，所以这部分测试还没有接到可直接执行的标准开发流程里。

## 已知限制与下一步

- 当前定位实验只有 1 个 stitched 蛙泳 clip，监督样本只有 6 张，数据量远不足以支撑泛化结论
- 由于只有单个 athlete/session，当前 `val.csv` 和 `test.csv` 为空，严格意义上的 held-out 验证还没有建立起来
- 当前 stitched 水线拼接和下肢远端关键点仍然是主要误差来源
- 默认 YOLO baseline 的首轮 smoke comparison 已记录在 `artifacts/reports/yolo_underwater_smoke_comparison.json`，但它只用于验证接线，不足以作为正式 promotion 依据
- 半监督、SupCon 与 bridge 分支仍然保留，但当前定位主线的优先级已经转到“补标注 + 水下域适应 + 时序稳定化”
- 下一阶段最重要的工作不是继续堆复杂时空模型，而是补充更多 athlete/session 的可用标注数据，并建立真正的训练/验证/测试划分
