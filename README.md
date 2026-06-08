# 3DGS 场景三维热场重建项目

## 1. 项目简介

本项目服务于毕业论文“基于 3DGS 的场景三维热场重建 / 热场虚拟仿真”。核心思路是在 RGB-3DGS 几何基底上，结合热红外监督和多源先验（geometry / semantic / shadow / sun-facing 等），训练一个可在 3DGS 场景上赋予 thermal 属性的网络，并输出可用于论文的静态图、对比图和评测结果。

当前仓库已经不是单纯的上游 3DGS 工程，而是一个“RGB-3DGS + 热场前处理 + Phase4/5 训练 + ROI 评测 + 交互可视化”的复合项目。

## 2. 项目整体流程

### 阶段 A：RGB 场景准备

1. UAV 可见光图像放入 `data/uav_scene/input`
2. 用 `convert.py` 跑 COLMAP：生成 `distorted/database.db`、`distorted/sparse/0`、`images/`、`sparse/0`
3. 用 `colmap2transforms.py` 生成 `transforms.json`
4. 用 `train_rgb.py` 训练 RGB-3DGS，输出 `output/debug_run`

### 阶段 B：热场训练前置准备

1. `render_top_down.py` 生成 RGB 顶视图 `topdown_final.png`
2. `manual_align.py` 把热图对齐到顶视图，生成 `lst_gt.png`
3. `bake_priors_physics.py` 生成几何先验 `priors.pt`

### 阶段 C：Phase4 先验构建

1. `tools/render_dsm_topdown.py` 渲染 DSM
2. `tools/bake_hillshade.py` 生成 hillshade / shadow proxy
3. `tools/segment_semantic.py` 或 `tools/segment_semantic_supervised.py` 生成语义图
4. `tools/bake_cast_shadow.py` 生成 cast visibility / effective shadow
5. `tools/fuse_priors_v2.py` 把 2D 先验投影回 3D 点
6. `tools/merge_priors_v2_two_shadow.py` 合并双 shadow
7. `tools/augment_priors_v2_to_v3.py` 升级成 v3 priors

### 阶段 D：Phase5 训练与消融

1. `train_thermal_robust.py` 训练主模型与消融组
2. `tools/crop_roi_pipeline.py` 切出训练视域 ROI
3. `esttools/prep_eval_inputs.py` + `esttools/metrics_eval_image.py` 进行 ROI 指标评测
4. `esttools/dump_all_eval_json.py` / `aggregate_metrics.py` 汇总结果

### 阶段 E：结果导出与论文图

1. `tools/view_phase5_roi_splat.py` 交互查看 thermal/RGB 模型并截图
2. `tools/render_thermal_topdown_pred.py` 导出顶视热场图
3. `tools/render_thermal_still.py` 导出指定视角静态图
4. `render_video_robust.py` 导出环绕热视频

## 3. 环境与依赖

### 已知环境信息

- Python：3.10 `[需本地验证具体小版本]`
- Conda 环境：`D:\conda_envs\ntrgs`
- PyTorch：当前环境为 `torch 2.4.1+cu121` `[需本地验证是否与当时训练环境完全一致]`
- OpenCV：4.12.0 `[需本地验证]`
- NumPy：2.1.2 `[需本地验证]`

### 关键系统依赖

- COLMAP：用于 `convert.py`
- CUDA：用于 3DGS 训练/渲染
- ImageMagick：仅在使用 `convert.py --resize` 时需要 `[待核验]`

### 关键 Python 依赖

- `torch`
- `opencv-python`
- `numpy`
- `plyfile`
- `tqdm`
- `tensorboard`
- `open3d`
- `trimesh`
- `pandas`（loss / eval 汇总脚本）

### 子模块 / C++ 扩展

- `submodules/simple-knn`
- `submodules/diff-gaussian-rasterization`

### 已知注意事项

- `tools/segment_semantic.py` 支持 SegFormer，但当前环境若没有 `transformers` 会回退到 `cv_fallback`。
- `configs/phase4.json` 不是完整主流程的一键入口，详见第 7 节。

## 4. 目录结构说明

| 目录 | 作用 |
| --- | --- |
| 根目录 | RGB-3DGS、GT 对齐、主训练与主渲染脚本 |
| `tools/` | Phase4/5 先验生成、ROI 裁切、交互查看器、审计与批量导图 |
| `esttools/` | 评测、校准、loss 曲线与表格汇总 |
| `data/uav_scene/` | UAV RGB 数据、COLMAP 产物、`transforms.json` |
| `output/debug_run/` | RGB-3DGS 基底、topdown、GT、基础 priors |
| `output/runs/<run_tag>/` | Phase4/5 的 artifacts、priors、phase5、metrics、paper_figs |

## 5. 主要脚本说明

### 5.1 根目录脚本

| 脚本 | 类别 | 作用 | 输入 / 输出 | 示例命令 | 与论文主流程关系 |
| --- | --- | --- | --- | --- | --- |
| `bake_priors.py` | 预处理 | 旧版几何 priors 烘焙器 | `point_cloud.ply -> priors.pt` | `python bake_priors.py -m output\debug_run` | 历史脚本，不推荐主用 |
| `bake_priors_physics.py` | 预处理 | 当前主用几何 priors 烘焙器 | `model_path -> priors.pt` | `python bake_priors_physics.py -m output\debug_run --backup_old` | 直接相关 |
| `colmap2transforms.py` | 预处理 | `sparse/0 -> transforms.json` | `cameras.bin/images.bin -> transforms.json` | `python colmap2transforms.py -s data\uav_scene` | 直接相关 |
| `convert.py` | 预处理 | COLMAP 包装器 | `input -> distorted/, images/, sparse/0` | `python convert.py -s data\uav_scene` | 直接相关 |
| `full_eval.py` | 辅助工具 | 上游 benchmark 总评测 | 外部 benchmark 数据集 | 不建议用于本项目 | 无关主论文 |
| `manual_align.py` | 预处理 | 交互式热图对齐 | `topdown_final + lst_full -> lst_gt` | `python manual_align.py` | 直接相关 |
| `metrics_temp.py` | 评估 | 旧温度指标脚本 | render/gt 目录 | `[待核验]` | 历史辅助 |
| `render_rgb.py` | 可视化 | 渲染 train/test RGB 视角 | `model_path -> train/test renders` | `python render_rgb.py -m output\debug_run --iteration 7000` | 相关 |
| `render_thermal_video.py` | 可视化 | 旧 thermal 视频渲染 | `thermal_ckpt -> mp4` | `python render_thermal_video.py --thermal_ckpt ...` | 历史辅助 |
| `render_top_down.py` | 可视化 | 生成 `topdown_final.png` | `model_path -> topdown_final.png` | `python render_top_down.py -m output\debug_run --width 2048 --height 2048 --resolution 4 --multiplier 0.85 --shift_x 0 --shift_y -1.2 --angle -31` | 直接相关 |
| `render_video_robust.py` | 可视化 | 当前 robust thermal orbit 视频 | `model/priors/ckpt -> mp4` | `python render_video_robust.py --model_path ... --priors_path ... --thermal_ckpt ...` | 相关 |
| `summary.py` | 调试/一次性 | 硬编码旧结果汇总 | 硬编码目录 -> `summary.csv` | 不建议直接调用 | 无关主论文 |
| `train_rgb.py` | 训练 | RGB-3DGS 主训练 | `data/uav_scene -> output/debug_run` | `python train_rgb.py -s data\uav_scene -m output\debug_run --resolution 4 --iterations 7000` | 直接相关 |
| `train_thermal_robust.py` | 训练 | 当前主 thermal 训练脚本 | `model/gt/priors -> thermal_net_robust.pth` | `python train_thermal_robust.py ...` | 直接相关 |

### 5.2 tools 脚本

| 脚本 | 类别 | 作用 | 输入 / 输出 | 示例命令 | 与论文主流程关系 |
| --- | --- | --- | --- | --- | --- |
| `tools/audit_priors_v3.py` | 辅助工具 | 快速导出 v3 priors 统计 | `priors -> audit.json` | `python tools\audit_priors_v3.py --priors ... --out_json ...` | 辅助 |
| `tools/augment_priors_v2_to_v3.py` | 预处理 | v2 升 v3，补 one-hot + sun_facing | `priors_v2 -> priors_v3` | `python tools\augment_priors_v2_to_v3.py --priors_in ... --ply ... --cameras ... --out_priors ... --semantic_num_classes 3` | 直接相关 |
| `tools/bake_cast_shadow.py` | 预处理 | 计算 cast visibility / effective shadow | `dsm -> cast_visibility/shadow_effective` | `python tools\bake_cast_shadow.py ...` | 直接相关 |
| `tools/bake_hillshade.py` | 预处理 | 计算 hillshade / shadow proxy | `dsm -> hillshade/shadow_map` | `python tools\bake_hillshade.py ...` | 直接相关 |
| `tools/check_ckpt_output.py` | 调试 | 只用 priors 检查 ckpt 是否退化 | `ckpt + priors` | `python tools\check_ckpt_output.py --ckpt ... --priors ...` | 辅助 |
| `tools/check_ckpt_output_full.py` | 调试 | 用真实 xyz+priors 检查 ckpt | `model + priors + ckpt` | `python tools\check_ckpt_output_full.py ...` | 辅助 |
| `tools/check_manifest.py` | 辅助工具 | run 前置清单检查 | `debug_root/run_dir` | `python tools\check_manifest.py --run_dir output\runs\... --debug_root output\debug_run` | 直接相关 |
| `tools/check_priors.py` | 辅助工具 | 对比 priors / priors_old | `priors/priors_old` | `python tools\check_priors.py --priors ... --priors_old ...` | 辅助 |
| `tools/check_priors_v3.py` | 辅助工具 | v3 priors 结构检查 | `priors_v3` | `python tools\check_priors_v3.py --priors ...` | 辅助 |
| `tools/crop_roi_pipeline.py` | 预处理 | ROI precheck/crop/postcheck | `run_base -> run_crop` | `python tools\crop_roi_pipeline.py --mode precheck --run_base ... --run_crop ... ...` | 直接相关 |
| `tools/fuse_priors_v2.py` | 预处理 | 2D semantic/shadow 投影回 3D 点 | `priors_v1 + semantic + shadow -> priors_v2` | `python tools\fuse_priors_v2.py --ply ... --cameras ... --priors_v1 ... --shadow_npy ... --semantic_map ... --out_priors_v2 ...` | 直接相关 |
| `tools/merge_priors_v2_two_shadow.py` | 预处理 | 合并双 shadow v2 | `priors_a + priors_b -> merged` | `python tools\merge_priors_v2_two_shadow.py --priors_a ... --priors_b ... --out ...` | 直接相关 |
| `tools/render_dsm_topdown.py` | 可视化/预处理 | topdown DSM + repro RGB | `ply/cameras -> dsm/repro` | `python tools\render_dsm_topdown.py ...` | 直接相关 |
| `tools/render_phase5_gallery.py` | 可视化 | 批量导出各组别 still gallery | `run_dir/groups -> gallery` | `python tools\render_phase5_gallery.py --run_dir ...` | 相关 |
| `tools/render_thermal_still.py` | 可视化 | 单视角导 thermal still 图 | `model/priors/ckpt -> gray/jet/raw` | `python tools\render_thermal_still.py -m output\debug_run --priors_path ... --thermal_ckpt ... --out_dir ...` | 直接相关 |
| `tools/render_thermal_topdown_pred.py` | 可视化 | 导出顶视 thermal 预测图 | `model/priors/ckpt -> topdown png` | `python tools\render_thermal_topdown_pred.py -m output\debug_run --priors_path ... --thermal_ckpt ... --out_dir ...` | 直接相关 |
| `tools/render_topdown_from_net.py` | 可视化/辅助 | 用训练式相机渲染 topdown | `model/priors/ckpt -> topdown` | `python tools\render_topdown_from_net.py ...` | 辅助 |
| `tools/reproduce_phase5_train_frame.py` | 辅助工具 | 精确复现 `train_5000` 帧 | `run_base_dir/group -> reproduced png` | `python tools\reproduce_phase5_train_frame.py --run_base_dir ... --group full_zscore` | 相关 |
| `tools/run_phase.py` | 主流程入口 | 读取 `phase4.json` 并调度 step | `config + steps -> run dir` | `python tools\run_phase.py --config configs\phase4.json --steps check,dsm --run_tag ...` | 直接相关 |
| `tools/segment_semantic.py` | 预处理 | 语义图生成，支持 SegFormer fallback | `topdown_rgb -> semantic_map` | `python tools\segment_semantic.py --rgb output\debug_run\topdown_final.png --out_png ... --out_npy ...` | 直接相关 |
| `tools/segment_semantic_supervised.py` | 预处理 | 有监督语义图生成 | `rgb + label_mask -> semantic_map` | `python tools\segment_semantic_supervised.py --rgb ... --label_mask ... --out_png ... --out_npy ...` | 相关 |
| `tools/shadow_stats.py` | 辅助工具 | 统计 shadow 通道分布 | `priors -> stdout` | `python tools\shadow_stats.py --priors ...` | 辅助 |
| `tools/shuffle_priors.py` | 预处理 | 生成 shuffled ablation priors | `priors -> shuffled priors` | `python tools\shuffle_priors.py --in_path ... --out_path ...` | 直接相关 |
| `tools/view_phase5_roi_splat.py` | 可视化 | 交互式 thermal/RGB splat 查看器 | `run_base/run_crop -> screenshots` | `python tools\view_phase5_roi_splat.py --scene_variant full --run_base_dir ... --run_crop_dir ...` | 直接相关 |
| `tools/view_rgb_3dgs.py` | 可视化 | 交互式 RGB splat 查看器 | `scene_variant/run_crop_dir -> screenshots` | `python tools\view_rgb_3dgs.py --scene_variant full --run_crop_dir ...` | 相关 |
| `tools/vis_priors_topdown.py` | 可视化 | 将 priors 投影到 topdown 做审计 | `priors/ply/cameras -> pngs` | `python tools\vis_priors_topdown.py --priors_pt ... --ply ... --cameras ... --out_dir ...` | 相关 |
| `tools/vis_priors_v3_full.py` | 可视化 | v3 priors 全量审计图主脚本 | `priors_v3 -> full audit` | `python tools\vis_priors_v3_full.py --priors ... --ply ... --cameras ... --ref_hw_from ... --out_dir ...` | 直接相关 |
| `tools/vis_shadow_mean.py` | 可视化 | 聚合 shadow 通道到 topdown | `priors/ply/cameras -> pngs` | `python tools\vis_shadow_mean.py --priors ... --ply ... --cameras ... --out_dir ...` | 辅助 |

### 5.3 esttools 脚本

| 脚本 | 类别 | 作用 | 输入 / 输出 | 示例命令 | 与论文主流程关系 |
| --- | --- | --- | --- | --- | --- |
| `esttools/aggregate_metrics.py` | 评估 | 汇总多 run 指标均值/标准差 | `run_tags -> csv/json` | `python esttools\aggregate_metrics.py --run_tags ... --rel_json metrics/eval_image_roi.json --out_csv ... --out_json ...` | 相关 |
| `esttools/all_loss.py` | 调试/一次性 | 批量拼接各组 loss 表 | 硬编码 `ROOT_DIR` | 不建议直接调用 | 辅助 |
| `esttools/calibrate_points.py` | 评估 | 用实测点标定灰度预测到摄氏度 | `pred + points_csv -> fit_json/err_csv` | `python esttools\calibrate_points.py --pred ... --points_csv ... --out_fit_json ... --out_err_csv ...` | 相关 |
| `esttools/dump_all_eval_json.py` | 评估 | 汇总 eval suite 结果 | `eval_base -> merged json/csv` | `python esttools\dump_all_eval_json.py --run_sum 2026-02-25_eval_suite_roi` | 直接相关 |
| `esttools/metrics_eval_image.py` | 评估 | ROI 图像指标计算 | `gt/pred/mask -> json/csv` | `python esttools\metrics_eval_image.py --gt ... --pred ... --mask ... --out_json ... --out_csv ...` | 直接相关 |
| `esttools/phase5_metrics_precheck.py` | 辅助工具 | 评测前检查 gt/pred/mask | `run_tag/gt/pred -> precheck` | `python esttools\phase5_metrics_precheck.py --run_tag ... --gt ... --pred ...` | 辅助 |
| `esttools/plot_loss.py` | 调试/一次性 | 从 Excel 画 loss 曲线 | 硬编码 Excel -> png | 不建议直接调用 | 辅助 |
| `esttools/prep_eval_inputs.py` | 评估 | 统一 GT 到预测图尺寸 | `gt/pred -> gt_train.png` | `python esttools\prep_eval_inputs.py --gt output\debug_run\lst_gt.png --pred ... --out_dir ...` | 直接相关 |
| `esttools/render_alpha_mask_topdown.py` | 评估/可视化 | 渲染 topdown alpha mask | `model -> alpha mask` | `python esttools\render_alpha_mask_topdown.py --model_path output\debug_run --out_dir ...` | 相关 |

## 6. 推荐执行顺序

### 6.1 从零开始的主线

1. `convert.py`
2. `colmap2transforms.py`
3. `train_rgb.py`
4. `render_top_down.py`
5. `manual_align.py`
6. `bake_priors_physics.py`
7. `tools/run_phase.py --steps check,dsm,hillshade,semantic`
8. `tools/bake_cast_shadow.py`
9. `tools/fuse_priors_v2.py`
10. `tools/merge_priors_v2_two_shadow.py`
11. `tools/augment_priors_v2_to_v3.py`
12. `train_thermal_robust.py`
13. `tools/crop_roi_pipeline.py`
14. `esttools/prep_eval_inputs.py`
15. `esttools/metrics_eval_image.py`
16. `tools/view_phase5_roi_splat.py` / `tools/render_thermal_topdown_pred.py`

### 6.2 如果只服务论文主实验

优先关注这些脚本：

- `convert.py`
- `colmap2transforms.py`
- `train_rgb.py`
- `render_top_down.py`
- `manual_align.py`
- `bake_priors_physics.py`
- `tools/render_dsm_topdown.py`
- `tools/bake_hillshade.py`
- `tools/segment_semantic.py`
- `tools/bake_cast_shadow.py`
- `tools/fuse_priors_v2.py`
- `tools/merge_priors_v2_two_shadow.py`
- `tools/augment_priors_v2_to_v3.py`
- `train_thermal_robust.py`
- `tools/crop_roi_pipeline.py`
- `esttools/prep_eval_inputs.py`
- `esttools/metrics_eval_image.py`
- `tools/view_phase5_roi_splat.py`
- `tools/render_thermal_topdown_pred.py`

## 7. 实验执行命令说明

### 7.1 命令核验结论

详细核验见 [docs/experiment_command_audit.md](D:/PycharmProjects/wangxiao_code/docs/experiment_command_audit.md)。

最重要的结论：

- `COLMAP -> colmap2transforms.py -> train_rgb.py` 必须写入正式复现文档
- `bake_priors.py`、`train_thermal_v1.py` 等旧脚本不应再作为主流程命令
- `fuse_priors_v2.py` 当前参数名是 `--semantic_map`，不是 `--semantic_png`
- `augment_priors_v2_to_v3.py` 当前参数名是 `--semantic_num_classes`
- `render_video_robust.py` 在根目录，不在 `tools`

### 7.2 推荐使用的命令

#### A. COLMAP 与 RGB-3DGS

```powershell
conda activate D:\conda_envs\ntrgs
cd /d D:\PycharmProjects\wangxiao_code
set PYTHONPATH=%cd%

python convert.py -s data\uav_scene
python colmap2transforms.py -s data\uav_scene
python train_rgb.py -s data\uav_scene -m output\debug_run --resolution 4 --iterations 7000
```

#### B. 顶视图、GT、基础 priors

```powershell
python render_top_down.py -m output\debug_run --width 2048 --height 2048 --resolution 4 --multiplier 0.85 --shift_x 0 --shift_y -1.2 --angle -31
python manual_align.py
python bake_priors_physics.py -m output\debug_run --backup_old
```

#### C. Phase4 priors 主链

```powershell
python tools\check_manifest.py --run_dir output\runs\2026-01-20_phase5_sunfix_el75 --debug_root output\debug_run
python tools\render_dsm_topdown.py ...
python tools\bake_hillshade.py ...
python tools\segment_semantic.py ...
python tools\bake_cast_shadow.py ...
python tools\fuse_priors_v2.py ...
python tools\merge_priors_v2_two_shadow.py ...
python tools\augment_priors_v2_to_v3.py ...
```

#### D. Phase5 训练

```powershell
python train_thermal_robust.py --model_path output\debug_run --gt_path output\debug_run\lst_gt.png --priors_path output\runs\2026-01-20_phase5_sunfix_el75\priors\priors_v3_2shadow_semOH_sunfacing.pt --semantic_map_path output\runs\2026-01-20_phase5_sunfix_el75\artifacts\semantic_map.npy --output_path output\runs\2026-01-20_phase5_sunfix_el75\phase5\full_zscore --iterations 5000 --lr 1e-3 --priors_norm zscore --pred_act sigmoid --predict_inview_only --amp --train_scale 0.5 --save_every 500 --seed 0
```

#### E. ROI 评测

```powershell
python tools\crop_roi_pipeline.py --mode precheck --run_base 2026-01-20_phase5_sunfix_el75 --run_crop 2026-01-20_phase5_cropROI --model_path output\debug_run --priors_path output\runs\2026-01-20_phase5_sunfix_el75\priors\priors_v3_2shadow_semOH_sunfacing.pt --cameras output\debug_run\cameras.json --ply output\debug_run\point_cloud\iteration_7000\point_cloud.ply --zoom 5.4 --shift_x 0.0 --shift_y -1.2 --angle -31.0 --multiplier 0.85
python tools\crop_roi_pipeline.py --mode crop --run_base 2026-01-20_phase5_sunfix_el75 --run_crop 2026-01-20_phase5_cropROI --model_path output\debug_run --priors_path output\runs\2026-01-20_phase5_sunfix_el75\priors\priors_v3_2shadow_semOH_sunfacing.pt --cameras output\debug_run\cameras.json --ply output\debug_run\point_cloud\iteration_7000\point_cloud.ply --zoom 5.4 --shift_x 0.0 --shift_y -1.2 --angle -31.0 --multiplier 0.85
python tools\crop_roi_pipeline.py --mode postcheck --run_base 2026-01-20_phase5_sunfix_el75 --run_crop 2026-01-20_phase5_cropROI --model_path output\debug_run --priors_path output\runs\2026-01-20_phase5_sunfix_el75\priors\priors_v3_2shadow_semOH_sunfacing.pt --cameras output\debug_run\cameras.json --ply output\debug_run\point_cloud\iteration_7000\point_cloud.ply --zoom 5.4 --shift_x 0.0 --shift_y -1.2 --angle -31.0 --multiplier 0.85
```

#### F. 论文图与交互查看

```powershell
python tools\render_thermal_topdown_pred.py -m output\debug_run --priors_path output\runs\2026-01-20_phase5_sunfix_el75\priors\priors_v3_2shadow_semOH_sunfacing.pt --thermal_ckpt output\runs\2026-01-20_phase5_sunfix_el75\phase5\full_zscore\thermal_net_robust.pth --out_dir output\runs\2026-01-20_phase5_sunfix_el75\paper_figs\topdown_full_zscore --zoom 5.4 --shift_y -1.2 --angle -31 --multiplier 0.85
```

```powershell
python tools\view_phase5_roi_splat.py --scene_variant full --run_base_dir output\runs\2026-01-20_phase5_sunfix_el75 --run_crop_dir output\runs\2026-01-20_phase5_cropROI --render_res 640 --save_res 1536 --start_group full_zscore --start_camera_mode train
```

## 8. 常见问题与注意事项

1. `phase4.json` 不是完整一键链。  
   它的 `pipeline` 不会自动覆盖 `cast_shadow -> merge -> augment -> ROI -> eval` 全链。

2. `fuse_priors_v2.py` 的旧命令容易写错。  
   当前应使用 `--semantic_map`，不要再写 `--semantic_png`。

3. `augment_priors_v2_to_v3.py` 的旧参数名已失效。  
   当前是 `--semantic_num_classes`。

4. `render_video_robust.py` 的脚本位置容易写错。  
   它在根目录，不在 `tools`。

5. `manual_align.py` 不是通用 CLI。  
   它依赖固定输入输出路径，更像交互工具。

6. `plot_loss.py`、`summary.py`、`all_loss.py` 带硬编码路径。  
   这些脚本更适合作者本地一次性整理，不适合直接对外发布为标准命令。

7. `segment_semantic.py` 若没有 `transformers`，会自动 fallback。  
   当前已有 run 的 `semantic_stats.json` 说明真实主结果用过 fallback。

## 9. 与论文复现直接相关的最小脚本集合

如果只为了“从零到论文结果图”，建议最少关注：

- `convert.py`
- `colmap2transforms.py`
- `train_rgb.py`
- `render_top_down.py`
- `manual_align.py`
- `bake_priors_physics.py`
- `tools/render_dsm_topdown.py`
- `tools/bake_hillshade.py`
- `tools/segment_semantic.py`
- `tools/bake_cast_shadow.py`
- `tools/fuse_priors_v2.py`
- `tools/merge_priors_v2_two_shadow.py`
- `tools/augment_priors_v2_to_v3.py`
- `train_thermal_robust.py`
- `tools/crop_roi_pipeline.py`
- `esttools/prep_eval_inputs.py`
- `esttools/metrics_eval_image.py`
- `tools/view_phase5_roi_splat.py`
- `tools/render_thermal_topdown_pred.py`

---

补充说明：

- 详细命令核验请看 [docs/experiment_command_audit.md](D:/PycharmProjects/wangxiao_code/docs/experiment_command_audit.md)
- 全量脚本职责总表请看 [docs/script_inventory.md](D:/PycharmProjects/wangxiao_code/docs/script_inventory.md)
