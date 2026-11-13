# DMAMSA-Net

**Arterial-Phase Liver MRI Synthesis via Dynamic Modality Adaptation and Multi-Scale Grouped Attention**

DMAMSA-Net 用于从非增强 MRI 生成动脉期肝脏 MRI 候选图像。模型结合动态模态适应系统（DMAS）和多尺度分组注意力（MSGA），建模动脉期内部的潜在增强差异，并提取不同尺度的解剖特征。

论文目前为投稿准备稿，正式发表信息后续补充。

## 方法概览

- **DMAS**：训练时，从真实动脉期图像提取潜在分支编码，通过三处特征仿射调制引导生成器；推理时，直接指定分支编号，逐个生成候选图像。
- **MSGA**：通过点、局部、中距离和全局注意力分支融合多尺度信息。
- **训练目标**：最小二乘对抗损失与 L1 重建损失；配置中的 `lambda_id` 表示重建损失权重。

默认有两个潜在分支，其编号不直接代表经过临床标注验证的早期或晚期动脉期。

## 目录结构

```text
DMAMSA/
├── main.py                    # 训练与候选图像生成入口
├── utils.py                   # 配置、PSNR/SSIM、学习率与图像保存
├── model/
│   ├── model.py               # 生成器、特征调制与判别器
│   ├── LWGA.py                # 多尺度分组注意力
│   └── phase_classifier.py    # 潜在分支选择
├── data/
│   ├── base.py                # 体数据读取、归一化与配对检查
│   ├── data_lld.py            # LLD-MMRI / Duke Liver
│   └── data_nih.py            # MRISegmenter
├── configs/                   # YAML 配置
├── data_list/                 # 病例或序列编号清单
├── requirements.txt
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```

## 环境安装

已验证的兼容环境为 Python 3.8.5、PyTorch 1.8.0、torchvision 0.9.0。
```bash
conda create -n dmamsa python=3.8 -y
conda activate dmamsa
python -m pip install -r requirements.txt
```

## 数据准备

数据加载器读取 NIfTI 三维体数据，并按病例清单的顺序划分训练、验证与测试集。`modalities` 按顺序指定输入模态和目标模态。

### LLD-MMRI 和 Duke Liver

使用已完成配准、调整到 256×256 的数据，目录格式如下。文件名可通过 `modalities` 修改。

```text
<data_root>/
├── <case_or_sequence_id>/
│   ├── C_pre_256.nii.gz
│   └── C_A_256_registered.nii.gz
└── ...
```

`data_list` 每行一个病例或序列目录名。LLD-MMRI 默认使用 `data_list/lld_shuffled.txt`；Duke 默认使用 `data_list/duke_list.txt`。

### MRISegmenter（配置名 NIH）

```text
<data_root>/
├── <case_id>/
│   ├── PRE.nii.gz
│   └── ART.nii.gz
└── ...
```

加载器默认反射填充或居中裁剪到 320×320，再双线性缩放到 256×256。可通过 `target_size` 和 `resize_size` 修改；设为 `null` 可关闭相应步骤。

训练/验证默认使用 `nih_shuffled.txt`，测试使用 `nih_test.txt`，两者配置为不同的数据根目录。

### 预处理约定

体数据按各自体积的最小值和最大值归一化到 `[-1, 1]`，常量体数据映射到 `-1`。`start_slice` 包含起点，`end_slice` 不包含终点；负值表示从末尾倒数，`null` 表示保留到末尾。

病例索引同样使用左闭右开区间。

## 配置与训练

优先使用以下配置：

| 数据集 | 配置文件 | 输入 → 目标 |
| --- | --- | --- |
| LLD-MMRI | `configs/config_lld.yaml` | `C_pre_256` → `C_A_256_registered` |
| Duke Liver | `configs/config_duke.yaml` | `C_pre_256` → `C_A_256_registered` |
| MRISegmenter | `configs/config_nih.yaml` | `PRE` → `ART` |

先修改所选配置中的 `dataroot`、`ckpt_dir`、GPU 编号及病例范围，然后运行：

```bash
python main.py --config configs/config_lld.yaml --mode train
```

`--mode` 会覆盖 YAML 顶层的 `mode`。运行时合并 `common` 与当前模式的配置，同名字段由当前模式覆盖，因此可以分别指定训练与测试的数据目录。

常用配置项：

| 配置项 | 含义 |
| --- | --- |
| `dataset` | `lld`、`duke` 或 `nih` |
| `modalities` | 输入和目标文件名，不包含 `.nii.gz` |
| `img_height` / `img_width` | 预期图像尺寸，均须能被 4 整除 |
| `num_classes` | 潜在增强分支数量，默认 2 |
| `att_kernel` | 中距离条带卷积核大小，必须为正奇数 |
| `num_refinement_blocks` | 解码器末端细化模块数量 |
| `start_num` / `end_num` | 当前训练或测试集的病例索引 |
| `start_num_val` / `end_num_val` | 验证集的病例索引 |
| `debugging` | 训练/验证最多加载 128 个切片 |
| `n_cpu` | DataLoader 工作进程数；本地调试可设为 0 |
| `profile` | 可选 THOP 统计；默认关闭 |

每轮保存 `G_latest.pth`、`D_latest.pth`，按 `checkpoint_interval` 保存带轮次的权重。训练与验证 PSNR/SSIM 写入 `ckpt_dir/log.txt`；验证集完整保留，并分别评价所有分支后取平均。当前仅保存模型权重，未实现完整断点续训，训练时 `epoch` 应设为 0。重复使用同一个输出目录会覆盖同名日志和权重。

## 推理与结果保存

在 `test` 中指定权重轮次或 `latest`，默认遍历全部分支：

```yaml
test:
  which_epoch: latest
  phase_index: all
  notes: ""
  start_num: 340
  end_num: 440
```

```bash
python main.py --config configs/config_lld.yaml --mode test
```

推理只读取非增强输入，真实动脉期图像可以不存在。结果保存为：

```text
<ckpt_dir>/epoch_<which_epoch><notes>/
├── branch_0/<case_id>_<slice_index>.png
└── branch_1/<case_id>_<slice_index>.png
```

设置 `phase_index: 0` 或 `1` 可只生成对应分支。PNG 为由 `[-1, 1]` 映射得到的 8 位灰度图。

也可以直接调用模型：

```python
import torch
from model import TLP

model = TLP(att_kernel=9, num_classes=2, num_refinement_blocks=2)
model.load_state_dict(torch.load("checkpoints/G_latest.pth", map_location="cpu"))
model.eval()
input_image = torch.zeros(1, 1, 256, 256)  # Replace with a normalized non-contrast image.
with torch.no_grad():
    candidates = [model(input_image, phase_index=i) for i in range(2)]
```

## 许可证与致谢

本项目代码采用 [MIT License](LICENSE)。部分基础模块参考 [Restormer](https://github.com/swz30/Restormer)，`DropPath` 来自 [timm](https://github.com/huggingface/pytorch-image-models)，相关说明和上游许可见 [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md)。
