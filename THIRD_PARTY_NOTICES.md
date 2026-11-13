# 第三方代码与依赖

项目自身代码采用 [MIT License](LICENSE)。第三方代码与依赖继续遵循各自的许可证。

## Restormer

`model/model.py` 中的 `OverlapPatchEmbed`、`Downsample` 和 `Upsample` 与 Restormer 的实现一致或经过调整，因此保留上游版权与许可文本。

- 上游项目：https://github.com/swz30/Restormer
- 参考文件：https://github.com/swz30/Restormer/blob/main/basicsr/models/archs/restormer_arch.py
- 原始许可证：https://github.com/swz30/Restormer/blob/main/LICENSE.md

```text
MIT License

Copyright (c) 2022 Syed Waqas Zamir and contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```

## timm

`model/LWGA.py` 从 `timm.models.layers` 导入 `DropPath`，未将其源码复制进仓库。检查环境使用 timm 0.6.13，其上游许可证为 [Apache-2.0](https://github.com/huggingface/pytorch-image-models/blob/v0.6.13/LICENSE)。

## 数据与论文

数据集、原始医学影像、第三方模型权重及论文稿件不因本项目的 MIT 许可证而改变授权条件。数据应从原始提供方申请或下载。
