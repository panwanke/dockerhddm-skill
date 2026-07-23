# dockerHDDM skill

> [English](README.md)

一个可复用的智能体技能，用于构建、解释、模拟、推断、组织和调试 **dockerHDDM 1.1** 项目。它封装了受维护的 HDDM、Kabuki、PyMC2、ArviZ 与 ssm-simulators 技术栈，让基于试次的漂移扩散（DDM / HDDM）建模可以通过一条斜杠命令实现可复现。

斜杠命令：**`/dockerhddm-workflow`**

## 概述

该技能在编写任何代码之前，先将请求路由到正确的工作流，随后强制执行严格的数据契约与分层调试规范。它面向被固定的 dockerHDDM `1.1.0` 运行时设计，而非任意的上游 HDDM 安装。

## 工作流

| 模式 | 适用场景 | 入口 |
|------|----------|------|
| **推断（Inference）** | 已存在试次级数据 | `references/inference-workflow.md` |
| **模拟 / 预测** | 已有参数、取值范围或理论 | `references/simulation-prediction-workflow.md` |
| **调试（Debugging）** | 报错、崩溃、结果不合理或设计存疑 | `references/debugging.md` |
| **项目初始化** | 需要干净可复用的项目 | `scripts/init_project.py` + `references/project-layout.md` |
| **API 解释** | 解释 HDDM / Kabuki / PyMC2 调用流程 | `references/api-hddm.md`、`references/api-kabuki-pymc2.md` |
| **实验设计** | 任务或样本量问题 | `references/experimental-design.md` |

混合请求按以下顺序处理：模拟 → 参数恢复 → 拟合真实数据 → 诊断 → PPC / 模型比较 → 解释。

## 快速开始

从内置模板初始化可复用项目结构：

```bash
python scripts/init_project.py <项目目录> --mode inference
python scripts/init_project.py <项目目录> --mode simulation
python scripts/init_project.py <项目目录> --mode hybrid
```

在假设任何 API 之前，先检查目标环境：

```bash
python scripts/inspect_environment.py
```

在昂贵采样前估算内存需求：

```bash
python scripts/estimate_memory.py
```

## 功能特性

- 将意图路由到精选工作流，而非临时拼凑的提示词。
- 强制执行「每行一个试次」的数据契约（RT 以秒为单位、必填列、编码需注明）。
- 生成完整推断流水线：校验 → 基线 → 冒烟拟合 → 生产采样 → 诊断 → PPC → 参数恢复。
- 区分前向模拟、条件/网格模拟、分层合成数据、后验预测模拟与参数恢复。
- 分层调试（环境 → 挂载 → 数据模式 → 模型 → 采样 → InferenceData → 收敛 → 假设）。
- 附带 40 篇转换后的参考 Notebook，以及精选阅读与文献指南。
- 注重隐私：将示例结构转换为通用角色目录，绝不暴露原始数据或凭据。

## 参考导航

- 精选 Notebook 导航：`references/notebook-routing.md`
- 全部转换 Notebook：`references/notebooks/index.md`
- 文章推荐：`references/zhihu-reading-guide.md`
- 文献综述：`references/literature-guide.md`
- Pan 等人（2025）dockerHDDM 论文：`references/literature/pan-2025-dockerhddm.md`
- Boag 等人（2025）实验设计：`references/literature/boag-2025-experimental-planning.md`

## 技术栈

| 组件 | 技术 |
|------|------|
| 贝叶斯推断 | HDDM、Kabuki、PyMC2 |
| 诊断 | ArviZ |
| 模拟器 | ssm-simulators |
| 运行时 | dockerHDDM 1.1.0（容器） |
| 语言 | Python |

## 许可证

MIT
