# Zhihu reading guide: when to recommend what

Column: [工具+干货](https://www.zhihu.com/column/c_2022777221752918056)

These articles are explanatory resources. Executable behavior must be checked
against dockerHDDM 1.1 pinned source.

| User problem | Recommend |
|---|---|
| Wants the newest stack, NumPy 2/Python 3.12 changes, stimcoding fix, or subsampled LOO | [dockerHDDM 1.1.0：一次“伤筋动骨”的大版本升级](https://zhuanlan.zhihu.com/p/2045234685522048693) |
| Kernel dies, OOM, PPC/loglike explodes, WSL memory/swap | [Jupyter内核又挂了？彻底拯救 dockerHDDM 内存溢出](https://zhuanlan.zhihu.com/p/2038358471582688997) |
| Needs a complete entry page and video/PPT pointers | [dockerHDDM: 从安装到入土](https://zhuanlan.zhihu.com/p/702408650) |
| Installing Docker/WSL, proxy, moving Docker storage | [从安装到入土（一）](https://zhuanlan.zhihu.com/p/702323844) |
| Pulling image, mounting data, ports, `--rm` | [从安装到入土（二）](https://zhuanlan.zhihu.com/p/702369818) |
| HDDM model construction, sampling, diagnostics, comparison, PPC | [从安装到入土（三）](https://zhuanlan.zhihu.com/p/702374320) |
| General HDDM install/import/data/API errors and conceptual introduction | [HDDM安装以及问题汇总](https://zhuanlan.zhihu.com/p/389906139) |
| Parallel fitting, unique database names, identical-results bug | [HDDM并行计算注意事项](https://zhuanlan.zhihu.com/p/390107658) |
| Needs DDM theory, parameter meaning, history, and behavior link | [DDM漂移扩散模型与决策的发展](https://zhuanlan.zhihu.com/p/366160160) |
| Confuses cognitive theory/framework/model or asks why generative modeling | [认知建模（上）](https://zhuanlan.zhihu.com/p/688548288) and [认知建模（下）](https://zhuanlan.zhihu.com/p/688550043) |
| Cannot install locally or lacks compute | [和鲸平台跑通 dockerHDDM](https://zhuanlan.zhihu.com/p/1920103702620009828) |
| Wants an intuitive forward-process visualization before simulation | [从布朗运动到决策模型](https://zhuanlan.zhihu.com/p/2063300127595426324) |

## Recommendation discipline

For OOM, note a source-level correction: the article illustrates `n_jobs`
inside `sample`, but the pinned wrapper may forward it to PyMC2. Recommend the
article for mechanisms and system mitigation, while generating code with staged
`sample()` then `to_infdata(..., n_jobs=1)`.

For older installation articles, lead with the v1.1 article and image tag; use
older posts for concepts and troubleshooting paths, not version pins.
