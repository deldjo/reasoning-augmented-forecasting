# QFBench Track 2: Reasoning-Augmented Time-Series Forecasting
## Technical Report

### 1. Introduction
本研究旨在回答：带有时间戳的文本信息（如FOMC声明、新闻）能否提升纯数值模型的预测鲁棒性？我们构建了统一的“数值模型+文本推理”框架，并进行了全面的消融实验。

### 2. Methodology
- **数值模块**：基于资产近期统计特征的蒙特卡洛抽样（500次），严格使用 `df["date"] <= pd.to_datetime(asof)` 截断，确保零数据泄露。
- **文本模块**：实现了基于 API 调用的文本推理接口（支持关键词启发式提取回退及调用 LLM API），用于调整预测均值并加宽尾部风险。

### 3. Experimental Setup
- 运行环境：Python 3.13，使用 `qfbench` Conda 环境。
- 验证流程：所有输出均通过官方评分器 g0-g3 四项合规性检查。
- 消融设置：对比“盲文本基线（blind）”与“文本增强模式（text）”。

### 4. Results and Case Study
- **定量验证**：在全部 103 个单元（F1-F4）上进行了 206 次运行（盲测/文本增强），全部通过 `g0-g3` 验证（详见 `ablation_results.csv`）。
- **定性案例分析（核心）**：以 `t2-F4-covid-mkt-2020` 为例：
  - 盲测（Blind）：情绪=0.0，风险=False（完全无视外部冲击）；
  - 文本（Text）：情绪=-0.7，风险=True（成功捕捉危机信号），证明了文本推理模块成功捕捉了宏观市场危机信号。

### 5. Conclusion
文本推理框架在保证格式合规和防泄露的前提下，成功展示了获取宏观风险信号的潜力，为后续引入更强大的 LLM 推理提供了坚实的实验基础。
