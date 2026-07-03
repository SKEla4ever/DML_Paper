# Research Specification V1

## 1. Working Title

**When Is Personalization More Communication-Efficient Than Compression? A Comparative Study on Federated Wearable Human Activity Recognition**

中文暂定题目：

**个性化何时比压缩更具通信效率？面向可穿戴行为识别的联邦学习比较研究**

## 2. Research Positioning

本研究定位为一篇**问题驱动的系统比较实验论文**。研究不预设某一种算法在所有条件下最优，也不将“降低单轮传输字节数”等同于真正的通信效率，而是在统一实验条件和通信预算下，比较三类联邦学习策略：

1. 全局模型与异质性修正方法；
2. 模型更新压缩方法；
3. 个性化联邦学习方法及其与适度压缩的组合。

研究场景为基于智能手机和可穿戴传感器的 Human Activity Recognition（HAR）。每名用户被视为一个联邦客户端，原始传感器数据保留在本地。

## 3. Motivation and Research Gap

可穿戴 HAR 具有三个同时存在的特征：

- 用户的动作模式、身体特征、设备型号和佩戴方式不同，形成自然的 user-level Non-IID 数据；
- 手机、手表和边缘设备受到带宽、电量、算力和在线时间限制；
- 一个全局模型的平均性能不能反映少数活动类别和表现较差用户的实际体验。

现有研究通常分别关注以下问题：

- 通过量化、稀疏化或减少通信轮次降低通信量；
- 通过个性化、聚类或局部参数保留处理用户异质性；
- 通过 FedProx、SCAFFOLD 等方法减轻 client drift。

但是，在可穿戴 HAR 场景中，仍缺少在**统一通信预算、统一模型和统一数据划分**下，对以上方法进行系统比较的研究。尤其不清楚：

- 减少传输字节是否必然提高端到端通信效率；
- 强压缩是否会对少数活动类别和弱势用户造成不成比例的损害；
- 个性化方法通过减少无效全局聚合和收敛轮数，是否可能比单纯压缩更加通信高效；
- 个性化与适度压缩能否形成更好的性能—通信 Pareto trade-off。

## 4. Research Objective

本研究的总体目标是：

> 在用户级 Non-IID 的可穿戴 HAR 场景中，建立统一的性能与通信成本评估框架，识别压缩、异质性修正和个性化方法在不同通信预算及异质性条件下的适用范围。

研究不试图寻找跨数据集、跨预算的唯一最优算法，而是寻找**条件性最优策略**，并形成可复现的实践建议。

## 5. Research Questions

### RQ1: Non-IID 对通信效率有什么影响？

随着用户数据异质性增强，FedAvg 及其他全局方法达到给定性能目标所需的通信轮数和总通信量如何变化？

### RQ2: 压缩带来的字节节省是否转化为有效性能收益？

不同量化位宽和稀疏比例如何影响平均性能、少数活动识别、用户间公平性及达到目标性能所需的总通信量？

### RQ3: 个性化能否被视为一种通信效率策略？

FedPer、FedRep、FedBN 或 FedProto 等方法是否能以更少的累计通信量达到与全局模型相同或更高的用户级性能？

### RQ4: 个性化与压缩是否存在互补关系？

适度压缩与个性化方法结合后，是否优于强压缩的全局模型，并形成更好的通信—性能 Pareto frontier？

### RQ5: 方法优势是否依赖具体条件？

不同方法的表现是否受到 Non-IID 程度、客户端参与率、设备分布偏移、活动类别不平衡及通信预算的显著影响？

## 6. Testable Hypotheses

- **H1:** 用户数据异质性增强时，全局方法达到目标 Macro-F1 所需的累计通信量显著增加。
- **H2:** 强量化或强稀疏化能够降低每轮通信量，但会更明显地降低 minority-activity F1 和 worst-user performance。
- **H3:** 个性化方法在中度至重度 Non-IID 条件下能够以更少累计通信量达到给定用户级性能。
- **H4:** 个性化方法与适度压缩的组合，在平均 Macro-F1、最差用户表现和总通信量之间形成优于单纯强压缩的 Pareto trade-off。
- **H5:** 不存在对所有数据集、Non-IID 程度和通信预算都最优的方法；最佳策略具有明显的条件依赖性。

## 7. Scope

### Included

- 基于惯性传感器时间序列的监督式活动分类；
- 横向联邦学习；
- 用户级客户端划分；
- natural Non-IID、label skew、quantity skew 和 device feature shift；
- 同步联邦训练；
- 模型更新的上行和下行通信成本；
- 全局、个性化、压缩及混合方法的比较。

### Excluded from V1

- 在线 concept drift；
- 标签噪声和半监督学习；
- 隐私机制，如差分隐私和安全聚合；
- 恶意客户端和鲁棒聚合；
- 异步联邦学习；
- 大规模 Transformer 或 foundation model；
- 无线信道、延迟和能耗的硬件级仿真。

以上问题可作为后续扩展，但不进入第一版核心实验。

## 8. Datasets and Federated Scenarios

### 8.1 UCI HAR

用途：框架调试、可行性验证和与经典 HAR 研究对齐。

- Client：每名受试者；
- Data：智能手机加速度计和陀螺仪窗口；
- Activities：walking、walking upstairs、walking downstairs、sitting、standing、laying；
- 特点：规模较小、处理成熟，但用户和设备多样性有限。

### 8.2 WISDM

用途：主要用户异质性实验。

- Client：每名用户；
- Data：手机或手表传感器时间序列；
- 特点：用户行为差异更自然，可用于观察个性化收益；
- 注意事项：明确选用的 WISDM 版本、设备类型和标签清洗规则。

### 8.3 HHAR

用途：用户异质性和设备特征偏移实验。

- Client：用户，或用户—设备组合；
- Data：不同品牌智能手机和智能手表的惯性传感器数据；
- 特点：可区分 user shift 与 device shift；
- 风险：样本量和客户端数量有限，需要谨慎划分训练、验证和测试数据。

### 8.4 Data Partitioning

每个数据集至少评估以下设置：

1. **Natural user split:** 保留真实用户边界和标签分布；
2. **Controlled label skew:** 通过 Dirichlet 参数控制标签异质性；
3. **Quantity skew:** 控制不同客户端的样本数量；
4. **Device shift:** 在 HHAR 中按设备型号或设备类型构建分布偏移。

需要确保每个客户端拥有独立的 train/validation/test 时间窗口，避免重叠窗口造成数据泄漏。

## 9. Shared Model Architecture

主模型采用轻量级 1D-CNN：

- 多通道传感器窗口输入；
- 2–3 个一维卷积块；
- normalization、activation 和 pooling；
- 一个共享表示层；
- 一个活动分类头。

设计原则：

- 所有可比较方法尽可能使用相同参数规模和训练配置；
- FedPer/FedRep 仅改变共享层和本地层的训练或聚合方式；
- FedBN 在客户端保留 normalization 参数；
- FedProto 交换类别原型并保留可比较的特征提取器；
- 暂不将 CNN、LSTM、Transformer 的架构差异作为研究变量。

## 10. Methods

### 10.1 Reference Bounds

- Centralized training：非隐私、非联邦的性能参考上限；
- Local-only：零通信参考线。

### 10.2 Global and Heterogeneity-Aware FL

- FedAvg；
- FedProx；
- FedAdam 或 SCAFFOLD，依据实现稳定性选择一个进入主实验；
- 另一个可进入补充实验。

### 10.3 Personalized FL

- FedPer 或 FedRep，主实验只保留其中一个；
- FedBN；
- FedProto。

### 10.4 Communication Compression

- Uniform quantization：8-bit 和 4-bit；
- Top-k sparsification：至少两个保留比例；
- 如使用 error feedback，必须明确报告并设置对应消融实验。

### 10.5 Hybrid Candidates

- Personalized FL + 8-bit quantization；
- Personalized FL + mild Top-k sparsification；
- 是否保留为正式方法由 pilot study 决定。

V1 不预先宣称 hybrid 是原创算法。只有当初步实验揭示稳定、可解释且现有工作未充分覆盖的机制时，才考虑发展成新方法。

## 11. Communication Accounting

通信成本必须以真实传输内容为基础计算，不能仅报告 rounds。

每种方法统计：

- 每轮每客户端 uplink bytes；
- 每轮每客户端 downlink bytes；
- 累计 uplink、downlink 和 total bytes；
- 达到预设 Macro-F1 阈值所需累计通信量；
- 未达到目标时的固定预算最终性能。

需要计入：

- 模型参数、梯度或模型差分；
- 量化 scale、索引和稀疏编码元数据；
- SCAFFOLD control variates；
- FedProto prototypes 和类别标识；
- 个性化方法实际共享的参数子集。

主实验至少提供两种通信预算：

- Low-budget；
- Medium-budget；
- 如计算资源允许，再加入 High-budget。

具体预算值应在 pilot study 后根据完整 FedAvg 模型每轮传输量确定。

## 12. Evaluation Metrics

### Predictive Performance

- Accuracy；
- Macro-F1；
- Per-activity precision、recall 和 F1；
- Minority-activity recall/F1。

### Personalization and Fairness

- Mean per-user accuracy/Macro-F1；
- Median per-user performance；
- Worst 10% user performance；
- 用户间性能标准差；
- 需要时报告 10th percentile。

### Communication Efficiency

- Total transmitted MB；
- Rounds to target Macro-F1；
- MB to target Macro-F1；
- Final Macro-F1 under fixed communication budget；
- Area under the performance-versus-communication curve。

`Macro-F1 per MB` 仅作为辅助指标，因为简单比值可能偏向通信量接近零但性能很低的方法。

### Computational Cost

- 每轮本地训练时间；
- 总训练时间；
- 客户端峰值内存或模型参数量。

计算指标作为补充，不与真实设备能耗等同。

## 13. Experimental Matrix

### Pilot Phase

- Dataset：UCI HAR；
- Methods：Local-only、FedAvg、FedProx、FedPer/FedRep、FedProto、FedAvg-Q8、Personalized-Q8；
- Seeds：至少 3；
- 目标：验证实现正确性、通信统计和主要趋势。

### Main Phase

- Datasets：WISDM、HHAR；
- Non-IID：natural split + 至少两个 controlled severity；
- Participation：至少两个客户端参与率；
- Compression：无压缩、适度压缩、强压缩；
- Seeds：正式结果建议 5；
- 所有主要比较使用相同训练预算和调参协议。

### Ablations

- 个性化层位置或共享比例；
- 量化位宽；
- Top-k 保留比例；
- 是否使用 error feedback；
- personalization-only、compression-only、hybrid；
- uplink-only 与双向通信统计的差异。

## 14. Statistical Analysis

- 对多随机种子报告 mean ± standard deviation；
- 对主要方法进行成对统计比较；
- 优先使用适合多数据集/多条件的非参数检验；
- 同时报告 effect size 和置信区间，不只报告 p-value；
- 不根据单次最佳结果得出结论；
- 对超参数搜索范围和选择规则保持一致并完整记录。

## 15. Expected Results and Decision Rules

研究结果可能出现三种情况：

### Outcome A: Personalization clearly dominates under equal budgets

支持“个性化能够作为通信效率机制”的中心论点，并进一步分析其在何种 Non-IID 条件下成立。

### Outcome B: Compression dominates in mild Non-IID, personalization dominates in severe Non-IID

形成条件性方法选择框架。这是非常有价值且可能最现实的结果。

### Outcome C: Existing methods show no stable advantage

分析失败模式。如果发现固定压缩率对不同客户端造成明显不公平，可进一步研究 client-adaptive compression 或 personalization-aware compression。

只有在 pilot 和 main experiments 显示明确缺口后，才决定是否提出新方法。

## 16. Intended Contributions

在不预设新算法的前提下，论文计划贡献为：

1. 建立 wearable HAR 中 personalization、compression 和 heterogeneity correction 的统一比较框架；
2. 采用真实字节计量，在固定通信预算下比较平均性能、少数活动性能和弱势用户表现；
3. 揭示方法优势随 Non-IID、设备偏移和通信预算变化的条件；
4. 提供面向实际 wearable HAR 系统的方法选择建议；
5. 发布可复现实验配置、数据划分、通信计量代码和结果。

如果实验支持，将增加：

6. 提出并验证一个轻量的 personalization-aware compression 方案。

## 17. Threats to Validity

- 公共 HAR 数据集规模和设备种类有限；
- 模拟通信字节数不能完全代表真实网络延迟和能耗；
- 单一 1D-CNN 的结论不一定适用于所有模型；
- 不同 PFL 方法共享参数范围不同，必须谨慎保证比较公平；
- Dirichlet label skew 不能完全替代真实用户行为差异；
- 窗口切分可能产生时间相关的数据泄漏；
- 调参预算不一致可能造成方法排名偏差。

## 18. Deliverables

- 冻结后的 Research Specification；
- 文献矩阵和 baseline 实现清单；
- 可复现的数据预处理与 federated split；
- 统一实验代码与通信统计模块；
- pilot report；
- main experiment tables and figures；
- ablation and statistical analysis；
- paper manuscript；
- README、配置文件和随机种子记录。

## 19. Immediate Next Steps

1. 审阅并修改本规范的题目、范围、数据集和方法清单；
2. 为每个 RQ 定义最终表格或图形输出；
3. 检查 WISDM、HHAR 的版本、许可、用户数量和传感器字段；
4. 确定 FedPer/FedRep 以及 FedAdam/SCAFFOLD 的取舍；
5. 固定 UCI HAR pilot 实验配置；
6. 再开始实现，不在实现过程中随意扩大研究范围。
