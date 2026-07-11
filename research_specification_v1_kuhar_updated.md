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

研究场景为基于智能手机和可穿戴传感器的 Human Activity Recognition（HAR）。原则上每名用户被视为一个联邦客户端，原始传感器数据保留在本地；仅在 HHAR device-shift 专项实验中使用 user-device pair 作为分析单元，并同时按 physical user 聚合报告结果。

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

FedRep、Ditto，以及面向 device feature shift 的 FedBN 和 prototype-based FedProto，是否能以更少的累计通信量达到与全局模型相同或更高的用户级性能？

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

### 8.1 CAPTURE-24

用途：主要 natural Non-IID、personalization 和 user fairness 实验。

- Client：每名 participant，共 151 个真实 user-level clients；
- Data：dominant-wrist Axivity AX3 三轴 accelerometer，官方采样率为 100 Hz，每人约 24 小时 free-living recording；
- Scale：约 3,883 小时，其中约 2,562 小时具有 activity annotation；
- Labels：206 种 fine-grained text annotations，并提供多种 published coarse-label mappings；
- Primary label protocol：使用官方 Willetts 10-class activities-of-daily-living mapping，包括 `sleep`、`sitting`、`standing`、`household-chores`、`manual-work`、`walking`、`mixed-activity`、`vehicle`、`sports` 和 `bicycling`；
- Reproducibility：记录 `annotation-label-dictionary.csv` 中实际使用的 column name、文件 checksum、十类 label order 和完整 fine-to-coarse mapping；不得在观察方法结果后修改 label mapping；
- Missing annotation：无标签时间段表示 unknown，不自动归入 `mixed-activity` 或其他活动类，不进入 supervised loss 和主要 performance metrics，但保留其时长与分布统计；
- Label sensitivity：官方 Walmsley 4-class intensity mapping（`sleep`、`sedentary`、`light physical activity`、`moderate-to-vigorous physical activity`）仅进入 supplementary sensitivity experiment；
- 特点：用户数量较多，具有真实的 label、quantity 和 behavioral heterogeneity，最适合支撑 personalization 与 worst-user performance 的主要结论；
- 限制：所有 participant 使用相同型号的 wrist accelerometer，不能用于验证 device-model shift；数据虽于 2024 年正式发表，实际采集于 2014–2016 年。

主实验使用 10-second non-overlapping windows 作为初始协议，与官方 benchmark 对齐。若 pilot 显示计算成本过高，可在不改变时间覆盖和 split 边界的前提下比较 25、50 和 100 Hz resampling，并在主实验开始前冻结 input sampling rate。

CAPTURE-24 primary split 使用 deterministic grouped, duration-balanced, stratified 60/20/20 protocol：

1. **Atomic bout construction:** 对每名 participant 按 timestamp 排序并删除重复 timestamp；exact raw annotation 改变、annotation 进入或离开 missing、timestamp gap 大于 2 秒、或 timestamp 非单调时开启新的 atomic bout；
2. **Map after segmentation:** 先根据 exact raw annotation 构建 atomic bouts，再映射到 Willetts 10-class labels；即使相邻 bouts 映射到同一 coarse label，也不合并，以免制造 pseudo-bout；
3. **Grouped allocation:** 以完整 atomic bout 为最小分配单位，先按 global coarse class 执行 deterministic 60/20/20 duration balancing，再进行 client-aware refinement；
4. **Joint duration balancing:** 分层目标按 retained 10-second window count 而不是 bout 数量计算；refinement 最小化 `2 × N_clients × global-class ratio error + client-total ratio error + client-class ratio error`，从而避免 singleton long bouts 系统性地全部进入 train，同时兼顾每个 client 的 split feasibility；
5. **Windowing after split:** 完成 bout allocation 后，在每个 atomic bout 内生成 10-second non-overlapping windows；window 不跨 bout boundary，不足 10 秒的尾部丢弃；
6. **No forced duplication:** singleton 或 low-support class bout 只进入一个 split，不复制、不切分，也不通过 oversampling 改变 validation/test；
7. **Global coverage constraint:** 在不违反 client-local bout integrity 的前提下，确保十个 coarse classes 在 global train、validation 和 test 中均有 support；
8. **Frozen manifest:** 固定 split seed，发布 participant、atomic-bout、split、raw annotation、coarse label、window count 和时间边界 manifest 及 checksum。

Primary protocol 不删除 bout boundary 周围的 samples。作为 targeted sensitivity analysis，对 FedAvg、FedRep 和 Ditto 比较 10-second boundary guard；另使用 chronological 70/10/20 grouped split 检验结论对 temporal distribution shift 的敏感性。Chronological split 不进入 primary comparison，因为它会额外混入 time-of-day 和 behavior-sequence shift。

#### CAPTURE-24 Manifest Audit V1（2026-06-15）

- Source archive SHA-256：`69740c22d3e000367988373336dc6486c10fe3f3f929811fd66e4d84861e40e2`；官方 archive CRC check 通过，共识别 151 个 participant files；
- Dictionary：`label:WillettsSpecific2018`，206 条 exact fine-to-coarse mappings，dictionary SHA-256 为 `d90c949724aad7e25c884bfa1fdd7eee8e5a1b56c1c34f60e2849c5d3124d34a`，unmapped annotation 为 0；
- Atomic bouts：26,407；其中 13,109 个 missing-annotation bouts 被排除，189 个 annotated bouts 因不足一个 10-second window 被排除；
- Supervised windows：915,769；global train/validation/test 为 564,966 / 175,196 / 175,607，即 61.69% / 19.13% / 19.18%；
- Class balance：十类在三个 splits 中均有 support；最大 class-level absolute ratio deviation 为 4.63 percentage points，来自 indivisible long `sleep` bouts；
- Client feasibility：150/151 clients 满足至少 3 个 supported test classes；P094 仅有 2 个，因此保留在 global metrics，但不进入 primary per-user denominator；
- Seen/unseen support：833 个 supported client-class pairs 中 832 个为 seen；P078 的 `sports` 为唯一 locally-unseen supported pair，并按预注册规则单独报告；
- Integrity：window manifest 的 915,769 行与 assigned bouts、window counts、offsets、labels 和 splits 完全一致；同 seed 独立 regeneration 得到相同 SHA-256；
- Frozen artifacts 与完整审计见 `data/processed/capture24/` 和 `reports/capture24/capture24_manifest_audit_v1.md`。任何后续修改必须产生新的 manifest version，不得覆盖 V1 的实验身份。

### 8.2 KU-HAR

用途：severe label skew、quantity skew 和 weak-client stress test。

- Source：Mendeley Data Version 5，DOI `10.17632/45f952y38r.5`，license 为 CC BY 4.0；
- Files：使用 `2.Trimmed_interpolated_data.zip` 中的原始 recording files，不使用已汇总的 pre-windowed table；
- Client：subject ID；官方说明为 90 participants，Version 5 archive 实际包含 89 个 distinct subject IDs；
- Primary modality：三轴 accelerometer，100 Hz；gyroscope 仅进入 supplementary six-channel sensitivity experiment；
- Activities：18 类 scripted activities；minimum-support main cohort 的固定 label space 为 activity IDs `0`–`16`，`Table-tennis`（`17`）只进入 full sparse 18-class sensitivity；
- Interpretation：该异质性很大程度上由 data-collection protocol 和 activity assignment 造成，因此称为 protocol-induced skew，不将其等同于完全自然的 behavioral Non-IID。

Primary protocol 使用完整 recording file 作为不可分割 split unit，并采用 3-second non-overlapping windows：

1. **Quality exclusions:** 缺失 accelerometer 的 recording 从 primary protocol 排除；缺失 gyroscope 的 recording 仍可进入 accelerometer-only primary protocol，但不得进入 six-channel experiment；
2. **Exact duplicates:** same-subject exact duplicate 仅保留 lexicographically first recording；cross-subject exact duplicate 的所有副本均排除，避免 user-level leakage；
3. **Sensor mismatch:** accelerometer/gyroscope valid length 不一致不影响 accelerometer primary protocol；six-channel experiment 必须按每个 recording 的 common valid length 截断；
4. **Index-based windowing:** 由于个别 recording 存在 timestamp quantization，window boundary 使用 interpolation 后的 row order/sample index，不依赖 timestamp；timestamp 仅用于 integrity audit；
5. **Split before windowing:** 先分配完整 recording，再在 recording 内构建 300-sample windows；window 不跨 recording boundary，不足 300 samples 的尾部丢弃；
6. **Support reservation:** 某个 client-activity 至少有两个独立 recordings，且每个至少包含 3 个完整 windows，才定义为 split-feasible activity；对每个 evaluable client 的三个 strongest-repeat-support activities，分别保留一个完整 train recording 和一个完整 test recording；
7. **Joint allocation:** 其余 recordings 使用 seed `20260615` 进行 deterministic joint balancing，目标同时包含 full-cohort class ratio、minimum-cohort class ratio、client-total ratio 和 client-class ratio；不切分、不复制 recording，也不通过 validation/test oversampling 修正比例；
8. **Evaluation support:** client-class 至少有 3 个 test windows 才定义为 supported test class；client 至少有 3 个 supported test classes 才进入 per-user Macro-F1、worst 10% 和用户间标准差的 denominator。

KU-HAR 冻结以下两个实验 cohort：

1. **Full sparse cohort:** 质量排除后有 88 个 usable clients，使用 18-class label space；全部 88 个 clients 可作为 training clients，其中 54 个具有至少 3 个 split-feasible activities，构成冻结的 per-user evaluable set，其余 34 个 clients 不进入主要 per-user fairness denominator；
2. **Minimum-support cohort:** client 必须属于 full-sparse evaluable set、覆盖至少 8 个 observed activities，并具有至少 5 分钟 retained accelerometer data；最终得到 50 个 clients，使用固定 17-class label space，作为 KU-HAR 主要 per-user fairness comparison。

Subject `1089` 的两个 recordings 均属于 cross-subject exact duplicate groups，严格排除后无 usable window，因此不计为 federated client。该处理优先保证数据身份和 leakage control，不为维持 nominal client count 而保留可疑副本。

#### KU-HAR Recording Split Audit V1（2026-06-15）

- Source archive SHA-256：`9fe5d0052f2f1d6711afac42ee4badd968116afa8ba4b8ba591f4fdd771c2ec2`；ZIP CRC check 通过，共扫描 1,945 个 CSV recordings；
- Quality：primary accelerometer protocol 保留 1,938 个 recordings、排除 7 个；1 个 recording 缺失 accelerometer，1 个缺失 gyroscope，752 个存在 sensor valid-length mismatch；
- Duplicate policy：排除 2 个 same-subject duplicate copies 和 4 个 cross-subject ambiguous copies；所有冻结规则在方法训练前确定；
- Full sparse split：19,641 个 3-second windows；train/validation/test 为 12,351 / 3,578 / 3,712，即 62.88% / 18.22% / 18.90%；18 类在三个 splits 中均有 support，最大 class-level absolute ratio deviation 为 5.79 percentage points；
- Minimum-support split：13,627 个 windows；train/validation/test 为 8,177 / 2,632 / 2,818，即 60.01% / 19.31% / 20.68%；17 类在三个 splits 中均有 support，最大 class-level absolute ratio deviation 为 6.49 percentage points；
- Client feasibility：full sparse evaluable set 为 54/88 clients，minimum-support cohort 为 50 clients，且两组的冻结 clients 均实现至少 3 个 supported test classes；
- Seen/unseen support：54-client evaluable set 共有 279 个 supported client-class pairs，其中 244 个 seen、35 个 locally unseen；50-client minimum-support cohort 为 264 个 supported pairs，其中 229 个 seen、35 个 locally unseen；
- Integrity：所有 retained recordings 恰好进入一个 split，所有 excluded recordings 均未进入 window manifest，每个 usable client 都有 train data，window manifest 共 19,641 行，独立 regeneration 得到相同 checksums；
- Frozen artifacts 与完整审计见 `data/processed/kuhar/` 和 `reports/kuhar/kuhar_recording_split_audit_v1.md`。任何后续修改必须产生新的 manifest version，不得覆盖 V1 的实验身份。

### 8.3 HHAR

用途：targeted device feature shift 实验。

- Primary modality：phone accelerometer；
- Analysis unit：user-device pair，同时保留 physical-user identifier；
- Data：9 名 physical users、8 部 smartphones、6 类 activities，并包含不同型号和具体 device identifiers；
- Protocol：删除 `null` label，但使用 `null` 和 timestamp gaps 识别 recording boundaries；
- Resampling：仅在连续 non-null segment 内按 `Creation_Time` 排序，以 50 Hz linear interpolation，构建 3-second、150-sample、non-overlapping windows；超过 1 second 的 sensor-time gap 为 hard boundary；
- Synchronization：每个 physical-user/activity stratum 以多数设备的 run count 和 median `Arrival_Time` interval 定义 canonical executions；完整设备按 temporal ordinal 对齐，不完整设备按 interval overlap 对齐；跨越多个 canonical executions 的 ambiguous raw run 从 primary protocol 排除；
- 特点：同一 physical user 使用多种设备，可研究 device-model shift 和 FedBN；
- 限制：只有 9 名 physical users，user-device pairs 不是独立用户，不能用于主要 fairness 结论或将 pair 数量当作独立样本量。

HHAR 仅运行预先指定的 targeted method subset，不进入所有方法和所有 Non-IID 因素的完整实验矩阵。

#### HHAR Manifest Audit V1（2026-07-11）

- Source：UCI HHAR activity-recognition archive，DOI `10.24432/C5689X`，activity archive SHA-256 为 `d4c0c53b195b523859bf71f5a349d164c7a604a321ff6b0972fbed6e03b46582`，ZIP CRC check 通过；
- Raw phone accelerometer：13,062,475 rows，9 名 physical users、8 台 devices、4 个 device models、71 个 observed user-device pairs；
- Synchronization：冻结 130 个 canonical synchronized executions；所有 execution 均无 repeated same-device run；1 个 `s3mini_2` raw run 因跨越多个 canonical stair executions 而作为 ambiguous synchronization 排除；
- Windows：33,040 个 3-second、50 Hz windows；train/validation/test 为 20,364 / 6,367 / 6,309，即 61.63% / 19.27% / 19.10%；6 类在三个 splits 中均有 support；
- Clients：69 个 user-device pairs 至少保留一个有效 window，且全部具有 train data；`s3mini_2` 因 window volume 低于 device median 的 25% 被标记为 weak-device condition，但除 ambiguous run 外仍保留；
- Diagnostic：tiny-overfit 与 random-label controls 通过，centralized learning 明显有效，real/IID FedAvg 在相同 communication 下呈现可解释差异；但 validation 对 centralized、real FedAvg 和 IID FedAvg 均系统性难于 test，因此 algorithm tuning 前必须运行预注册的 split-seed sensitivity，不依赖单一 split 作方法选择。

### 8.4 UCI HAR

用途：smoke test、pipeline correctness、communication accounting 和与经典 HAR 工作对齐。

- Client：每名受试者，共 30 个 clients；
- Data：50 Hz smartphone accelerometer 和 gyroscope，官方提供 128-sample、50% overlap windows；
- Activities：walking、walking upstairs、walking downstairs、sitting、standing、laying；
- Protocol：使用 `Inertial Signals`，不使用 561-dimensional engineered features；
- 限制：规模较小且每个用户六类齐全，不支撑主要 natural Non-IID、fairness 或 device-shift 结论。

### 8.5 Shared Sensor Protocol

Primary cross-dataset comparison 使用三轴 accelerometer，以保持 CAPTURE-24、KU-HAR、HHAR 和 UCI HAR 的输入 modality 一致。KU-HAR 和其他具有 gyroscope 的数据集可在 supplementary modality sensitivity experiment 中使用 accelerometer + gyroscope，但不得将 sensor modality improvement 归因于 federated algorithm。

模型和 communication budget 在每个数据集内保持一致。由于 sampling rate、window length 和 input channels 可能不同，不直接比较跨数据集的 absolute transmitted MB；跨数据集综合结论以相对于该数据集 FedAvg reference cost 的 normalized communication budget 和方法排名稳定性为主。

### 8.6 Data Partitioning

数据划分遵循以下原则：

1. **Preserve client identity:** 不将不同用户的 samples 重新分配给 synthetic clients；
2. **Split before windowing:** 先按 recording、activity bout 或连续时间块划分 train/validation/test，再生成 windows；
3. **No boundary crossing:** window 不跨 activity transition、missing-annotation interval 或 data gap；
4. **CAPTURE-24:** 使用第 8.1 节定义的 raw-annotation atomic bouts 和 deterministic grouped 60/20/20 protocol；
5. **KU-HAR:** 使用第 8.2 节冻结的 quality exclusions、完整 recording split、support reservation 和 deterministic joint allocation；V1 不使用 contiguous temporal block 回避 recording support 不足；
6. **HHAR:** 同一次 activity execution 的多设备同步记录必须进入相同 split，避免同一动作同时出现在 train 和 test；
7. **UCI HAR:** 相邻的 50% overlap windows 不得随机分散到不同 split，尽可能按完整 activity run 分组。

Controlled label skew 通过在真实 client 边界内进行 class-aware downsampling 构造；quantity skew 通过 client-wise sample-budget subsampling 单独构造。两者不在同一 primary experiment 中同时变化，以保持因果解释清晰。所有 inclusion、downsampling 和 split rules 在查看方法比较结果前冻结。

## 9. Shared Model Architecture

主模型采用轻量级 1D-CNN：

- 多通道传感器窗口输入；
- 2–3 个一维卷积块；
- normalization、activation 和 pooling；
- 一个共享表示层；
- 一个活动分类头。

设计原则：

- 所有可比较方法尽可能使用相同参数规模和训练配置；
- FedRep 将 feature representation 设为共享部分，将 classification head 保留在客户端；
- Ditto 的 global model 与 personalized model 使用相同基础架构，personalized model 不参与传输；
- FedBN 在客户端保留 normalization 参数；
- FedProto 交换类别原型并保留可比较的特征提取器；
- 暂不将 CNN、LSTM、Transformer 的架构差异作为研究变量。

## 10. Methods

### 10.1 Reference Bounds

- Centralized training：非隐私、非联邦的性能参考上限；
- Local-only：零通信参考线。

### 10.2 Global and Heterogeneity-Aware FL

- **FedAvg:** 标准 global FL baseline，也是通信预算和 rounds-to-target 的主要参照；
- **FedProx:** 通过 proximal regularization 限制 local model drift，作为低额外通信开销的 heterogeneity-aware baseline；
- **SCAFFOLD:** 使用 server/client control variates 修正 Non-IID 下的 client drift，进入主实验；其额外 control-variate payload 必须纳入通信统计；
- **FedAdam:** 作为 supplementary method，用于区分 server-side adaptive optimization 与显式 client-drift correction 的效果。

### 10.3 Personalized FL

- **FedRep:** 主实验中的 partial-model personalization 方法，共享 feature representation、保留 local classification head；FedPer 因机制高度重叠，不进入主实验；
- **Ditto:** 主实验中的 full-model personalized regularization 方法，用于评估 personalization 对 mean per-user、worst-user performance 和用户间公平性的影响；
- **FedBN:** device feature shift 专项方法，主要用于 HHAR 的跨设备条件，不要求参加所有 label-skew 实验；
- **FedProto:** prototype-based communication baseline，通过交换 class prototypes 而非完整 model updates 进行协作；因通信协议不同，结果需同时单独报告并谨慎解释。

上述方法覆盖 partial-model personalization、full-model personalization、normalization localization 和 prototype exchange 四类不同机制。V1 不再将 FedPer、pFedMe、FedDyn、FedALA 或 meta-learning 方法加入主实验，以控制实验规模；必要时可在 supplementary experiments 中选择一个代表方法进行 sensitivity check。

### 10.4 Communication Compression

- Uniform quantization：8-bit 和 4-bit；
- Top-k sparsification：至少两个保留比例；
- 如使用 error feedback，必须明确报告并设置对应消融实验。

### 10.5 Hybrid Candidates

- **Primary hybrid:** FedRep + 8-bit uniform quantization，对实际共享的 representation update 进行量化；
- **Secondary candidates:** FedRep + mild Top-k sparsification，以及 Ditto global branch + 8-bit quantization；
- 是否保留 secondary hybrid 由 pilot study 中预先定义的 validation Macro-F1、worst-user performance 和 MB-to-target 规则决定。

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
- FedRep 仅计入实际共享的 representation 参数，明确排除未传输的 local head；
- Ditto 仅计入传输的 global branch，明确排除仅在本地维护的 personalized model；
- FedBN 仅计入共享参数，明确排除客户端本地保存的 normalization parameters 和运行统计；
- 个性化方法实际共享的其他参数子集。

FedAdam 的 optimizer states 仅保存在 server，不计入网络传输，但需在 computational/storage cost 中说明。对于 FedProto 等非完整模型交换协议，除 total bytes 外，还需报告 payload composition，避免仅凭较小消息体得出不公平结论。

主实验至少提供两种通信预算：

- Low-budget；
- Medium-budget；
- 如计算资源允许，再加入 High-budget。

对每个数据集，将默认 participation rate 下的一轮 full-precision FedAvg 总传输量定义为一个 normalized communication unit，同时始终报告实际 uplink、downlink 和 total bytes。Low、Medium 和 High budget 使用该数据集 communication unit 的预先冻结倍数定义，避免因 input channels 或模型细节差异而直接比较跨数据集 absolute MB。

## 12. Evaluation Metrics

### Predictive Performance

- Accuracy；
- Macro-F1；
- Per-activity precision、recall 和 F1；
- Minority-activity recall/F1；
- CAPTURE-24 headline metric 为包含全部十类的 `Macro-F1-10`；
- CAPTURE-24 同时报告 `Core-9 Macro-F1`，其计算时排除 `mixed-activity`，但模型仍执行相同的 10-class prediction，且 `mixed-activity` F1 必须单独报告；
- Walmsley 4-class intensity Macro-F1 仅作为 label-granularity sensitivity metric。

### Personalization and Fairness

- Mean per-user accuracy/Macro-F1；
- Median per-user performance；
- Worst 10% user performance；
- 用户间性能标准差；
- 需要时报告 10th percentile；
- 每个数据集预先报告具有有效 test support 的 client 数量、每类 support 和被排除 client 的原因；
- CAPTURE-24 仅在有 annotation 的 test windows 上计算 supervised metrics；
- CAPTURE-24 中某个 client-class 至少有 10 个 test windows 才定义为 supported test class；
- CAPTURE-24 中某个 client 至少有 3 个 supported test classes，才进入 per-user Macro-F1、worst 10% 和用户间标准差的 primary denominator；
- CAPTURE-24 evaluable-client set 和 supported-class set 在任何方法训练前由 frozen split manifest 确定，并对所有方法保持一致；
- CAPTURE-24 分别报告 seen-class performance（local train 和 test 均至少有 10 个 windows）与 locally-unseen-class performance（test 有 support，但 local train 少于 10 个 windows）；
- KU-HAR 的 client-class 至少有 3 个 test windows 才定义为 supported test class，client 至少有 3 个 supported test classes 才进入 per-user denominator；
- KU-HAR full sparse cohort 使用冻结的 54-client evaluable set，minimum-support cohort 使用冻结的 50-client set；分别报告 seen-class 与 locally-unseen-class performance，且所有方法使用相同 support sets。

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

- Smoke-test dataset：UCI HAR；
- Methods：Centralized、Local-only、FedAvg、FedProx、SCAFFOLD、FedRep、Ditto、FedProto、FedAvg-Q8、FedRep-Q8；
- Seeds：至少 3；
- 目标：验证 implementation correctness、communication accounting、target threshold 可达性和 evaluation pipeline；
- Dataset sanity checks：在 CAPTURE-24 的预先固定小型 client subset、KU-HAR 冻结的 50-client minimum-support cohort 和 HHAR device-shift subset 上验证 parsing、split integrity、label coverage 和 memory/runtime，不将该阶段结果作为论文结论；
- FedBN 不在 UCI HAR smoke test 中强制运行，在 HHAR device-shift sanity check 中单独验证。

### Main Phase

- Primary dataset：CAPTURE-24 natural user-level cohort；
- Secondary stress dataset：KU-HAR minimum-support cohort；
- Sparse-client sensitivity：KU-HAR full sparse cohort；
- Targeted device-shift dataset：HHAR；
- Non-IID：CAPTURE-24 natural heterogeneity，以及预先定义的 controlled label-skew 和 quantity-skew sensitivity；
- Participation：至少两个客户端参与率；
- Compression：无压缩、适度压缩、强压缩；
- Core methods：FedAvg、FedProx、SCAFFOLD、FedRep、Ditto、FedProto；
- Core compression comparisons：FedAvg-Q8、FedAvg-Q4、FedAvg-Top-k，以及 FedRep-Q8 primary hybrid；
- Conditional method：FedBN 用于 HHAR device-shift 条件；
- Supplementary methods：FedAdam 和通过 pilot selection rule 保留的 secondary hybrid；
- Seeds：CAPTURE-24 和 KU-HAR minimum-support primary comparisons 建议 5；高成本 sensitivity experiments 至少 3；
- HHAR split sensitivity：正式 hyperparameter selection 前至少运行 3 个预注册 split seeds；model-seed sensitivity 与 split-seed sensitivity 分开报告，不将两种 variance 混为一个标准差；
- Local computation：使用固定 local optimizer steps 或固定 local sample budget，避免 CAPTURE-24 的 natural quantity skew 使大客户端在每轮获得不受控的额外计算；
- 所有主要比较使用相同训练预算和调参协议。

Main Phase 不执行所有因素的完整 Cartesian product：

- CAPTURE-24 在默认 participation rate 下运行所有 core methods 和 core compression comparisons，形成 primary comparison grid；
- KU-HAR minimum-support cohort 运行 core methods、FedAvg-Q8 和 FedRep-Q8，用于检验结论在 protocol-induced severe skew 下是否稳定；
- KU-HAR full sparse cohort 仅运行 FedAvg、SCAFFOLD、FedRep、Ditto 和 FedAvg-Q8，并将训练参与统计与可评估-client metrics 分开报告；
- KU-HAR minimum-support main comparison 使用冻结的 17-class label space；full sparse sensitivity 使用 18-class label space，两者不直接比较 absolute Macro-F1；
- HHAR 仅运行 FedAvg、FedProx、SCAFFOLD、FedBN、FedRep 和 FedAvg-Q8；
- participation-rate sensitivity、controlled skew、sensor-modality sensitivity 和高成本 supplementary methods 只在预先指定的方法与条件子集上运行。

具体 subset、client inclusion rule 和 resource cap 必须在 primary method results 揭盲前写入实验协议，以控制计算成本和 multiple-comparison risk。

### Ablations

- FedRep 的 personalization layer 位置和共享参数比例；
- Ditto regularization coefficient；
- FedBN 中 local normalization 的贡献；
- FedProto 的 prototype dimension、缺失类别处理和 prototype payload；
- CAPTURE-24 的 Willetts 10-class ADL 与 Walmsley 4-class intensity mapping；
- CAPTURE-24 primary grouped 60/20/20 split 与 chronological 70/10/20 split；
- CAPTURE-24 是否使用 10-second bout-boundary guard；
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
- CAPTURE-24 的 user-level confidence interval 和 bootstrap 以 physical user 为 resampling unit；
- CAPTURE-24 primary metric、Core-9 metric、evaluable-client set 和 supported-class threshold 均在方法训练前冻结；
- HHAR 的统计分析以 physical user 而不是 user-device pair 为独立 resampling unit，避免 pseudo-replication；
- KU-HAR 分别报告 minimum-support cohort 与 full sparse cohort，不将两个 cohort 的 per-user distributions 合并；
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

- CAPTURE-24 具有较多 users 和真实 free-living behavior，但所有用户使用同型号 wrist accelerometer，无法单独支持 device-shift 结论；
- CAPTURE-24 的 camera annotation 采样较稀疏并经过时间外推，且存在大量未标注区间，可能引入 label noise 和 activity-dependent missingness；
- CAPTURE-24 的 Willetts 10-class mapping 将多个 fine-grained activities 合并，可能造成 class-internal heterogeneity；`mixed-activity` 也不是单一可分辨动作；
- grouped duration-balanced split 改善 leakage control，但不能保证每个 client 的每个 class 同时出现在 train、validation 和 test，因此必须区分 seen 与 locally-unseen classes；
- CAPTURE-24 虽于 2024 年发表，但采集于 2014–2016 年，不能完全代表最新 consumer wearable hardware；
- KU-HAR Version 5 的实际 subject IDs、usable client count、activity coverage 和数据量与概述中的“90 participants、18 activities”存在差异，且 skew 部分来自 collection protocol；
- KU-HAR minimum-support filtering 会改变 client population，必须与 full sparse cohort 并列报告以揭示 selection bias；
- KU-HAR minimum-support cohort 不包含 `Table-tennis`，因此其 17-class results 与 full sparse 18-class sensitivity 不可直接按 absolute Macro-F1 排名；
- HHAR 只有 9 名 physical users，user-device pairs 不是独立用户，其结果只支持 targeted device-shift 分析；
- 不同数据集的 sampling rate、window length 和可用 sensor channels 不完全一致，跨数据集 absolute communication cost 不直接可比；
- 模拟通信字节数不能完全代表真实网络延迟和能耗；
- 单一 1D-CNN 的结论不一定适用于所有模型；
- 不同 PFL 方法共享参数范围不同，必须谨慎保证比较公平；
- FedProto 与 model-update methods 的通信协议不同，直接排名可能混入 protocol advantage；
- FedBN 依赖 normalization 选择和 local batch size，其收益不应泛化到所有 Non-IID 类型；
- synthetic controlled label/quantity skew 不能完全替代真实用户行为差异；
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

1. 继续逐节审阅并冻结题目、Research Questions、Scope 和方法角色；
2. 建立 literature matrix 与 baseline implementation audit，记录原始论文、官方实现、license、关键超参数和实际通信 payload；
3. **已完成（2026-06-15）：** 按已冻结的 Willetts 10-class ADL、missing-annotation 和 atomic-bout joint 60/20/20 protocol 生成 CAPTURE-24 split manifest；V1 audit 已冻结 global/per-client class support、150-client evaluable cohort、seen/unseen coverage、split deviation 和 checksums；
4. **已完成（2026-06-15）：** 完成 KU-HAR V5 recording-level quality audit，冻结 88-client full sparse cohort、54-client evaluable set、50-client minimum-support cohort、17/18-class label spaces、support-aware whole-recording split、seen/unseen definitions、manifest 与 checksums；
5. **已完成（2026-07-11）：** 冻结 HHAR phone-accelerometer subset、user-device client identity、physical-user grouping、canonical synchronized-execution split、50 Hz/3-second window protocol、manifest checksums 与 setup diagnostic；
6. 为每个 RQ 预先定义最终 table/figure、target Macro-F1、normalized communication budgets 和 statistical comparison；
7. 冻结 1D-CNN、normalization 选择、FedRep split point、Ditto regularization search space、fixed local computation budget 及统一调参预算；
8. 写出完整的 UCI HAR smoke-test protocol 和 CAPTURE-24 primary experiment protocol，包括 input sampling rate、seeds、participation rate、stopping rule、validation rule 和 resource cap；
9. 完成以上 specification 工作后再开始实现，不在实现过程中随意扩大研究范围。
