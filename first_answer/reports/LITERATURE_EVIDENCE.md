# C题文献证据库：内容摘要、可支持结论与引用格式

更新日期：2026-09-23。

用途：供《服务于脑机接口与精神性疾病诊断的脑电图计算模型》三问建模、程序实现和论文写作统一引用。本文整理的是论文的核心方法、与本题的对应关系和引用元数据，不替代原文，也不把文献方法当成本数据已经成立的结论。正式写论文时，只引用实际使用过的方法，不为“显得文献多”而堆砌。

统一原则：

1. 原始MAT回答“数据里发生了什么”；原实验说明回答“事件代表什么”；论文回答“分析方法为什么合理”；题解只提供候选思路。
2. 文献不能证明±1/±2的本题语义、A/B身份、正确性规则、真实AUC或疾病诊断效果。
3. 三通道Fz/F3/F4足以做低维传感器级ERP和解码，但不足以唯一定位LGN、海马或脑网络连接。
4. 反应时间只有在刺激起点和行为终点均被实验元数据确认后才计算；否则写“事件间隔”。
5. 下列参考文献按GB/T 7714—2015的期刊格式给出；机器可读BibTeX见 `references/c_problem_references.bib`。

## 一、先回答：这些论文能否“解决”本题？

| 本题环节 | 可以依靠程序解决 | 需要论文定义/约束 | 仍必须补原实验资料 |
| --- | --- | --- | --- |
| 事件表 | 检出提示、±1段、±2边沿和时间差 | 事件锁定与透明报告原则 | 每个码的真实含义、超时和正确规则 |
| 问题一 | 滤波、分段、质量指标、ERP、分半与半模拟 | 预处理、ERP自由度控制、可重复性 | 单位、参考电极、在线滤波 |
| 问题二 | 生成低维模型并在留出EEG上分类 | 时间分辨MVPA、线性权重解释、神经质量模型 | 原刺激尺寸/位置及逐试次布局 |
| 问题三 | 拟合V/M/C状态空间并做嵌套消融 | 输入—状态—观测、工作记忆与动力学依据 | 若验证行为：真实动作、正误、反应时日志 |
| 临床诊断 | 当前不能 | 文献只能说明潜在应用 | 患者/对照、诊断标签和独立临床队列 |

结论：这些论文能够把三问变成一套合格、可验证的科学计算方案，但不能补造数据字典。当前可以开工问题一、问题二的传感器级模型和问题三的功能性状态模型；行为与临床结论必须降级，直到原实验元数据或新数据补齐。

## 二、问题一：事件、预处理、ERP与可靠性

### [R1] Pernet et al. (2020)：M/EEG可重复性和报告规范

核心内容：OHBM COBIDAS MEEG委员会系统讨论EEG/MEG采集、事件、预处理、试次选择、统计与共享中影响可重复性的因素，强调完整报告事件、参数、排除标准和分析流程。

本题用途：建立400试次事件表；保留不同事件锁定候选；记录过滤、基线、试次剔除、窗口和代码版本。支撑“先确认事件，再解释认知”。

不能支持：它不能告诉我们本数据±1/±2是什么，也不能证明某个间隔就是反应时间。

建议正文表述：为避免将不同实验事件对应的神经活动错误归于同一认知过程，本文先建立逐试次事件表，并在事件语义确认后才进行认知解释。

GB/T 7714：PERNET C R, GARRIDO M I, GRAMFORT A, et al. Issues and recommendations from the OHBM COBIDAS MEEG committee for reproducible EEG and MEG research[J]. Nature Neuroscience, 2020, 23(12): 1473-1483. DOI: 10.1038/s41593-020-00709-0.

### [R2] Luck & Gaspelin (2017)：防止ERP分析自由度制造假阳性

核心内容：说明事后选择电极、时间窗、条件和分析方式会大幅提高假阳性风险；推荐预先规定分析或将探索与独立验证分开。

本题用途：普通ERP保留为基准；训练/验证集确定窗口和电极组合，冻结后只在测试集评价；不允许看完整左右曲线后挑差异最大的区间。

不能支持：不能提前规定本题必须出现P300，也不能证明250—500 ms内任意正峰都是P300。

GB/T 7714：LUCK S J, GASPELIN N. How to get statistically significant effects in any ERP experiment (and why you shouldn't)[J]. Psychophysiology, 2017, 54(1): 146-157. DOI: 10.1111/psyp.12639.

### [R3] Bigdely-Shamlo et al. (2015)：PREP标准化早期预处理

核心内容：提出面向大规模EEG的标准化早期预处理，包括线噪处理、稳健参考和坏通道识别，并输出可审计的处理信息。

本题用途：支撑“先固定基础预处理，再比较加权和平滑”，以及保存质量报告。

限制：PREP主要面向多通道数据；本题仅3个EEG电极，无法可靠使用其坏通道插值或稳健平均参考思想的完整版本。因此只借鉴流程化、质量审计和固定参数，不照搬算法。

GB/T 7714：BIGDELY-SHAMLO N, MULLEN T, KOTHE C, et al. The PREP pipeline: standardized preprocessing for large-scale EEG analysis[J]. Frontiers in Neuroinformatics, 2015, 9: 16. DOI: 10.3389/fninf.2015.00016.

### [R4] Delorme & Makeig (2004)：EEGLAB和单试次/ICA工具链

核心内容：介绍EEGLAB对连续EEG、事件、分段、单试次动态和ICA等分析的支持。

本题用途：说明事件分段、ERP、单试次分析和伪迹检查属于成熟工具链；可借鉴数据结构和处理顺序。

限制：论文证明ICA是可用方法，不证明3通道ICA能有效分离眼电、肌电和脑源。本题不将强ICA去伪迹作为默认方案。

GB/T 7714：DELORME A, MAKEIG S. EEGLAB: an open source toolbox for analysis of single-trial EEG dynamics including independent component analysis[J]. Journal of Neuroscience Methods, 2004, 134(1): 9-21. DOI: 10.1016/j.jneumeth.2003.10.009.

### [R5] Polich (2007)：P3a/P3b与P300解释边界

核心内容：综述P300的幅度、潜伏期、P3a/P3b、注意、记忆更新和任务难度等因素。

本题用途：若数据、参考方式和头皮分布支持，可把预先定义窗口内的事件相关正成分作为P300候选；同时报告幅度和潜伏期测量方式。

限制：只有Fz/F3/F4且事件语义不全时，不能仅凭一个正峰确认P3b或其脑源；本题主指标应先写“事件相关响应/候选P300”。

GB/T 7714：POLICH J. Updating P300: An integrative theory of P3a and P3b[J]. Clinical Neurophysiology, 2007, 118(10): 2128-2148. DOI: 10.1016/j.clinph.2007.04.019.

### [R6] Oostenveld et al. (2011)：FieldTrip成熟分析流程

核心内容：介绍用于EEG/MEG和侵入电生理的MATLAB工具箱，覆盖预处理、分段、时频、统计和可视化。

本题用途：说明本项目的过滤—epoch—ERP—统计流程属于标准方法链；可用于复核自建Python实现的定义。

限制：工具箱存在不代表默认参数适合本题；采样率、滤波边界和事件语义仍须单独确定。

GB/T 7714：OOSTENVELD R, FRIES P, MARIS E, et al. FieldTrip: Open Source Software for Advanced Analysis of MEG, EEG, and Invasive Electrophysiological Data[J]. Computational Intelligence and Neuroscience, 2011, 2011: 156869. DOI: 10.1155/2011/156869.

## 三、问题二：线性分析、动态解码与生成模型

### [R7] Parra et al. (2005)：EEG线性分析的一般框架

核心内容：讨论多通道EEG的线性组合与判别/降维方法，强调用明确目标函数估计空间权重。

本题用途：支撑低维线性基准、正则化LDA/logistic和受控空间组合；先用简单模型，再与复杂机制模型比较。

限制：原文大量场景为高密度EEG；本题只有3个通道，不能声称获得高分辨率空间滤波或脑源分离。

GB/T 7714：PARRA L C, SPENCE C D, GERSON A D, et al. Recipes for the linear analysis of EEG[J]. NeuroImage, 2005, 28(2): 326-341. DOI: 10.1016/j.neuroimage.2005.05.032.

### [R8] Grootswagers et al. (2017)：时间分辨MVPA

核心内容：提供在诱发时间序列上逐时间点/时间窗解码的教程，讨论分类、交叉验证和结果解释。

本题用途：以Fz/F3/F4或少量预先定义时窗特征训练时间分辨LDA/logistic，输出AUC(t)、平衡准确率和置换区间；所有标准化和选窗在训练折内完成。

限制：方法不能保证本数据可解码；AUC≈0.84的教程仿真不属于真实EEG证据。

GB/T 7714：GROOTSWAGERS T, WARDLE S G, CARLSON T A. Decoding Dynamic Brain Patterns from Evoked Responses: A Tutorial on Multivariate Pattern Analysis Applied to Time Series Neuroimaging Data[J]. Journal of Cognitive Neuroscience, 2017, 29(4): 677-697. DOI: 10.1162/jocn_a_01068.

### [R9] Haufe et al. (2014)：分类权重不等于神经贡献

核心内容：区分用于预测的后向模型权重和用于解释信号来源的前向模式，说明相关噪声会使分类器权重产生误导。

本题用途：预测结果用分类性能评价；若解释电极模式，计算Haufe变换后的前向模式并报告不确定性。

限制：即使转换为前向模式，3个头皮通道仍不能把F3/F4的模式直接解释成LGN、海马或其他具体脑源。

GB/T 7714：HAUFE S, MEINECKE F, GÖRGEN K, et al. On the interpretation of weight vectors of linear models in multivariate neuroimaging[J]. NeuroImage, 2014, 87: 96-110. DOI: 10.1016/j.neuroimage.2013.10.067.

### [R10] Wilson & Cowan (1972)：兴奋—抑制群体动力学

核心内容：以耦合非线性微分方程描述局部兴奋性与抑制性神经群体，展示稳态、滞后和极限环等动力学。

本题用途：为问题二的E/I神经群扩展提供方程来源，刺激作为外部输入，群体状态再映射到三电极观测。

限制：方程是群体级近似，不等于逐神经元真实机制；参数不能仅凭3通道自由估计，必须固定大部分并做可辨识性检查。

GB/T 7714：WILSON H R, COWAN J D. Excitatory and Inhibitory Interactions in Localized Populations of Model Neurons[J]. Biophysical Journal, 1972, 12(1): 1-24. DOI: 10.1016/S0006-3495(72)86068-5.

### [R11] David & Friston (2003)：MEG/EEG神经质量模型

核心内容：以少量神经群的耦合和神经动力学生成可观测MEG/EEG响应，强调群体动力学和连接对信号形态的作用。

本题用途：支撑“图形输入→低维神经群状态→EEG观测”而非直接手画一条脑电波；用于机制扩展和半模拟实验。

限制：它不提供本题图形类别的固定参数，也不证明左右三角形必然产生镜像ERP。

GB/T 7714：DAVID O, FRISTON K J. A neural mass model for MEG/EEG: coupling and neuronal dynamics[J]. NeuroImage, 2003, 20(3): 1743-1755. DOI: 10.1016/j.neuroimage.2003.07.015.

### [R12] Azzopardi & Petkov (2014)：轮廓空间配置的形状表示

核心内容：COSFIRE模型以局部滤波响应及其相对空间配置形成可训练的形状选择性表示。

本题用途：支撑问题二从轮廓、方向和相对位置构造低维刺激特征；可用于检验“形状”而非简单像素差。

限制：COSFIRE是计算视觉模型，不是EEG观测模型；模型特征不能直接称为神经活动，且原刺激的位置、大小需统一控制。

GB/T 7714：AZZOPARDI G, PETKOV N. Ventral-stream-like shape representation: from pixel intensity values to trainable object-selective COSFIRE models[J]. Frontiers in Computational Neuroscience, 2014, 8: 80. DOI: 10.3389/fncom.2014.00080.

### [R13] Nguyen-Danse et al. (2021)：低密度EEG源连接的局限

核心内容：用模拟和真实数据比较低密度与高密度EEG的源功能连接重建；电极减少会降低定位和连接强度重建的可靠性。

本题用途：为“三通道只做传感器级低维功能变量，不做精确脑源/连接定位”提供直接限制依据。

限制：该研究最低仍高于本题3个EEG通道，不能据此声称三通道源定位可行；它主要用来界定结论边界。

GB/T 7714：NGUYEN-DANSE D A, SINGARAVELU S, CHAUVIGNÉ L A S, et al. Feasibility of Reconstructing Source Functional Connectivity with Low-Density EEG[J]. Brain Topography, 2021, 34: 709-719. DOI: 10.1007/s10548-021-00866-w.

## 四、问题三：输入—隐状态—观测、记忆和模型比较

### [R14] Friston et al. (2003)：动态因果建模的结构思想

核心内容：把系统写成外部输入、隐状态动力学和观测输出，通过生成模型估计状态耦合和输入调制。

本题用途：借鉴u→z→y结构，将提示、位置和任务作为输入，将V/M/C作为低维隐状态，将三电极作为观测；采用嵌套模型比较。

限制：原论文以fMRI DCM为主要发展对象。本项目只借鉴结构，不声称实现完整DCM，也不作因果脑网络结论。

GB/T 7714：FRISTON K J, HARRISON L, PENNY W. Dynamic causal modelling[J]. NeuroImage, 2003, 19(4): 1273-1302. DOI: 10.1016/S1053-8119(03)00202-7.

### [R15] Glomb et al. (2022)：EEG计算模型的尺度与可验证性

核心内容：综述从细胞/微回路、神经质量到全脑网络的EEG计算模型，以及模型如何与实验EEG和行为联系；强调模型要服务于具体可测问题并在现实与简化之间平衡。

本题用途：支撑选择低维、可检验模型而非追求生理细节；区分机制模拟、数据拟合和临床应用。

限制：综述不能给出本题专属参数，且没有患者标签时不能把模型包装为精神疾病诊断器。

GB/T 7714：GLOMB K, CABRAL J, CATTANI A, et al. Computational Models in Electroencephalography[J]. Brain Topography, 2022, 35(1): 142-161. DOI: 10.1007/s10548-021-00828-2.

### [R16] Daume et al. (2024)：工作记忆的多组件控制与保持

核心内容：在人类颅内记录中研究额叶控制与海马持续活动的耦合，结果支持工作记忆包含控制和存储相关组件。

本题用途：为把“记忆”设计成与视觉和认知控制不同的功能状态M提供生理动机，并支持比较V、V+M、V+M+C模型。

限制：这是颅内神经元/群体层面的证据，不能用三通道头皮EEG声称测到海马神经元或相位—幅度耦合。

GB/T 7714：DAUME J, KAMIŃSKI J, SCHJETNAN A G P, et al. Control of working memory by phase–amplitude coupling of human hippocampal neurons[J]. Nature, 2024, 629(8011): 393-401. DOI: 10.1038/s41586-024-07309-z.

### [R17] Kriegeskorte et al. (2008)：表征相似性分析

核心内容：以表征差异矩阵（RDM）比较脑活动、行为和计算模型的表示几何，不要求模型单元与测量通道逐一对应。

本题用途：当存在至少3—4个可靠条件时，比较模型状态和EEG条件模式的距离结构，作为逐点波形误差之外的补充。

关键限制：只有左/右两个条件时，2×2 RDM只有一个独立非对角距离，相关系数没有有效自由度，不能构成有意义的RSA。必须先获得形状×位置等可靠多条件标签，否则删除RSA。

GB/T 7714：KRIEGESKORTE N, MUR M, BANDETTINI P A. Representational similarity analysis—connecting the branches of systems neuroscience[J]. Frontiers in Systems Neuroscience, 2008, 2: 4. DOI: 10.3389/neuro.06.004.2008.

### [R18] Acebrón et al. (2005)：Kuramoto同步模型

核心内容：综述耦合相位振子的同步动力学、数值方法和扩展。

本题用途：仅在最终确实分析相位同步、并能从数据定义相位观测时作为可选扩展。

限制：当前主模型是状态空间/E-I动态；不能把任意加权模值称为神经同步，也不应为“模型高级”强行加入Kuramoto。

GB/T 7714：ACEBRÓN J A, BONILLA L L, PÉREZ VICENTE C J, et al. The Kuramoto model: A simple paradigm for synchronization phenomena[J]. Reviews of Modern Physics, 2005, 77(1): 137-185. DOI: 10.1103/RevModPhys.77.137.

## 五、文献—问题—公式—程序—评价对照表

| 论文位置 | 建模动作 | 主要公式/程序输出 | 验收标准 | 对应文献 |
| --- | --- | --- | --- | --- |
| 数据与事件 | 逐试次解析提示、±1起止、±2；保留未知语义 | `trial_table`、事件时间线、证据等级 | 400次可追溯；不按排序强配对 | R1 |
| 问题一基准 | 固定滤波、epoch、基线、普通ERP | ERP、保留率、固定窗幅度 | split-half、留出误差、区间 | R1—R6 |
| 问题一候选 | 伪迹质量权重+受罚平滑 | 权重、n_eff、平滑系数 | 半模拟恢复优于基准且不扭曲 | R2—R4 |
| 问题二编码 | 受控形状特征→低维状态→三电极 | y_hat、RMSE、R²、残差 | 留出预测优于类别均值基线 | R10—R12, R15 |
| 问题二解码 | 真实EEG→线性时间分辨分类 | AUC(t)、平衡准确率、置换区间 | 嵌套CV；仿真与真实分开 | R7—R9 |
| 问题三主模型 | z=[V,M,C]，任务输入进入状态方程 | M0—M3、RMSE、R²、状态曲线 | 同一留出集嵌套比较和消融 | R14—R16 |
| 表示层补充 | 多条件模型RDM与EEG RDM比较 | Spearman ρ/置换 | 至少3—4个可靠条件 | R17 |
| 结论边界 | 传感器级而非脑源/临床诊断 | 局限与待补资料 | 不越界到海马电位或疾病准确率 | R13, R15 |

## 六、最终论文建议引用分配

- “数据与预处理”段：R1、R3、R4、R6；不必四篇全堆在一句话里，按实际实现选择。
- “ERP窗口与统计控制”段：R2；若真正讨论P300，再加R5。
- “问题二解码”段：R8；解释权重时加R9；线性空间方法总述可加R7。
- “问题二生成模型”段：若采用E/I方程用R10、R11；若采用形状空间配置用R12。
- “问题三状态空间”段：R14、R15；记忆状态的功能动机用R16。
- “RSA”只在满足多条件前提并实际计算后引用R17。
- “低密度限制”段引用R13，用于解释为何不做精确源定位。
- R18只有实际实现相位同步模型才进入正文，否则留在候选文献，不放参考文献表。

引用纪律：一条引用只支撑它真正讨论的论断。不要用工具箱论文证明生理机制，不要用综述证明具体参数，不要用仿真论文证明本数据性能，也不要用工作记忆的颅内研究证明三通道头皮EEG直接观测到海马。

## 七、v7.1补充：候选子空间与独立推断

### R19. Kriegeskorte等（2009）：循环分析与重复使用数据

原题名：Circular analysis in systems neuroscience: the dangers of double dipping。

内容概括：在同一批噪声观测上先挑选响应、变量或区域，再用不独立的统计量确认效果，会扭曲效应估计或显著性。文章用分析示例讨论这种选择偏差，并强调选择与后续推断的独立性。

本题用途：U和秩只在训练中估计；Q1共同/差分主检验先使用固定P0，自适应U的效果在独立块上评估。不能因为训练数据驱动就宣布U是真实形状特征，也不能挑到最大峰后再进行未校正检验。

限制：论文不是本数据可交换性或显著性的证明，不提供当前降噪有效性的数值保证。本文块方案仍须满足自己的统计假设。

GB/T 7714：KRIEGESKORTE N, SIMMONS W K, BELLGOWAN P S F, et al. Circular analysis in systems neuroscience: the dangers of double dipping[J]. Nature Neuroscience, 2009, 12(5): 535-540. DOI: 10.1038/nn.2303.

作者开放稿：[PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC2841687/)；BibTeX键：kriegeskorte2009circular。

实现核对另用[SciPy单样本检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_1samp.html)和[配对检验](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_rel.html)文档；它们说明独立性及相关样本的适用条件，不代替本实验的设计说明。
