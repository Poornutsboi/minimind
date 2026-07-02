# 基于 MiniMind 的个性化充电方案调价模型技术路线

## 1. 研究目标

本文档设计一套基于 MiniMind 后训练的个性化充电调价系统。系统假设外部候选方案生成器已经为每辆车给出多个可选充电方案，例如在哪些站点充电、各站点充电多长时间、预计等待时间、对站点排队延误和用电压力的影响等。MiniMind 不负责生成这些候选充电方案，而是负责为每个候选方案生成个性化的站点充电单价或折扣策略，以刺激用户选择对系统贡献更优的方案，同时避免过度补贴导致利润损失。

核心问题可以概括为：

```text
给定用户状态、候选充电方案集合、每个方案的系统影响，以及过去时间窗内用户对价格的响应，
学习一个个性化价格策略，使用户更可能遵循系统贡献较优的方案，并在利润、排队延误、电网压力和用户接受率之间取得平衡。
```

该任务不是单纯的最优定价，也不是单纯的路径推荐，而是一个带有用户响应反馈的闭环个性化激励控制问题。

## 2. 系统总览

推荐采用“外部候选方案生成器 + MiniMind 个性化价格策略模型 + 用户响应预测器 + 在线校准器 + 安全约束层”的混合架构。

```text
用户充电请求
  -> 候选充电方案生成器
  -> 方案影响评估器
  -> 状态与候选方案编码
  -> MiniMind 生成个性化调价策略
  -> 用户响应预测器估计接受概率
  -> 安全约束层过滤价格
  -> 在线选择并投放
  -> 收集接受、拒绝、改约、取消、实际充电结果
  -> 滚动窗口敏感度更新
  -> 周期性 SFT / DPO / Agentic RL 后训练
```

各模块职责如下：

| 模块 | 职责 | 是否由 MiniMind 完成 |
| --- | --- | --- |
| 候选方案生成器 | 生成每辆车的多个可行充电方案 | 否 |
| 方案影响评估器 | 计算排队延误、电网压力、负载均衡、系统贡献 | 否 |
| 用户响应预测器 | 估计用户对价格、绕路、等待、换站的敏感度 | 否，建议独立建模 |
| MiniMind 调价模型 | 为每个候选方案生成站点级单价或折扣 | 是 |
| 安全约束层 | 控制价格上下限、波动幅度、最低利润、公平性 | 否 |
| 在线校准器 | 根据滚动窗口反馈实时调整敏感度和价格偏置 | 否 |
| 后训练管线 | SFT、DPO、Agentic GRPO/CISPO | 是，基于 MiniMind 训练 |

## 3. 问题建模

### 3.1 状态空间

每次用户请求到来时，系统构造状态 `s_t`：

```json
{
  "user_context": {
    "user_id_hash": "u_123",
    "vehicle_type": "sedan",
    "soc_now": 0.22,
    "soc_target": 0.80,
    "energy_need_kwh": 38.5,
    "deadline": "2026-07-01T20:30:00+08:00",
    "time_flexibility_min": 45,
    "detour_tolerance_km": 4.0,
    "historical_accept_rate": 0.57,
    "estimated_price_sensitivity": -1.18
  },
  "rolling_window": {
    "window_minutes": 30,
    "shown_offer_count": 320,
    "accept_rate": 0.42,
    "reject_rate": 0.46,
    "reschedule_rate": 0.09,
    "cancel_rate": 0.03,
    "avg_discount_per_kwh": 0.16,
    "avg_incentive_cost": 3.2,
    "avg_profit_per_session": 8.6
  },
  "station_context": {
    "stations": [
      {
        "station_id": "A",
        "base_price": 1.48,
        "queue_len": 9,
        "expected_wait_min": 18,
        "grid_pressure": 0.82
      }
    ]
  }
}
```

其中 `estimated_price_sensitivity` 不建议完全由 LLM 学习，应该由独立的响应预测器或 contextual bandit 在线估计，并作为 MiniMind 的输入特征。

### 3.2 候选方案集合

外部函数为用户生成 `K` 个候选充电方案：

```json
{
  "candidate_plans": [
    {
      "plan_id": "p1",
      "stations": [
        {"station_id": "A", "start_time": "18:30", "duration_min": 20},
        {"station_id": "B", "start_time": "19:10", "duration_min": 15}
      ],
      "total_energy_kwh": 38.5,
      "detour_km": 2.6,
      "expected_user_wait_min": 8.5,
      "queue_delay_delta": -4.2,
      "grid_pressure_delta": -0.13,
      "system_contribution_score": 0.84,
      "baseline_user_cost": 56.8
    }
  ]
}
```

这里的 `system_contribution_score` 可以由外部评估器计算，融合排队延误改善、站点负载均衡、电网峰值压力、服务率等指标。

### 3.3 动作空间

MiniMind 输出候选方案对应的个性化价格策略 `a_t`：

```json
{
  "pricing_strategy": [
    {
      "plan_id": "p1",
      "station_prices": [
        {"station_id": "A", "price_per_kwh": 1.32},
        {"station_id": "B", "price_per_kwh": 1.18}
      ],
      "total_discount": 4.6,
      "expected_accept_prob": 0.61,
      "expected_incentive_cost": 4.6,
      "reason_code": "high_system_score_with_moderate_discount"
    }
  ],
  "recommended_plan_id": "p1",
  "exploration_ratio": 0.05
}
```

如果生产系统更关注展示简洁性，也可以把动作空间简化为“每个候选方案总价/总折扣”，再由规则层拆分到各站点。但论文和算法研究更建议保留站点级价格，因为它能表达不同站点拥堵和电网压力的局部差异。

## 4. 优化目标

目标函数应同时考虑用户遵循、系统收益、激励成本和约束风险：

```text
R =
  alpha * Compliance
+ beta  * SystemBenefit
+ eta   * Profit
- gamma * IncentiveCost
- lambda_q * QueuePenalty
- lambda_g * GridPressurePenalty
- lambda_r * RejectionPenalty
- lambda_v * PriceVolatilityPenalty
- lambda_c * ConstraintViolationPenalty
```

关键指标定义如下：

| 指标 | 含义 |
| --- | --- |
| `Compliance` | 用户是否选择系统推荐的高贡献方案，或选择的方案贡献分是否足够接近最优 |
| `SystemBenefit` | 方案对排队延误、站点负载、电网压力的综合改善 |
| `Profit` | 用户实际充电带来的毛利 |
| `IncentiveCost` | 为刺激用户选择该方案所付出的折扣或补贴成本 |
| `QueuePenalty` | 用户选择后导致的站点排队恶化 |
| `GridPressurePenalty` | 对电网高峰负载造成的压力 |
| `RejectionPenalty` | 报价过高导致拒绝、取消或流失 |
| `PriceVolatilityPenalty` | 价格短时间内跳变过大 |
| `ConstraintViolationPenalty` | 违反价格上下限、最低利润、公平性或业务规则 |

论文中可以重点提出一个归一化指标：

```text
SystemBenefit per IncentiveCost = 系统贡献提升 / 激励成本
```

该指标用于衡量“每花 1 元补贴带来多少系统改善”，避免模型学到一味降价。

## 5. 数据集整理

### 5.1 原始日志

需要保存每次报价决策的完整链路：

```text
用户请求日志:
  user_id_hash, request_time, soc_now, soc_target, deadline,
  origin_location, destination_location, vehicle_type

候选方案日志:
  request_id, plan_id, station_sequence, start_times, durations,
  energy_kwh, detour_km, predicted_wait, queue_delay_delta,
  grid_pressure_delta, system_contribution_score

报价日志:
  request_id, plan_id, station_prices, total_price,
  discount, model_version, policy_version, exploration_flag

用户响应日志:
  request_id, shown_plan_ids, accepted_plan_id,
  rejected, rescheduled, canceled, response_latency_sec

实际执行日志:
  actual_station, actual_start_time, actual_duration,
  actual_energy_kwh, actual_wait_min, actual_payment,
  no_show, early_leave

系统状态日志:
  station_queue, charger_utilization, grid_load,
  electricity_cost, local_congestion_level
```

### 5.2 训练样本构造

每个训练样本以一次用户请求为单位：

```json
{
  "conversations": [
    {
      "role": "system",
      "content": "你是一个个性化充电调价模型。给定用户状态、候选充电方案和滚动窗口反馈，输出每个候选方案的站点级充电单价。必须输出合法 JSON。"
    },
    {
      "role": "user",
      "content": "{...user_context, rolling_window, candidate_plans...}"
    },
    {
      "role": "assistant",
      "content": "{...pricing_strategy...}"
    }
  ]
}
```

从原始日志中可以构造三类数据。

第一类是 SFT 数据。选择历史上表现好的报价，或由启发式优化器、数值搜索器、离线仿真器生成高质量报价，作为监督答案。SFT 阶段重点是让模型学会输入字段、业务约束、JSON 格式和基础调价逻辑。

第二类是偏好数据。对同一个用户请求和候选方案集合，构造多个价格策略，并用离线评估函数打分。得分高的是 `chosen`，得分低的是 `rejected`。该数据用于 DPO 训练。

第三类是 Agentic RL 数据。保留用户状态和候选方案，训练时让 MiniMind 生成价格策略，再调用外部评估工具返回接受概率、系统影响、利润和约束违规，最后根据 reward 更新模型。

### 5.3 反事实数据增强

由于真实日志只观察到用户对已展示价格的响应，未展示价格下的响应不可直接观测。为了支撑论文实验和后训练，需要做反事实增强：

1. 对同一候选方案生成不同折扣水平，例如 `-0.05, -0.10, -0.20, -0.30` 元/kWh。
2. 使用用户响应模型估计各价格下的接受概率。
3. 使用站点仿真器评估不同选择概率下的系统影响。
4. 过滤明显不合理价格，保留可解释的反事实样本。

这一步需要在论文中明确区分“真实观测反馈”和“模型估计反馈”，避免把反事实估计误写成真实标签。

## 6. 基座模型选择

推荐从 MiniMind 已完成对话和工具调用能力的 SFT 权重开始，而不是从纯预训练权重开始。

| 基座 | 优点 | 缺点 | 用途 |
| --- | --- | --- | --- |
| MiniMind 64M Dense | 延迟低、训练便宜、适合在线服务 | 表达能力有限 | 生产原型、消融实验 |
| MiniMind 198M MoE | 容量更强，能处理更复杂候选方案 | 推理和训练成本更高 | 论文主模型或增强版本 |
| 大模型教师 | 能生成高质量调价解释和初始答案 | 成本高，不适合直接在线部署 | 数据蒸馏、SFT 伪标签 |

建议论文采用两个模型规模：

```text
MiniMind-64M: 验证小模型可行性和低延迟部署。
MiniMind-MoE: 验证更大容量下的收益上限。
```

## 7. SFT 阶段

SFT 的目标不是直接获得最优调价策略，而是让 MiniMind 学会稳定执行以下行为：

1. 理解用户状态和候选方案字段。
2. 区分系统贡献高低和用户接受难度。
3. 输出合法 JSON。
4. 遵守价格上下限、最低利润和最大折扣。
5. 给高系统贡献方案更有吸引力的价格，但不过度补贴。

推荐 SFT prompt：

```text
你是一个个性化充电调价模型。
你会收到用户状态、过去时间窗内用户响应、多个候选充电方案以及每个方案的系统影响。
你的任务是为每个候选方案生成站点级充电单价，使用户更可能选择系统贡献较高的方案，同时控制激励成本。
必须输出合法 JSON，不要输出额外文本。
```

SFT 输出 schema 应固定，并在训练和推理中统一校验。

训练建议：

```text
from_weight: full_sft
method: LoRA 或 full_sft
max_seq_len: 根据候选方案数量设置，建议 1024-2048
loss: assistant JSON token 上的交叉熵
```

如果候选方案过多，应先由外部排序器保留 Top-K，例如 K=4 或 K=8，再交给 MiniMind。

## 8. DPO 偏好后训练

DPO 用于让模型偏好高质量价格策略。每条偏好样本包含同一个输入和两个输出：

```json
{
  "prompt": "{user_context + candidate_plans}",
  "chosen": "{high_compliance_low_cost_pricing}",
  "rejected": "{over_discount_or_low_compliance_pricing}"
}
```

偏好标签可由以下评分函数生成：

```text
Score =
  0.35 * normalized_system_benefit
+ 0.25 * normalized_profit
+ 0.20 * compliance_probability
- 0.15 * normalized_incentive_cost
- 0.05 * violation_penalty
```

DPO 的优点是稳定、成本低，不需要在线 rollout。它适合作为 SFT 和 Agentic RL 之间的过渡阶段，先让模型知道哪些价格策略“相对更好”。

## 9. Agentic RL / GRPO-CISPO 后训练

### 9.1 工具调用环境

复用 MiniMind 的 Agentic RL 思路，把外部评估函数注册为工具：

```json
{
  "type": "function",
  "function": {
    "name": "evaluate_pricing_strategy",
    "description": "评估给定用户、候选充电方案和价格策略下的用户响应概率、系统影响、利润和约束违规。",
    "parameters": {
      "type": "object",
      "properties": {
        "user_context": {"type": "object"},
        "candidate_plans": {"type": "array"},
        "pricing_strategy": {"type": "object"}
      },
      "required": ["user_context", "candidate_plans", "pricing_strategy"]
    }
  }
}
```

训练 rollout：

```text
1. MiniMind 读取用户状态和候选方案。
2. MiniMind 输出价格策略。
3. 环境调用 evaluate_pricing_strategy。
4. 工具返回 expected_accept_prob、chosen_plan_prob、system_benefit、profit、incentive_cost、violations。
5. 根据 reward 计算轨迹分数。
6. 对同一输入采样多个价格策略，用 GRPO/CISPO 更新模型。
```

### 9.2 为什么用 GRPO/CISPO

PPO 需要训练 critic，显存和稳定性压力更大。GRPO 使用同一 prompt 下多个采样结果的组内相对分数估计 advantage，更适合 MiniMind 这类小模型。CISPO 可以作为 GRPO 的稳定损失变体，减少 clip 后梯度路径被截断的问题。

推荐训练顺序：

```text
MiniMind full_sft
  -> 领域 SFT
  -> DPO
  -> Agentic CISPO
```

### 9.3 RL 奖励设计

推荐 reward：

```text
R =
  1.5 * compliance_score
+ 1.0 * system_benefit
+ 0.8 * profit_score
- 1.0 * incentive_cost_score
- 0.8 * rejection_risk
- 0.7 * queue_penalty
- 0.7 * grid_penalty
- 0.5 * volatility_penalty
- 2.0 * violation_penalty
```

其中：

```text
compliance_score = P(user chooses recommended high-contribution plan)
system_benefit = normalized(queue_improvement + grid_pressure_reduction + load_balance_gain)
profit_score = normalized(expected_revenue - energy_cost - incentive_cost)
incentive_cost_score = normalized(discount_amount)
rejection_risk = max(target_min_accept - expected_accept_prob, 0)
volatility_penalty = max(abs(new_price - previous_price) - allowed_step, 0)
```

对于非法 JSON、价格越界、负利润、超过最大折扣、输出缺失字段，应直接给强负奖励。

## 10. 在线调整机制

线上不建议每个时间窗实时更新 MiniMind 权重。实时更新应放在响应预测器和校准器中，MiniMind 权重按小时级或天级周期性后训练。

在线状态更新：

```text
past_15min_accept_rate
past_30min_accept_rate
past_60min_accept_rate
plan_compliance_rate
avg_discount
avg_profit_loss
reject_reason_distribution
station_congestion_level
```

在线校准逻辑：

```text
如果拒绝率升高:
  增强该用户或该场景的价格敏感度估计，增加目标方案折扣。

如果接受率过高但利润下降:
  说明补贴过强，收缩折扣或提高目标方案价格。

如果用户接受但没有选择系统贡献最高方案:
  增大高贡献方案与低贡献方案之间的相对价格差。

如果目标站点排队恶化:
  提高该站点价格，降低替代站点价格。

如果同一用户连续拒绝:
  降低探索幅度，采用更保守价格或回退到默认策略。
```

可以将在线校准器实现为 contextual bandit：

```text
context = 用户特征 + 候选方案差异 + 时间窗反馈
action = 对 MiniMind 输出价格的偏置或折扣倍率
reward = 实际遵循率 + 系统收益 - 激励成本
```

这样 LLM 负责生成结构化价格策略，bandit 负责快速适应短期价格敏感度漂移。

## 11. 安全约束与可控性

必须在 MiniMind 输出后加入规则层：

```text
price_min <= price_per_kwh <= price_max
abs(price_new - price_last) <= max_step_change
expected_profit >= min_profit
discount <= max_discount
同一用户短期内价格差异 <= fairness_threshold
高拥堵站点不能被过度降价吸引更多流量
```

如果 MiniMind 输出无效，系统应回退到：

```text
1. 启发式调价策略
2. 历史最优模板策略
3. 无个性化的站点分时价格
```

生产中建议始终执行“多候选采样 + 外部评估重排”：

```text
MiniMind 采样 N 个价格策略
  -> schema 校验
  -> 安全约束过滤
  -> 外部评估函数打分
  -> 选择最高可行策略
```

## 12. 实验设计

### 12.1 Baseline

建议至少比较以下方法：

| 方法 | 描述 |
| --- | --- |
| Fixed Price | 固定站点价格 |
| Time-of-Use Price | 传统分时电价 |
| Rule-based Incentive | 基于排队和负载的启发式折扣 |
| Optimization-only | 外部优化器直接给方案，不做个性化激励 |
| Response-model Pricing | 只用用户响应模型求价格，不用 LLM |
| MiniMind-SFT | 只做领域 SFT |
| MiniMind-SFT-DPO | SFT + DPO |
| MiniMind-Agent-RL | SFT + DPO + Agentic GRPO/CISPO |
| MiniMind-RL-Online | 加入在线 bandit 校准 |

### 12.2 主指标

```text
用户遵循率: compliance_rate
系统贡献: average_system_benefit
单位激励收益: system_benefit_per_incentive_cost
平均利润: profit_per_session
拒绝率: reject_rate
取消率: cancel_rate
平均等待时间: avg_wait_min
P95 等待时间: p95_wait_min
电网峰值压力: peak_grid_pressure
价格波动: price_volatility
违规率: constraint_violation_rate
```

### 12.3 消融实验

建议做以下消融：

```text
去掉 rolling_window 特征
去掉用户历史敏感度特征
去掉 DPO
去掉 Agentic RL
去掉在线 bandit 校准
只输出方案总价，不输出站点级价格
不使用安全约束重排
不同候选方案数量 K
不同采样数量 N
```

### 12.4 泛化实验

```text
新站点泛化
新时间段泛化
节假日或突发高峰
极端排队拥堵
低电价低负载场景
高电价高负载场景
价格敏感用户与不敏感用户分组
```

## 13. MiniMind 仓库落地映射

当前仓库已有以下可复用能力：

| 文件 | 可复用点 |
| --- | --- |
| `dataset/lm_dataset.py` | `SFTDataset`、`DPODataset`、`AgentRLDataset` 的数据格式基础 |
| `trainer/train_full_sft.py` | 领域 SFT |
| `trainer/train_lora.py` | 低成本领域 LoRA |
| `trainer/train_dpo.py` | 偏好后训练 |
| `trainer/train_agent.py` | 多轮工具调用 + Agentic GRPO/CISPO 后训练 |
| `trainer/rollout_engine.py` | torch / sglang rollout 后端 |
| `model/tokenizer_config.json` | `<tool_call>`、`<tool_response>`、`<think>` 模板 |
| `scripts/eval_toolcall.py` | 工具调用评估脚本原型 |

建议新增或改造：

```text
dataset/ev_pricing_sft.jsonl
dataset/ev_pricing_dpo.jsonl
dataset/ev_pricing_agent_rl.jsonl

trainer/train_ev_pricing_agent.py
trainer/ev_pricing_reward.py
scripts/eval_ev_pricing.py
scripts/simulate_ev_pricing_env.py
```

其中 `train_ev_pricing_agent.py` 可以从 `train_agent.py` 派生，替换工具定义、工具执行函数和 reward 计算逻辑。

## 14. 论文式方法命名

可以将方法命名为：

```text
LLM-PIP: LLM-guided Personalized Incentive Pricing for EV Charging
```

论文贡献点可以写为：

1. 提出一个将“候选充电方案生成”和“个性化价格诱导”解耦的 EV 充电调度框架。
2. 设计基于 MiniMind 的结构化个性化价格策略模型，使小型 LLM 能输出可校验的站点级价格激励。
3. 构建融合用户响应、系统贡献、利润和激励成本的后训练 reward。
4. 引入滚动窗口价格敏感度校准，实现在线自适应调价。
5. 通过仿真和真实日志回放验证系统在遵循率、排队延误、利润和单位激励收益上的改进。

## 15. 推荐实施路线

第一阶段：离线数据整理和仿真环境。

```text
整理用户请求、候选方案、报价、响应、执行结果日志。
实现 evaluate_pricing_strategy。
实现价格 schema 校验和安全约束。
构造 SFT / DPO / Agent RL 数据。
```

第二阶段：领域 SFT。

```text
基于 full_sft 权重训练 MiniMind 输出合法个性化价格 JSON。
先用 LoRA 快速验证，再决定是否 full SFT。
用 schema pass rate、价格约束违规率、离线 reward 评估。
```

第三阶段：DPO。

```text
构造 chosen/rejected 价格策略。
训练模型偏好高系统贡献、低激励成本、低拒绝风险的输出。
比较 SFT-only 与 SFT-DPO。
```

第四阶段：Agentic GRPO/CISPO。

```text
接入 evaluate_pricing_strategy 工具。
同一输入采样多个价格策略。
用 reward 做组内相对优势更新。
重点监控 reward hacking、过度降价和格式退化。
```

第五阶段：在线校准。

```text
上线只更新响应预测器和 bandit 校准器。
MiniMind 权重周期性离线更新。
先灰度 A/B，再扩大流量。
```

## 16. 主要风险

1. 真实日志存在选择偏差，只观察到已展示价格下的用户响应。
2. 用户响应模型如果不准，RL 会优化错误 reward。
3. MiniMind 可能学会通过过度降价提高遵循率，需要强约束激励成本。
4. 小模型处理长候选方案列表能力有限，需要 Top-K 筛选和结构化压缩。
5. 在线探索可能造成短期用户体验波动，需要保守探索率和回退策略。
6. 个性化价格可能引入公平性争议，需要加入价格差异约束和可解释日志。

## 17. 参考方向

- Direct Preference Optimization: https://arxiv.org/abs/2305.18290
- DeepSeekMath / GRPO: https://arxiv.org/abs/2402.03300
- Constrained Policy Optimization: https://arxiv.org/abs/1705.10528
- Contextual dynamic pricing / bandit pricing: https://arxiv.org/abs/2109.07340

