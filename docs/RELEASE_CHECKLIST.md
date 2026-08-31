# 发布前核对清单（RELEASE_CHECKLIST）

> 合并自 `docs/RELEASE_CLEANLIST.md`（K3 §25 硬条件）。**原则：发布物 = 干净导出，不是 push 工作仓；git 历史里同样有秘密。**
> 对外发布前逐项签字，任一未过不发布。本文件面向 MIT 开源发布（GitHub 公开仓）。

---

## 一、发布前必读

- [x] 先读 `docs/ARCHITECTURE.md`（分层边界：什么随仓发布、什么依赖 verl 不发布）
- [x] 先读 `docs/CALIBER_TABLE.md`（对外数字口径唯一来源）
- [ ] **发布物 = 干净导出**：从工作仓 `git clone` 全新副本（不带 .git 历史）→ 或 `git archive`，**在新副本上**执行清扫 + grep

---

## 二、秘密清扫（逐项检查 + 签字）

> 发布前必须确认公开仓中不包含任何内部信息。

| # | 清扫项 | 检查方式 | 状态 |
|---|---|---|---|
| 1 | `sk-*` API key | `grep -rn "sk-" 全仓` | ☐ |
| 2 | 内部 pod IP（`10.199.x.x` / `fdbd:` IPv6） | `grep -rnE "10\.199\.|fdbd:"` | ☐ |
| 3 | `data/ISOLATED_LEAK/` 泄漏数据 | `ls data/ISOLATED_LEAK/` | ☐ |
| 4 | 内部纪要 docs（`K3_*` / `STATUS` / `HANDOVER` / 运维流水） | 排除或降级为私有 | ☐ |
| 5 | verl_patch 私有路径（`/mnt/hdfs` `/mnt/bn` `/mnt/pfs`） | `grep -rnE "/mnt/hdfs|/mnt/bn|/mnt/pfs"` | ☐ |
| 6 | 硬编码 server_url / eval server IP（yaml / 脚本） | `grep -rn "server_url\|REWARD_SERVER_URL\|EVAL_IP"` | ☐ |
| 7 | `qwen3_tokenizer.py` 内嵌内部 IP / debug dump | 删除 debug dump | ☐ |
| 8 | `settings.py SECRET_KEY` | 改强制 env | ☐ |
| 9 | 根 `.env`（redis 密码 / 内部 IP / REDIS_PORT） | 确认排除（不入 git） | ☐ |
| 10 | 软链替换为独立副本 | `kernelgym/` `drkernel/` 为实文件（非软链） | ☐ |
| 11 | 中英混杂运维注释（A2/A4/L6 等内部编号） | 清理或英文化 | ☐ |
| 12 | 训练 checkpoint / 内部数据集 / 日志 | 排除（不入仓） | ☐ |
| 13 | 私有 git 历史 / 内部分支 | 用 `git archive` 或新 `git init`，不带历史 | ☐ |

### 全仓 grep 复核（在新副本上执行）

```bash
grep -rnE "sk-[a-zA-Z0-9]{16,}|10\.199\.|fdbd:|/mnt/hdfs|/mnt/bn|/mnt/pfs|/opt/tiger|ISOLATED_LEAK" .
grep -rn "REWARD_SERVER_URL\|REDIS_PASSWORD\|server_url\|EVAL_IP" .
```

逐项签字（填表 `[x]` + 日期 + 执行人）。

---

## 三、数字口径（发布红线）

> **对外计时数字一律等 v2 口径（refcache=ON）重测后发布。** 39% (refcache=OFF) 与 v2 ~20-29% 并存 = 自相矛盾，禁止。

### 3.1 标物理机 / refcache 状态

每个对外数字必须附四件套：

| 属性 | 说明 |
|---|---|
| **物理机** | 哪台（如 8×A800 开发机）；跨机器数字不可直接比较 |
| **refcache 状态** | `reference_cache=ON/OFF`；**发布数字必须 refcache=ON**（v2 口径） |
| **agg_eval.py SHA** | 权威聚合脚本的 git SHA（`evals/agg_eval.py`） |
| **口径版本** | best-of-history / best-turn / 预算（采样×轮次） |

### 3.2 来源行（比较基准必须标注）

| 来源 | 数字 | 必须标注 |
|---|---|---|
| Dr.Kernel | 47.8%（L2 fast@1.2） | **best-turn（STTS†），预算未披露** |
| daVinci | 27.1%（L2 Fast@1.2） | best-turn |
| daVinci | 70.6%（L2 Fast1） | **Fast@1（≥1.0x）≠ Fast@1.2，勿混用** |
| 我们 | TBD（v2 重测中） | refcache=ON + agg_eval SHA |

> **口径不可混用**：fast@1 vs fast@1.2、best-of vs best-turn、≥1.0x vs ≥1.2x、单次 vs best-of 必须各标定义（详见 `docs/eval_calibre.md`、`docs/CALIBER_TABLE.md`）。

### 3.3 4 数报告标准

对外统一报：**sample_solve_rate / correctness / fast@1 / fast@1.2** + 标注 best-of 预算。不得只报 pass@1。

---

## 四、License / 合规

- [ ] `LICENSE` = MIT（已就位）
- [ ] 第三方代码版权头保留（drkernel 内 verl 相关 = Apache-2.0，见根 `NOTICE` 隔离声明）
- [ ] 依赖 `pyproject.toml` 中列出（`train` extra 含 verl/ray/vllm，标注不随仓发布）
- [ ] 无专利/商业秘密泄露

### 占位符替换（发布前必须）

| # | 文件 | 占位符 | 发布前填什么 |
|---|---|---|---|
| 1 | `SECURITY.md` | `[INSERT_SECURITY_EMAIL]` / `[INSERT_KEY_URL]` | 维护者安全联系邮箱 / GPG 公钥地址 |
| 2 | `CODE_OF_CONDUCT.md` | `[INSERT_CONTACT_EMAIL]` | 项目团队举报邮箱 |
| 3 | `CHANGELOG.md` | `<owner>` | GitHub org/repo 所有者 |

> ⚠️ 占位符未替换 = 仓库未就绪，禁止公开发布。

---

## 五、导出流程

1. 从工作仓 `git clone` 全新副本（不带 .git 历史）→ 或 `git archive`
2. 在**新副本**上执行二、秘密清扫 + grep
3. 逐项签字（填表 `[x]` + 日期 + 执行人）
4. 在**新副本**上跑 `bash setup.sh` + `python3 smoke_test.py http://<eval-server>:8004` 验证可运行
5. 确认数字均来自 v2 口径（refcache=ON）+ agg_eval SHA
6. 发布

---

## 六、签字记录

| 项 | 执行人 | 日期 | 签字 |
|---|---|---|---|
| 二、秘密清扫 | | | |
| 三、数字口径 | | | |
| 四、License/合规 | | | |
| 五、导出+冒烟 | | | |
