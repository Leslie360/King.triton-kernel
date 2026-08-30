# 发布清扫清单（K3 §25 硬条件①, 2026-08-30）

> **原则**: 发布物 = **干净导出**, 不是 push 工作仓。git 历史里同样有秘密。
> 导出后**全仓 grep 复核, 清单逐项签字**, 任一未过不发布。

## 一、清扫项（逐项检查 + 签字）

| # | 清扫项 | 检查方式 | 状态 |
|---|---|---|---|
| 1 | `sk-*` API key | grep -rn "sk-" 全仓 | ☐ |
| 2 | 内部 pod IP (10.199.x.x / fdbd: IPv6) | grep -rnE "10\.199\.|fdbd:" | ☐ |
| 3 | `data/ISOLATED_LEAK/` 泄漏数据 | ls data/ISOLATED_LEAK/ | ☐ |
| 4 | docs 内部纪要 (K3_* / STATUS / HANDOVER) | 排除或降级 | ☐ |
| 5 | verl_patch 私有路径 (/mnt/hdfs /mnt/bn) | grep -rn "/mnt/hdfs\|/mnt/bn" | ☐ |
| 6 | 硬编码 server_url (kernel_trainer.yaml:16 / kernel_grading.yaml:284) | 改可配置占位符 | ☐ |
| 7 | qwen3_tokenizer.py 内嵌内部 IP/路径 | 删除 debug dump | ☐ |
| 8 | settings.py SECRET_KEY | 改强制 env | ☐ |
| 9 | 根 .env (redis 密码/内部 IP) | 确认排除 | ☐ |
| 10 | 软链替换为独立副本 | kernelgym/ drkernel/ 实文件 | ☐ |
| 11 | 中英混杂运维注释 (A2/A4/L6) | 清理或英文化 | ☐ |

## 二、导出流程

1. 从工作仓 `git clone` 全新副本（不带 .git 历史）→ 或 `git archive`
2. 在**新副本**上执行上表清扫 + grep
3. 全仓 grep 复核:
```bash
grep -rnE "sk-[a-zA-Z0-9]{16,}|10\.199\.|fdbd:|/mnt/hdfs|/mnt/bn|/opt/tiger|ISOLATED_LEAK" .
grep -rn "REWARD_SERVER_URL\|REDIS_PASSWORD\|server_url" .
```
4. 逐项签字（填表 `[x]` + 日期 + 执行人）

## 三、发布红线（K3 §25 硬条件②③）

- **对外计时数字一律等 v2 口径（refcache=ON）重测后发布**。39% (refcache=OFF) 与 v2 ~20-29% 并存=自相矛盾, 禁止。
- **Dr.Kernel 47.8% = best-turn (STTS†)**, 必须标注。
- **daVinci 27.1% = L2 Fast@1.2, 70.6% = L2 Fast@1, 两口径各标定义不许混用**。
- 每个数字附: 口径 / 物理机 / refcache 状态 / agg_eval.py SHA（CALIBER_TABLE §八）。
