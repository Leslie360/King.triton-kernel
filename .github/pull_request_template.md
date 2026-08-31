## 变更内容

<!-- 用一句话说明这个 PR 做了什么。 -->

## 类型

- [ ] Bug fix
- [ ] 新功能 / 新模块
- [ ] 文档 / 治理文件
- [ ] 评测口径工具
- [ ] 重构 / 性能优化
- [ ] 其他（请说明）

## 改动清单

- [ ] 改了哪些文件 / 模块（列出路径与改动要点）
- [ ] 是否新增 / 修改了对外脚本或配置

## 测试

- [ ] 运行过 `ruff check .` 与 `ruff format --check .`
- [ ] 运行过 `pytest -m "not gpu and not redis and not slow"`（纯逻辑测试）
- [ ] （如涉及 GPU 逻辑）运行过对应 GPU 测试 / 冒烟
- [ ] 测试结果摘要：___

## 评测口径纪律（若改动涉及任何数字/指标）

> 本项目铁律：**口径不可混用**。任何对外数字必须标注完整口径, 否则视为不诚实。
> 请在 PR 描述中明确说明你引用的数字属于哪种口径:

- 指标: sample_solve_rate / correctness / fast@1 / fast@1.2 / 其他
- 采样方式: best-of N / best-turn / greedy
- 轮次范围: 全部修复轮次 / 最终轮
- speedup 阈值: ≥1.0x / ≥1.2x / 其他
- reference_cache: ON / OFF
- 口径工具脚本 SHA: ___

## 检查项

- [ ] 未引入内部路径 / 私有数据 / checkpoint（发布清扫纪律, 见 docs/RELEASE_CLEANLIST.md）
- [ ] 未混用评测口径
- [ ] 文档已同步（如涉及 README / docs）

## 相关 Issue / PR

<!-- 关联的 issue 或 PR 编号。 -->
