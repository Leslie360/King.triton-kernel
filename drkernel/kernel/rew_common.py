"""rew_common.py — 训练/eval 共享的代码提取逻辑 (K3 L4: 双路径统一, 防"改错引擎"复发)"""

import re


def extract_code_robust(text: str) -> str:
    """从文本提取最终 Python 代码:
    ①取**最后一个**代码块(最终答案, 非思考/工具调用的块)
    ②**剥离开头思考散文**(从第一个 import/from/class/def 开始)——模型常把
      "Thinking, analysis..." 写进 ```python 块, 导致语法错(实测修 41% syntax 错)。
    兼容 ```python / ``` / ```<lang> 标记。
    """
    if not text:
        return ""
    blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    code = blocks[-1].strip() if blocks else text
    # 剥离开头散文: 从第一个 import/from/class/def 开始
    m = re.search(r"(?:^|\n)((?:import |from |class |def ))", code)
    if m:
        code = code[m.start():]
    return code.strip()
