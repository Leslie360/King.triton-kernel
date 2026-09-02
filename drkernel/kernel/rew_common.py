"""rew_common.py — shared code-extraction logic for training and eval (K3 L4: unify
the two paths to prevent "wrong-engine modification" regression)."""

import re


def extract_code_robust(text: str) -> str:
    """Extract the final Python code from text:
    1) Take the **last** code block (the final answer, not the thinking/tool-call blocks)
    2) Strip the leading thinking prose (start from the first import/from/class/def) —
       the model often writes "Thinking, analysis..." inside a ```python block, which
       causes syntax errors (in practice, this fixes 41% of syntax errors).
    Compatible with ```python / ``` / ```<lang> markers.
    """
    if not text:
        return ""
    blocks = re.findall(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    code = blocks[-1].strip() if blocks else text
    # Strip the leading prose: start from the first import/from/class/def
    m = re.search(r"(?:^|\n)((?:import |from |class |def ))", code)
    if m:
        code = code[m.start():]
    return code.strip()
