"""从可能带解释文字的模型输出中提取首个完整 JSON 对象。"""

import json


def extract_json_object(text: str) -> str:
    """扫描花括号，并忽略字符串内部及转义后的括号。"""
    start = text.find("{")
    if start < 0:
        raise ValueError("输出中没有 JSON 对象起始符 {")
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                candidate = text[start:index + 1]
                json.loads(candidate)
                return candidate
    raise ValueError("JSON 对象没有闭合")
