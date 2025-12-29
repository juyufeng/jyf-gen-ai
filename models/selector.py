import json
import os
import re

def load_models(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

from typing import Optional, Dict, List

def pick(models: List[Dict], region: str) -> Optional[Dict]:
    ms = [m for m in models if region in m.get("regions", [])]
    if not ms and models:
        ms = models
    if not ms:
        return None
    return sorted(ms, key=lambda m: m.get("priority", 0), reverse=True)[0]

def select_model(query: str, region: str, allow_vision: bool = True, allow_code: bool = True, force_model: Optional[str] = None) -> str:
    if force_model:
        return force_model
    data = load_models(os.path.join(os.path.dirname(__file__), "alibl_models.json"))
    q = (query or "").lower()
    code_hit = bool(re.search(r"(class |def |function|异常|Exception|Traceback|API|HTTP|SQL|Java|Python|TypeScript|错误|报错|编译|单元测试|断言|diff|文件路径|import )", q))
    vision_hit = bool(re.search(r"(视频|短视频|播放|帧率|剪辑|识别|OCR|图像|图片|截图|坐标|视觉|相片)", q))
    if allow_code and code_hit:
        m = pick(data.get("code", []), region)
        if m:
            return m["id"]
    if allow_vision and vision_hit:
        m = pick(data.get("vision", []), region)
        if m:
            return m["id"]
    m = pick(data.get("text", []), region)
    if m:
        return m["id"]
    return "qwen3-max"
