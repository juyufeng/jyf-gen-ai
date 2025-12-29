import unittest
from models.selector import select_model

class TestSelector(unittest.TestCase):
    def test_code(self):
        m = select_model("修复 Python 报错并添加单元测试", "beijing", allow_vision=True, allow_code=True)
        self.assertEqual(m, "qwen3-coder-plus")

    def test_vision(self):
        m = select_model("解析视频并识别按钮坐标", "beijing", allow_vision=True, allow_code=True)
        self.assertEqual(m, "qwen-vl-max")

    def test_text(self):
        m = select_model("写一段中文摘要", "beijing", allow_vision=True, allow_code=True)
        self.assertEqual(m, "qwen3-max")
