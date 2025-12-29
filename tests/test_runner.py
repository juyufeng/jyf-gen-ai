import unittest
from sandbox.runner import run

class DummyAPI:
    def __init__(self):
        self.actions = []
    def navigate(self, url): self.actions.append(("navigate", url))
    def click_at(self, x, y): self.actions.append(("click_at", x, y))

class TestRunner(unittest.TestCase):
    def test_run_ok(self):
        api = {
            "navigate": lambda url: None,
            "click_at": lambda x, y: None,
            "hover_at": lambda x, y: None,
            "type_text_at": lambda x, y, text: None,
            "scroll_document": lambda d: None,
            "scroll_at": lambda x, y, d, m=800: None,
            "wait_5_seconds": lambda: None,
            "go_back": lambda: None,
            "go_forward": lambda: None,
            "key_combination": lambda keys: None,
            "drag_and_drop": lambda x, y, dx, dy: None,
        }
        run("navigate('https://example.com')\\nclick_at(500,300)", api)

    def test_run_block_import(self):
        api = {"navigate": lambda url: None}
        with self.assertRaises(Exception):
            run("import os\\nnavigate('https://example.com')", api)
