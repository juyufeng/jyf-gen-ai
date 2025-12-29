import os
import json
import time
import uuid

class ContextLogger:
    def __init__(self, base_dir: str):
        self.base_dir = base_dir if isinstance(base_dir, str) and base_dir else "./logs/context"
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        self._counter = 0

        os.makedirs(self.base_dir, exist_ok=True)

    def save_screenshot(self, screenshot_bytes: bytes) -> str:
        self._counter += 1
        path = os.path.join(self.run_dir, f"screenshot_{self._counter:04d}.png")
        with open(path, "wb") as f:
            f.write(screenshot_bytes)
        return path

    def save_text(self, content: str, suffix: str = "txt") -> str:
        self._counter += 1
        path = os.path.join(self.run_dir, f"text_{self._counter:04d}.{suffix}")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content or "")
        return path

    def write_event(self, event: dict) -> None:
        event = dict(event or {})
        event["ts"] = time.time()
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
