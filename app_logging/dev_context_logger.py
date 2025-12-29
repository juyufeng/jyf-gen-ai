import os
import json
import time
import uuid

SENSITIVE_KEYS = ("key", "token", "secret", "password", "apikey", "api_key")

class DevContextLogger:
    def __init__(self, base_dir: str, module: str = "default", session_label: str = "", strict_redaction: bool = True):
        self.base_dir = base_dir if isinstance(base_dir, str) and base_dir else "./logs/dev"
        self.module = module or "default"
        self.session_label = session_label or ""
        self.strict_redaction = strict_redaction
        run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:8]
        self.run_dir = os.path.join(self.base_dir, run_id)
        os.makedirs(self.run_dir, exist_ok=True)
        self._paths = {
            "changes": os.path.join(self.run_dir, "changes.jsonl"),
            "context": os.path.join(self.run_dir, "context.json"),
            "ai": os.path.join(self.run_dir, "ai_interactions.jsonl"),
            "debug": os.path.join(self.run_dir, "debug.jsonl"),
            "index": os.path.join(self.run_dir, "index.json"),
        }
        self._index = {
            "module": self.module,
            "session_label": self.session_label,
            "start_ts": time.time(),
            "files": self._paths,
            "counts": {"changes": 0, "ai": 0, "debug": 0},
        }
        self._write_file(self._paths["index"], self._index)

    def _redact(self, obj):
        if not self.strict_redaction:
            return obj
        if isinstance(obj, dict):
            ret = {}
            for k, v in obj.items():
                if any(s in str(k).lower() for s in SENSITIVE_KEYS):
                    ret[k] = "***"
                else:
                    ret[k] = self._redact(v)
            return ret
        if isinstance(obj, list):
            return [self._redact(x) for x in obj]
        return obj

    def _append_jsonl(self, path: str, data: dict):
        data = dict(data or {})
        data["ts"] = time.time()
        data["ts_iso"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(data["ts"]))
        data = self._redact(data)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False) + "\n")

    def _write_file(self, path: str, data):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(data, ensure_ascii=False, indent=2))

    def log_change(self, file_path: str, change_type: str, diff: str, intent: str, expected_effect: str, ref_links: list = None):
        self._append_jsonl(self._paths["changes"], {
            "module": self.module,
            "file_path": file_path,
            "change_type": change_type,
            "diff": diff,
            "intent": intent,
            "expected_effect": expected_effect,
            "ref_links": ref_links or []
        })
        self._index["counts"]["changes"] += 1
        self._write_file(self._paths["index"], self._index)

    def log_context(self, module_desc: str, business_logic: str, architecture_notes: str, external_services: list):
        payload = {
            "module": self.module,
            "module_desc": module_desc,
            "business_logic": business_logic,
            "architecture_notes": architecture_notes,
            "external_services": external_services or []
        }
        payload = self._redact(payload)
        self._write_file(self._paths["context"], payload)

    def log_ai_interaction(self, question: str, suggestion: str, decision: str, rationale: str, related_change_ids: list = None):
        self._append_jsonl(self._paths["ai"], {
            "module": self.module,
            "question": question,
            "suggestion": suggestion,
            "decision": decision,
            "rationale": rationale,
            "related_change_ids": related_change_ids or []
        })
        self._index["counts"]["ai"] += 1
        self._write_file(self._paths["index"], self._index)

    def log_debug(self, error: str, exception: str, stack: str, attempts: list, env: dict):
        self._append_jsonl(self._paths["debug"], {
            "module": self.module,
            "error": error,
            "exception": exception,
            "stack": stack,
            "attempts": attempts or [],
            "env": self._redact(env or {})
        })
        self._index["counts"]["debug"] += 1
        self._write_file(self._paths["index"], self._index)
