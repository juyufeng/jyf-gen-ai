import os
import time
import base64
import json
import hashlib
import sys
import platform
import subprocess
from typing import Literal, Optional, Union, Any, List, Dict
from openai import OpenAI
from app_logging.context_logger import ContextLogger
from app_logging.dev_context_logger import DevContextLogger
from models.selector import select_model
from sandbox.runner import run as run_sandbox
import termcolor
from rich.console import Console
from rich.table import Table

from computers import EnvState, Computer

# Schema for Qwen/OpenAI tools
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_web_browser",
            "description": "Opens a web browser. Should be called at the start."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click_at",
            "description": "Clicks the mouse at the specified coordinates (0-1000).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The x coordinate (0-1000)."},
                    "y": {"type": "integer", "description": "The y coordinate (0-1000)."}
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "hover_at",
            "description": "Hovers the mouse at the specified coordinates (0-1000).",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The x coordinate (0-1000)."},
                    "y": {"type": "integer", "description": "The y coordinate (0-1000)."}
                },
                "required": ["x", "y"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_text_at",
            "description": "Types text at the specified coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The x coordinate (0-1000)."},
                    "y": {"type": "integer", "description": "The y coordinate (0-1000)."},
                    "text": {"type": "string", "description": "The text to type."},
                    "press_enter": {"type": "boolean", "description": "Whether to press Enter after typing."},
                    "clear_before_typing": {"type": "boolean", "description": "Whether to clear the field before typing."}
                },
                "required": ["x", "y", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_document",
            "description": "Scrolls the document.",
            "parameters": {
                "type": "object",
                "properties": {
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "The direction to scroll."}
                },
                "required": ["direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_for_selector",
            "description": "Wait for a selector to appear.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "timeout_ms": {"type": "integer"}
                },
                "required": ["selector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_into_view",
            "description": "Scroll element into view by selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"}
                },
                "required": ["selector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "focus_selector",
            "description": "Focus element by selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"}
                },
                "required": ["selector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "click_selector",
            "description": "Click element by selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"}
                },
                "required": ["selector"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "type_selector",
            "description": "Type text into element by selector.",
            "parameters": {
                "type": "object",
                "properties": {
                    "selector": {"type": "string"},
                    "text": {"type": "string"},
                    "press_enter": {"type": "boolean"},
                    "clear_before_typing": {"type": "boolean"}
                },
                "required": ["selector", "text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "scroll_at",
            "description": "Scrolls at the specified coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "The x coordinate (0-1000)."},
                    "y": {"type": "integer", "description": "The y coordinate (0-1000)."},
                    "direction": {"type": "string", "enum": ["up", "down", "left", "right"], "description": "The direction to scroll."},
                    "magnitude": {"type": "integer", "description": "The amount to scroll."}
                },
                "required": ["x", "y", "direction"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "wait_5_seconds",
            "description": "Waits for 5 seconds."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_back",
            "description": "Navigates back in history."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "go_forward",
            "description": "Navigates forward in history."
        }
    },
    {
        "type": "function",
        "function": {
            "name": "navigate",
            "description": "Navigates to a specific URL (e.g., 'https://www.baidu.com').",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The URL to navigate to."}
                },
                "required": ["url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "key_combination",
            "description": "Presses a combination of keys.",
            "parameters": {
                "type": "object",
                "properties": {
                    "keys": {"type": "string", "description": "The keys to press, separated by '+' (e.g., 'Control+c')."}
                },
                "required": ["keys"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_script",
            "description": "Executes constrained Python code to control the browser via safe APIs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string", "description": "Python code using allowed APIs like navigate(), click_at(x,y), type_text_at(x,y,text)."}
                },
                "required": ["code"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "drag_and_drop",
            "description": "Drags and drops from one location to another.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer", "description": "Start x coordinate (0-1000)."},
                    "y": {"type": "integer", "description": "Start y coordinate (0-1000)."},
                    "destination_x": {"type": "integer", "description": "Destination x coordinate (0-1000)."},
                    "destination_y": {"type": "integer", "description": "Destination y coordinate (0-1000)."}
                },
                "required": ["x", "y", "destination_x", "destination_y"]
            }
        }
    }
]

console = Console()

import re

class QwenAgent:
    def __init__(
        self,
        browser_computer: Computer,
        query: str,
        model_name: str = "qwen-vl-max",
        verbose: bool = True,
        logger: Optional[ContextLogger] = None,
        region: str = "beijing",
        force_model: Optional[str] = None,
        dev_logger: Optional[DevContextLogger] = None,
    ):
        self._browser_computer = browser_computer
        self._query = query
        self._model_name = model_name
        self._verbose = verbose
        self._logger = logger
        self._region = region
        self._force_model = force_model
        self._dev_logger = dev_logger
        
        # Initialize OpenAI client for DashScope
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
            
        base_url = "https://dashscope.aliyuncs.com/compatible-mode/v1" if self._region == "beijing" else "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
        chosen = select_model(self._query, self._region, allow_vision=True, allow_code=True, force_model=self._force_model) or self._model_name
        self._model_name = chosen
        self._client = OpenAI(api_key=api_key, base_url=base_url, timeout=15)
        self._code_client = OpenAI(api_key=api_key, base_url=base_url, timeout=15)
        self._code_model = "qwen3-coder-plus"
        self._code_model_used = False
        self._last_screenshot_hash = ""
        self._last_dom_digest = ""
        self._failed_attempts = 0
        self._export_done = False
        
        self._consecutive_text_responses = 0

        self._messages = [
            {
                "role": "system",
                "content": """You are a browser automation agent. Your goal is to complete the user's request by directly controlling the browser.
1. You MUST use the provided tools to interact with the browser. Do NOT just describe what you are going to do. Just DO it.
2. The screen coordinates are normalized to 0-1000 range (top-left is 0,0).
3. If the browser is not open, you MUST call 'open_web_browser' first.
4. You will receive screenshots of the browser state after each action. Use them to locate elements for the next action.
5. You may also receive a list of "Detected Interactive Elements" with selectors, labels and coordinates. PREFER using SELECTOR-based tools over coordinates.
6. Only when the task is fully complete should you respond with a final text answer.
7. CRITICAL: Continually assess if the current page contains the answer to the user's request. If you see the answer or have completed the task, STOP immediately.
   - Do NOT keep clicking or scrolling just for the sake of it.
   - If the task is "Search for X", and you see X in the search results, the task is DONE.
   - When done, output a final text response starting with "TASK_COMPLETED:" followed by a brief summary.
8. If you need to search, navigate, click, or type, use the corresponding tools immediately. PRIORITIZE: wait_for_selector, scroll_into_view, focus_selector, click_selector, type_selector.
9. AFTER each action, you MUST validate change: check visible text or DOM summary change; if no change, try an alternative selector or scroll into view.
9. IMPORTANT: Do NOT output the function calls as text in your response (e.g., do not write 'click_at(100, 200)' in the chat). You MUST use the native 'tool_calls' field to execute functions.

Available Tools: open_web_browser, click_at, hover_at, type_text_at, scroll_document, scroll_at, navigate, go_back, go_forward, wait_5_seconds, key_combination, drag_and_drop.
Selector Tools: wait_for_selector, scroll_into_view, focus_selector, click_selector, type_selector.
Do NOT invent tools like 'search_on_baidu'. Use 'navigate' to go to URLs.

IMPORTANT: When you perform an action (like click or type), you MUST wait for the page to update. Do NOT repeat the exact same action multiple times in a row without seeing a change. If an action fails or has no effect, try a slightly different coordinate or a different approach.
Preference: If the user query mentions a specific brand/site (e.g., gpt/openai/kimi), navigate directly to that official site first rather than general search.
"""
            },
            {
                "role": "user",
                "content": self._query + " (Remember to use tools directly!)"
            }
        ]

    def denormalize_x(self, x: int) -> int:
        return int(x / 1000 * self._browser_computer.screen_size()[0])

    def denormalize_y(self, y: int) -> int:
        return int(y / 1000 * self._browser_computer.screen_size()[1])
        
    def _bytes_hash(self, b: bytes) -> str:
        try:
            return hashlib.sha256(b or b"").hexdigest()
        except Exception:
            return ""

    def _dom_hash(self, s: Optional[str]) -> str:
        try:
            if not s:
                return ""
            return hashlib.sha256(s.encode("utf-8", "ignore")).hexdigest()
        except Exception:
            return ""

    def _pick_input_selector(self, interactables: list[dict]) -> Optional[str]:
        if not interactables:
            return None
        def score(it):
            sc = 0
            if it.get("contenteditable"):
                sc += 10
            ph = (it.get("placeholder") or "").lower()
            ar = (it.get("aria") or "").lower()
            lbl = (it.get("label") or "").lower()
            for kw in ["输入", "提问", "搜索", "问", "消息", "chat", "ask", "search"]:
                if kw in ph or kw in ar or kw in lbl:
                    sc += 5
            tag = it.get("tag") or ""
            if tag in ["input", "textarea"]:
                sc += 3
            return sc
        best = sorted(interactables, key=score, reverse=True)
        for it in best:
            sel = it.get("selector")
            if sel:
                return sel
        return None

    def _should_use_code_model(self) -> bool:
        q = (self._query or "").lower()
        for kw in ["咨询", "搜索", "输入", "问答", "聊天", "站内", "提问"]:
            if kw in q:
                return True
        return False
    def _prune_history(self):
        # Keep only recent screenshots to avoid token limits
        MAX_SCREENSHOTS = 2
        screenshots_found = 0
        
        # Iterate backwards
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") == "user" and isinstance(msg.get("content"), list):
                has_image = False
                for part in msg["content"]:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        has_image = True
                        break
                
                if has_image:
                    screenshots_found += 1
                    if screenshots_found > MAX_SCREENSHOTS:
                        # Remove image data to save tokens
                        new_content = []
                        for part in msg["content"]:
                            if isinstance(part, dict) and part.get("type") == "text":
                                new_content.append(part)
                        msg["content"] = new_content

    def _get_last_observed_url(self) -> Optional[str]:
        for i in range(len(self._messages) - 1, -1, -1):
            msg = self._messages[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content")
            if not isinstance(content, str):
                continue
            try:
                payload = json.loads(content)
            except Exception:
                continue
            url = payload.get("url")
            if isinstance(url, str) and url:
                return url
        return None

    def handle_tool_call(self, tool_call) -> dict:
        function_name = tool_call.function.name
        try:
            args = json.loads(tool_call.function.arguments)
        except json.JSONDecodeError:
            return {"error": "Invalid JSON arguments"}
        
        if self._verbose:
            termcolor.cprint(f"Executing {function_name} with args: {args}", "cyan")

        result_state = None
        
        try:
            if function_name == "open_web_browser":
                result_state = self._browser_computer.open_web_browser()
            elif function_name == "click_at":
                result_state = self._browser_computer.click_at(
                    x=self.denormalize_x(args["x"]),
                    y=self.denormalize_y(args["y"])
                )
            elif function_name == "hover_at":
                result_state = self._browser_computer.hover_at(
                    x=self.denormalize_x(args["x"]),
                    y=self.denormalize_y(args["y"])
                )
            elif function_name == "type_text_at":
                result_state = self._browser_computer.type_text_at(
                    x=self.denormalize_x(args["x"]),
                    y=self.denormalize_y(args["y"]),
                    text=args["text"],
                    press_enter=args.get("press_enter", False),
                    clear_before_typing=args.get("clear_before_typing", True)
                )
                try:
                    sel = self._pick_input_selector(getattr(result_state, "interactables", []))
                    if sel:
                        result_state = self._browser_computer.type_selector(
                            selector=sel,
                            text=args["text"],
                            press_enter=True,
                            clear_before_typing=True
                        )
                except Exception:
                    pass
            elif function_name == "scroll_document":
                result_state = self._browser_computer.scroll_document(args["direction"])
            elif function_name == "wait_for_selector":
                result_state = self._browser_computer.wait_for_selector(
                    selector=args["selector"],
                    timeout_ms=args.get("timeout_ms", 5000)
                )
            elif function_name == "scroll_into_view":
                result_state = self._browser_computer.scroll_into_view(args["selector"])
            elif function_name == "focus_selector":
                result_state = self._browser_computer.focus_selector(args["selector"])
            elif function_name == "click_selector":
                result_state = self._browser_computer.click_selector(args["selector"])
            elif function_name == "type_selector":
                result_state = self._browser_computer.type_selector(
                    selector=args["selector"],
                    text=args["text"],
                    press_enter=args.get("press_enter", True),
                    clear_before_typing=args.get("clear_before_typing", True)
                )
            elif function_name == "scroll_at":
                magnitude = args.get("magnitude", 800)
                direction = args["direction"]
                if direction in ("up", "down"):
                    magnitude = self.denormalize_y(magnitude)
                elif direction in ("left", "right"):
                    magnitude = self.denormalize_x(magnitude)
                
                result_state = self._browser_computer.scroll_at(
                    x=self.denormalize_x(args["x"]),
                    y=self.denormalize_y(args["y"]),
                    direction=direction,
                    magnitude=magnitude
                )
            elif function_name == "wait_5_seconds":
                result_state = self._browser_computer.wait_5_seconds()
            elif function_name == "go_back":
                result_state = self._browser_computer.go_back()
            elif function_name == "go_forward":
                result_state = self._browser_computer.go_forward()
            elif function_name == "search":
                result_state = self._browser_computer.search()
            elif function_name == "navigate":
                result_state = self._browser_computer.navigate(args["url"])
            elif function_name == "key_combination":
                result_state = self._browser_computer.key_combination(args["keys"].split("+"))
            elif function_name == "drag_and_drop":
                result_state = self._browser_computer.drag_and_drop(
                    x=self.denormalize_x(args["x"]),
                    y=self.denormalize_y(args["y"]),
                    destination_x=self.denormalize_x(args["destination_x"]),
                    destination_y=self.denormalize_y(args["destination_y"])
                )
            elif function_name == "run_script":
                code = args.get("code", "")
                self._run_script(code)
                result_state = self._browser_computer.current_state()
            else:
                return {"error": f"Unknown function {function_name}"}
        except Exception as e:
            return {"error": str(e)}

        # Convert result state to tool output format
        if isinstance(result_state, EnvState):
            # Include interactable elements in the response to help the model locate elements
            # Filter interactables to keep the prompt size manageable, or just pass a summary
            interactables_text = ""
            if hasattr(result_state, 'interactables') and result_state.interactables:
                elements = []
                for el in result_state.interactables[:50]:
                    label = el.get('label') or el.get('accessible_name') or ''
                    selector = el.get('selector') or ''
                    x = el.get('x'); y = el.get('y')
                    if x is not None and y is not None:
                        norm_x = int(x / self._browser_computer.screen_size()[0] * 1000)
                        norm_y = int(y / self._browser_computer.screen_size()[1] * 1000)
                        if label or selector:
                            elements.append(f"{label} [{selector}] at ({norm_x}, {norm_y})")
                    else:
                        if label or selector:
                            elements.append(f"{label} [{selector}]")
                if elements:
                    interactables_text = "\nDetected Interactive Elements:\n" + "\n".join(elements)

            try:
                prev_sh = self._last_screenshot_hash
                prev_dom = self._last_dom_digest
                new_sh = self._bytes_hash(result_state.screenshot)
                new_dom = self._dom_hash(getattr(result_state, "dom", None))
                self._last_screenshot_hash = new_sh
                self._last_dom_digest = new_dom
                if new_sh == prev_sh and new_dom == prev_dom:
                    self._failed_attempts += 1
                else:
                    self._failed_attempts = 0
                if self._failed_attempts >= 2 and not self._code_model_used and self._should_use_code_model():
                    self._code_model_used = True
            except Exception:
                pass
            # Do NOT append any non-tool messages here to avoid violating tool_call ordering
            screenshot_path = None
            dom_path = None
            if self._logger:
                try:
                    screenshot_path = self._logger.save_screenshot(result_state.screenshot)
                    if getattr(result_state, "dom", None):
                        dom_path = self._logger.save_text(result_state.dom, "html")
                    self._logger.write_event({
                        "tool": function_name,
                        "args": args,
                        "url": result_state.url,
                        "screenshot_path": screenshot_path,
                        "dom_path": dom_path
                    })
                except Exception:
                    pass
            return {
                "url": result_state.url,
                "status": "success",
                "interactables": interactables_text, # Add this field
                "screenshot_base64": base64.b64encode(result_state.screenshot).decode('utf-8')
            }
        return {"status": "success"}

    def _write_text_file_local(self, name: str, content: str) -> str:
        try:
            base_dir = os.path.join(os.getcwd(), "exports")
            os.makedirs(base_dir, exist_ok=True)
            fname = os.path.basename(name or "output.txt")
            target = os.path.join(base_dir, fname)
            with open(target, "w", encoding="utf-8") as f:
                f.write(content or "")
            return target
        except Exception:
            return ""

    def _respond_to_pending_tool_calls(self) -> bool:
        try:
            for idx in range(len(self._messages) - 1, -1, -1):
                msg = self._messages[idx]
                if not isinstance(msg, dict):
                    continue
                if msg.get("role") != "assistant":
                    continue
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    continue
                for tc in tool_calls:
                    tc_id = tc.get("id")
                    has_response = False
                    for j in range(idx + 1, len(self._messages)):
                        m2 = self._messages[j]
                        if isinstance(m2, dict) and m2.get("role") == "tool" and m2.get("tool_call_id") == tc_id:
                            has_response = True
                            break
                    if has_response:
                        continue
                    fn = tc.get("function", {})
                    name = fn.get("name", "")
                    arguments = fn.get("arguments", "{}")
                    tool_call_obj = type("obj", (object,), {
                        "id": tc_id,
                        "function": type("obj", (object,), {
                            "name": name,
                            "arguments": arguments
                        })
                    })
                    result = self.handle_tool_call(tool_call_obj)
                    content_dict = {k: v for k, v in result.items() if k != "screenshot_base64"}
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tc_id,
                        "content": json.dumps(content_dict)
                    })
            return True
        except Exception:
            return False
    def _maybe_export_text(self, state: EnvState):
        if self._export_done:
            return
        q = (self._query or "").lower()
        need_export = ("导出" in q) or ("txt" in q) or ("保存" in q) or ("csv" in q) or ("markdown" in q) or ("md" in q)
        if not need_export:
            return
        try:
            fmt = "txt"
            if "csv" in q:
                fmt = "csv"
            elif "markdown" in q or "md" in q:
                fmt = "md"
            items = []
            try:
                items = self._browser_computer.get_search_items()
            except Exception:
                items = []
            content = ""
            fname = "export.txt"
            if fmt == "csv":
                fname = "export.csv"
                for candidate in ["日本旅游路线.csv", "结果.csv", "export.csv"]:
                    if candidate.replace(".csv", "") in self._query:
                        fname = candidate
                        break
                const_header = "title,url,snippet\n"
                lines = [const_header]
                for it in items:
                    t = (it.get("title") or "").replace('"', '""')
                    u = (it.get("url") or "").replace('"', '""')
                    s = (it.get("snippet") or "").replace('"', '""')
                    lines.append(f'"{t}","{u}","{s}"')
                if len(lines) <= 1:
                    txt = self._browser_computer.get_result_text()
                    content = txt[:200000]
                else:
                    content = "\n".join(lines)
            elif fmt == "md":
                fname = "export.md"
                for candidate in ["日本旅游路线.md", "结果.md", "export.md"]:
                    if candidate.replace(".md", "") in self._query:
                        fname = candidate
                        break
                lines = ["# 日本旅游路线搜索结果", ""]
                for it in items:
                    t = it.get("title") or ""
                    u = it.get("url") or ""
                    s = it.get("snippet") or ""
                    lines.push = None
                    lines.append(f"- [{t}]({u})")
                    if s:
                        lines.append(f"  - {s}")
                if len(lines) <= 2:
                    txt = self._browser_computer.get_result_text()
                    content = txt[:200000]
                else:
                    content = "\n".join(lines)
            else:
                for candidate in ["日本旅游路线.txt", "结果.txt", "export.txt"]:
                    if candidate.replace(".txt", "") in self._query:
                        fname = candidate
                        break
                txt = self._browser_computer.get_result_text()
                if not txt or len(txt.strip()) < 200:
                    self._browser_computer.wait_5_seconds()
                    txt = self._browser_computer.get_visible_text()
                if (not txt or len(txt.strip()) < 200) and getattr(state, "dom", None):
                    txt = state.dom
                content = txt[:200000]
            path = self._write_text_file_local(fname, content)
            if path:
                self._export_done = True
                preview_lines = []
                try:
                    with open(path, "r", encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= 50:
                                break
                            preview_lines.append(line.rstrip("\n"))
                except Exception:
                    pass
            text_content = f"Exported file: {path}\nPreview (first {len(preview_lines)} lines):\n" + "\n".join(preview_lines)
            self._messages.append({
                "role": "user",
                "content": [
                    {"type": "text", "text": text_content}
                ]
            })
            if sys.stdin.isatty():
                try:
                    sysname = platform.system().lower()
                    if sysname == "darwin":
                        subprocess.run(["open", path], check=False)
                    elif sysname.startswith("win"):
                        os.startfile(path)
                    else:
                        subprocess.run(["xdg-open", path], check=False)
                except Exception:
                    pass
        except Exception:
            pass
    def _run_script(self, code: str):
        def _safe_write_text(path: str, content: str):
            try:
                base_dir = os.path.join(os.getcwd(), "exports")
                os.makedirs(base_dir, exist_ok=True)
                fname = "output.txt"
                if isinstance(path, str) and path.strip():
                    # Map /mnt/data/ to local exports
                    p = path.strip()
                    if p.startswith("/mnt/data/"):
                        p = p[len("/mnt/data/"):]
                    # Prevent directory traversal
                    p = os.path.basename(p)
                    if p:
                        fname = p
                target = os.path.join(base_dir, fname)
                with open(target, "w", encoding="utf-8") as f:
                    f.write(content or "")
                return target
            except Exception:
                return ""
        def _request_write(path: str, content_preview: str):
            try:
                if not isinstance(path, str) or not path.startswith("/mnt/data/"):
                    return {"status": "rejected", "reason": "Invalid path prefix"}
                base = os.path.basename(path.strip())
                if not base:
                    return {"status": "rejected", "reason": "Invalid filename"}
                if len(content_preview or "") > 1024 * 1024:
                    return {"status": "rejected", "reason": "Preview too large"}
                approved = True
                if sys.stdin.isatty():
                    prev = (content_preview or "")[:500]
                    print("Write preview:\n" + prev)
                    ans = input(f"Approve write to {path}? (y/n): ").strip().lower()
                    approved = ans == "y"
                if not approved:
                    return {"status": "rejected", "reason": "User rejected"}
                return {"status": "approved", "approved_path": f"/mnt/data/{base}"}
            except Exception as e:
                return {"status": "rejected", "reason": str(e)}
        api = {
            "navigate": lambda url: self._browser_computer.navigate(url),
            "click_at": lambda x, y: self._browser_computer.click_at(self.denormalize_x(x), self.denormalize_y(y)),
            "hover_at": lambda x, y: self._browser_computer.hover_at(self.denormalize_x(x), self.denormalize_y(y)),
            "type_text_at": lambda x, y, text: self._browser_computer.type_text_at(self.denormalize_x(x), self.denormalize_y(y), text, True, True),
            "scroll_document": lambda direction: self._browser_computer.scroll_document(direction),
            "scroll_at": lambda x, y, direction, magnitude=800: self._browser_computer.scroll_at(self.denormalize_x(x), self.denormalize_y(y), direction, magnitude),
            "wait_5_seconds": lambda: self._browser_computer.wait_5_seconds(),
            "go_back": lambda: self._browser_computer.go_back(),
            "go_forward": lambda: self._browser_computer.go_forward(),
            "key_combination": lambda keys: self._browser_computer.key_combination(keys.split("+")),
            "drag_and_drop": lambda x, y, dx, dy: self._browser_computer.drag_and_drop(self.denormalize_x(x), self.denormalize_y(y), self.denormalize_x(dx), self.denormalize_y(dy)),
            "wait_for_selector": lambda selector, timeout_ms=5000: self._browser_computer.wait_for_selector(selector, timeout_ms),
            "scroll_into_view": lambda selector: self._browser_computer.scroll_into_view(selector),
            "focus_selector": lambda selector: self._browser_computer.focus_selector(selector),
            "click_selector": lambda selector: self._browser_computer.click_selector(selector),
            "type_selector": lambda selector, text, press_enter=True, clear_before_typing=True: self._browser_computer.type_selector(selector, text, press_enter, clear_before_typing),
            "js_eval_safe": lambda script: self._browser_computer.js_eval_safe(script),
            "get_visible_text": lambda: self._browser_computer.get_visible_text(),
            "get_dom_summary": lambda: self._browser_computer.get_dom_summary(),
            "write_text_file": lambda path, content: _safe_write_text(path, content),
            "request_write": lambda path, preview: _request_write(path, preview),
        }
        run_sandbox(code, api)

    def _invoke_code_model(self):
        state = self._browser_computer.current_state()
        interactables = []
        if hasattr(state, "interactables") and state.interactables:
            interactables = state.interactables[:50]
        dom_summary = ""
        try:
            if hasattr(state, "dom") and state.dom:
                dom_summary = state.dom[:4000]
        except Exception:
            dom_summary = ""
        sys_prompt = (
            "You are a browser automation coding assistant. Output ONLY Python code using allowed APIs:\n"
            "- wait_for_selector(selector, timeout_ms)\n"
            "- scroll_into_view(selector)\n"
            "- focus_selector(selector)\n"
            "- click_selector(selector)\n"
            "- type_selector(selector, text, press_enter=True, clear_before_typing=True)\n"
            "- js_eval_safe(script)\n"
            "- navigate(url), wait_5_seconds()\n"
            "- get_visible_text(), get_dom_summary(), write_text_file(path, content)\n"
            "- request_write(path, content_preview) BEFORE write_text_file\n"
            "Rules:\n"
            "- Do NOT import or use any function not listed.\n"
            "- Decide a proper target site based on the query (e.g., gpt/openai/kimi) and call navigate(url) FIRST.\n"
            "- Locate input area: prefer [contenteditable=\"true\"], else input/textarea with placeholder/aria containing keywords: 输入/提问/搜索/问/消息/chat/ask/search.\n"
            "- For each candidate: wait_for_selector -> scroll_into_view -> focus_selector -> type_selector.\n"
            "- After submit, wait for a likely answer container (e.g. selectors containing message/chat/result/article). If no change, try next candidate. Max retries 2.\n"
            "- Do NOT use coordinates.\n"
            "Local Context:\n"
            f"- OS: {platform.system()}\n"
            f"- Project Root: {os.getcwd()}\n"
            f"- Export Dir: {os.path.join(os.getcwd(), 'exports')}\n"
            "- WRITE POLICY: Use request_write('/mnt/data/<name>.txt', preview) first; only on APPROVED use write_text_file.\n"
        )
        user_prompt = f"URL: {state.url}\nInteractables: {json.dumps(interactables, ensure_ascii=False)}\nDOM: {dom_summary}\nGoal: input the user's query into the site's chat input and submit, and export a TXT if requested.\nQuery: {self._query}"
        prompt = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ]
        completion = self._code_client.chat.completions.create(
            model=self._code_model,
            messages=prompt
        )
        code = completion.choices[0].message.content or ""
        try:
            import re
            m = re.search(r"```python\n([\s\S]*?)```", code)
            if m:
                code = m.group(1)
        except Exception:
            pass
        self._run_script(code)

    def agent_loop(self):
        # Initial step: Open browser if not already open (though user might prompt it)
        # We rely on the model to call open_web_browser if needed or if it's the first sensible action.
        
        while True:
            self._prune_history()
            
            if self._verbose:
                with console.status("Waiting for model response...", spinner="dots"):
                    try:
                        # Before requesting next turn, ensure pending tool_calls have tool responses
                        self._respond_to_pending_tool_calls()
                        response = self._client.chat.completions.create(
                            model=self._model_name,
                            messages=self._messages,
                            tools=TOOLS_SCHEMA,
                            tool_choice="auto"
                        )
                    except Exception as e:
                        msg = str(e)
                        termcolor.cprint(f"API Error: {msg}", "red")
                        if self._dev_logger:
                            try:
                                self._dev_logger.log_debug(error="API Error", exception=msg, stack="", attempts=[], env={"region": self._region, "model": self._model_name})
                            except Exception:
                                pass
                        if "tool_calls" in msg and "must be followed by tool messages" in msg:
                            # Try to append missing tool messages and retry once
                            if self._respond_to_pending_tool_calls():
                                try:
                                    response = self._client.chat.completions.create(
                                        model=self._model_name,
                                        messages=self._messages,
                                        tools=TOOLS_SCHEMA,
                                        tool_choice="auto"
                                    )
                                except Exception:
                                    self._messages = self._messages[:2]
                            else:
                                self._messages = self._messages[:2]
                        fallback_candidates = ["qwen3-max", "qwen-vl-max", "qwen3-coder-plus"]
                        for cand in fallback_candidates:
                            if cand == self._model_name:
                                continue
                            try:
                                self._model_name = cand
                                response = self._client.chat.completions.create(
                                    model=self._model_name,
                                    messages=self._messages,
                                    tools=TOOLS_SCHEMA,
                                    tool_choice="auto"
                                )
                                break
                            except Exception:
                                continue
                        else:
                            return

            message = response.choices[0].message
            # Append the assistant's response to history
            self._messages.append(message.model_dump())
            
            if message.tool_calls:
                self._consecutive_text_responses = 0
                for tool_call in message.tool_calls:
                    result = self.handle_tool_call(tool_call)
                    content_dict = {k: v for k, v in result.items() if k != "screenshot_base64"}
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(content_dict)
                    })
            else:
                # No tool calls, just text response
                content = message.content or ""
                termcolor.cprint(f"\nAssistant: {content}\n", "green")
                # Reinforce selector preference in dev logs
                if self._dev_logger:
                    try:
                        self._dev_logger.log_ai_interaction(
                            question=self._query,
                            suggestion=content,
                            decision="pending",
                            rationale="",
                            related_change_ids=[]
                        )
                    except Exception:
                        pass
                # Attempt local export when assistant emits text (safe: no pending tool_calls)
                try:
                    st = self._browser_computer.current_state()
                    self._maybe_export_text(st)
                except Exception:
                    pass
                
                # Check for explicit completion signal
                if "TASK_COMPLETED:" in content:
                    last_url = self._get_last_observed_url() or ""
                    if last_url and not (
                        last_url.startswith("about:blank")
                        or last_url.startswith("chrome-error://")
                        or last_url.startswith("data:")
                    ):
                        termcolor.cprint("Task marked as completed by agent. Handing over to user.", "green", attrs=["bold"])
                        if self._dev_logger:
                            try:
                                self._dev_logger.log_ai_interaction(
                                    question=self._query,
                                    suggestion=content,
                                    decision="accepted",
                                    rationale="",
                                    related_change_ids=[]
                                )
                            except Exception:
                                pass
                        break
                    termcolor.cprint("TASK_COMPLETED received but browser state is not valid. Continuing.", "yellow")
                    self._messages.append({
                        "role": "user",
                        "content": f"You said TASK_COMPLETED but the current URL is '{last_url or 'unknown'}'. Use tools to load the page and actually finish the task."
                    })
                    try:
                        self._invoke_code_model()
                    except Exception:
                        pass
                    continue
                
                # FALLBACK: Try to parse text-based tool calls from the message content
                content_text = message.content or ""
                
                # Regex to find function calls like tool_name(arg1, arg2) or tool_name("arg")
                # This is a simple parser and might need refinement for complex nested args
                func_calls = re.findall(r'(\w+)\s*\((.*?)\)', content_text)
                
                executed_fallback = False
                
                if func_calls:
                    termcolor.cprint(f"Fallback: Detected {len(func_calls)} text-based tool calls. Executing...", "yellow")
                    
                    for func_name, args_str in func_calls:
                        # Check if it's a valid tool
                        valid_tools = [t["function"]["name"] for t in TOOLS_SCHEMA]
                        if func_name not in valid_tools:
                            continue
                            
                        # Try to parse arguments
                        args = {}
                        if args_str.strip():
                            # Heuristic: try to parse key=value or positional args
                            # For now, let's handle simple cases manually or use a smarter parsing if needed
                            # But specifically for 'navigate("url")', 'type_text_at(x,y,"text")'
                            
                            # Clean up quotes
                            clean_args = [a.strip().strip('"\'') for a in args_str.split(',')]
                            
                            if func_name == "navigate" and len(clean_args) >= 1:
                                url = clean_args[0].strip()
                                if len(url) >= 2 and url.startswith("`") and url.endswith("`"):
                                    url = url[1:-1]
                                url = url.strip().strip("`").strip()
                                args = {"url": url}
                            elif func_name == "type_text_at" and len(clean_args) >= 3:
                                try:
                                    args = {
                                        "x": int(clean_args[0]), 
                                        "y": int(clean_args[1]), 
                                        "text": clean_args[2]
                                    }
                                    # 默认更稳：回车触发搜索
                                    args["press_enter"] = True
                                except: pass
                            elif func_name == "click_at" and len(clean_args) >= 2:
                                try:
                                    args = {"x": int(clean_args[0]), "y": int(clean_args[1])}
                                except: pass
                            elif func_name == "key_combination" and len(clean_args) >= 1:
                                args = {"keys": clean_args[0]}
                            elif func_name == "scroll_document" and len(clean_args) >= 1:
                                arg0 = clean_args[0].lower().strip("'\"")
                                if arg0 in ["up", "down", "left", "right"]:
                                    args = {"direction": arg0}
                                else:
                                    # Heuristic: handle cases like scroll_document(0, 500) or scroll_document(500)
                                    # We scan args for a significant number
                                    found_direction = None
                                    for arg in clean_args:
                                        try:
                                            val = int(arg)
                                            if val > 50:
                                                found_direction = "down"
                                            elif val < -50:
                                                found_direction = "up"
                                        except: pass
                                    
                                    if found_direction:
                                        args = {"direction": found_direction}
                                    else:
                                        # Default fallback if we can't figure it out, but saw args
                                        # If the first arg is '0', it might be scroll(0, y)
                                        if len(clean_args) > 1:
                                            args = {"direction": "down"}

                            elif func_name == "scroll_at" and len(clean_args) >= 3:
                                try:
                                    args = {
                                        "x": int(clean_args[0]),
                                        "y": int(clean_args[1]),
                                        "direction": clean_args[2].strip("'\"")
                                    }
                                    if len(clean_args) >= 4:
                                        args["magnitude"] = int(clean_args[3])
                                    
                                    # Fix direction if it's invalid (e.g. number)
                                    if args["direction"] not in ["up", "down", "left", "right"]:
                                        mag = args.get("magnitude", 800)
                                        if mag > 0:
                                            args["direction"] = "down"
                                        elif mag < 0:
                                            args["direction"] = "up"
                                            args["magnitude"] = abs(mag)
                                        else:
                                            args["direction"] = "down"
                                except: pass
                            
                            elif func_name == "hover_at" and len(clean_args) >= 2:
                                try:
                                    args = {"x": int(clean_args[0]), "y": int(clean_args[1])}
                                except: pass
                        
                        # Execute
                        result = self.handle_tool_call(type("obj", (object,), {
                            "function": type("obj", (object,), {
                                "name": func_name,
                                "arguments": json.dumps(args)
                            })
                        }))
                        
                        # Add result to history
                        self._messages.append({
                            "role": "tool",
                            "tool_call_id": f"fallback_{func_name}_{int(time.time())}",
                            "content": json.dumps({k: v for k, v in result.items() if k != "screenshot_base64"})
                        })

                        if isinstance(result, dict) and result.get("error"):
                            self._messages.append({
                                "role": "user",
                                "content": f"The last action '{func_name}' failed with error: {result.get('error')}. Try a corrected tool call."
                            })
                            executed_fallback = True
                            if func_name == "navigate":
                                break
                            continue
                        
                        if "screenshot_base64" in result:
                            text_content = f"Action completed. Current browser state (URL: {result.get('url')})"
                            if result.get("interactables"):
                                text_content += "\n" + result.get("interactables")
                            
                            self._messages.append({
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": text_content},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{result['screenshot_base64']}"
                                        }
                                    }
                                ]
                            })
                        executed_fallback = True
                        
                        # IMPORTANT: If we navigated, we MUST stop execution to allow the model to see the new page.
                        # This prevents "blind" typing on a page that hasn't been observed yet.
                        if func_name == "navigate":
                            termcolor.cprint("Navigation detected. Stopping further actions in this turn to allow observation.", "yellow")
                            break
                        
                # Check for open_web_browser specifically if no regex match (sometimes it's just the name)
                if not executed_fallback and "open_web_browser" in content_text and "open_web_browser" not in str(self._messages[-2:]):
                     # Construct a fake tool call
                    termcolor.cprint("Fallback: Detected 'open_web_browser' in text. Executing...", "yellow")
                    result = self.handle_tool_call(type("obj", (object,), {
                        "function": type("obj", (object,), {
                            "name": "open_web_browser",
                            "arguments": "{}"
                        })
                    }))
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": "fallback_call",
                        "content": json.dumps({k: v for k, v in result.items() if k != "screenshot_base64"})
                    })
                    if "screenshot_base64" in result:
                        text_content = f"Action completed. Current browser state (URL: {result.get('url')})"
                        if result.get("interactables"):
                            text_content += "\n" + result.get("interactables")
                        
                        self._messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{result['screenshot_base64']}"
                                    }
                                }
                            ]
                        })
                    executed_fallback = True

                if not executed_fallback and re.search(r"\bwait_5_seconds\b", content_text) and "wait_5_seconds" not in str(self._messages[-2:]):
                    termcolor.cprint("Fallback: Detected 'wait_5_seconds' in text. Executing...", "yellow")
                    result = self.handle_tool_call(type("obj", (object,), {
                        "function": type("obj", (object,), {
                            "name": "wait_5_seconds",
                            "arguments": "{}"
                        })
                    }))
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": "fallback_wait_5_seconds",
                        "content": json.dumps({k: v for k, v in result.items() if k != "screenshot_base64"})
                    })
                    if "screenshot_base64" in result:
                        text_content = f"Action completed. Current browser state (URL: {result.get('url')})"
                        if result.get("interactables"):
                            text_content += "\n" + result.get("interactables")
                        self._messages.append({
                            "role": "user",
                            "content": [
                                {"type": "text", "text": text_content},
                                {
                                    "type": "image_url",
                                    "image_url": {
                                        "url": f"data:image/png;base64,{result['screenshot_base64']}"
                                    }
                                }
                            ]
                        })
                    executed_fallback = True

                if executed_fallback:
                    self._consecutive_text_responses = 0
                    continue

                # Check for "chatty" loop
                # If we have executed a fallback, we should give it a chance to run.
                # If we have consecutive text responses that are NOT fallbacks, then we stop.
                
                # Check for exact repetition in fallback calls to prevent infinite loops
                if executed_fallback and len(self._messages) >= 4:
                    last_tool_msg = self._messages[-2]
                    prev_tool_msg = self._messages[-4]
                    if (last_tool_msg.get("role") == "tool" and prev_tool_msg.get("role") == "tool" and
                        last_tool_msg.get("content") == prev_tool_msg.get("content") and
                        "fallback" in last_tool_msg.get("tool_call_id", "")):
                         termcolor.cprint("Detected repetitive fallback actions. Stopping loop.", "red")
                         break

                if self._consecutive_text_responses >= 5: # Increased tolerance slightly
                    termcolor.cprint("Too many text responses without action. Stopping.", "red")
                    break

                self._consecutive_text_responses += 1
                
                # Append a user message forcing tool use
                self._messages.append({
                    "role": "user",
                    "content": "You provided a text response but did not call any tools. Please strictly use the provided tools (like 'open_web_browser', 'click_at', 'type_text_at') to perform the actions. Do not just describe them."
                })
                termcolor.cprint("Reminder sent to model to use tools...", "yellow")
                continue
