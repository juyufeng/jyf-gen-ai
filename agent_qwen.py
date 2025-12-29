import os
import time
import base64
import json
from typing import Literal, Optional, Union, Any, List, Dict
from openai import OpenAI
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
    ):
        self._browser_computer = browser_computer
        self._query = query
        self._model_name = model_name
        self._verbose = verbose
        
        # Initialize OpenAI client for DashScope
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if not api_key:
            raise ValueError("DASHSCOPE_API_KEY environment variable is not set")
            
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        
        self._consecutive_text_responses = 0

        self._messages = [
            {
                "role": "system",
                "content": """You are a browser automation agent. Your goal is to complete the user's request by directly controlling the browser.
1. You MUST use the provided tools to interact with the browser. Do NOT just describe what you are going to do. Just DO it.
2. The screen coordinates are normalized to 0-1000 range (top-left is 0,0).
3. If the browser is not open, you MUST call 'open_web_browser' first.
4. You will receive screenshots of the browser state after each action. Use them to locate elements for the next action.
5. You may also receive a list of "Detected Interactive Elements" with their coordinates. PREFER using these coordinates over guessing.
6. Only when the task is fully complete should you respond with a final text answer.
7. CRITICAL: Continually assess if the current page contains the answer to the user's request. If you see the answer or have completed the task, STOP immediately.
   - Do NOT keep clicking or scrolling just for the sake of it.
   - If the task is "Search for X", and you see X in the search results, the task is DONE.
   - When done, output a final text response starting with "TASK_COMPLETED:" followed by a brief summary.
8. If you need to search, navigate, click, or type, use the corresponding tools immediately.
9. IMPORTANT: Do NOT output the function calls as text in your response (e.g., do not write 'click_at(100, 200)' in the chat). You MUST use the native 'tool_calls' field to execute functions.

Available Tools: open_web_browser, click_at, hover_at, type_text_at, scroll_document, scroll_at, navigate, go_back, go_forward, wait_5_seconds, key_combination, drag_and_drop.
Do NOT invent tools like 'search_on_baidu'. Use 'navigate' to go to URLs.

IMPORTANT: When you perform an action (like click or type), you MUST wait for the page to update. Do NOT repeat the exact same action multiple times in a row without seeing a change. If an action fails or has no effect, try a slightly different coordinate or a different approach.
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
            elif function_name == "scroll_document":
                result_state = self._browser_computer.scroll_document(args["direction"])
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
                 # Just take a subset or format them nicely
                 # Format: "Label (x, y)"
                 elements = []
                 for el in result_state.interactables[:50]: # Limit to top 50 elements to save tokens
                     # Normalize coordinates back to 0-1000 for the model
                     norm_x = int(el['x'] / self._browser_computer.screen_size()[0] * 1000)
                     norm_y = int(el['y'] / self._browser_computer.screen_size()[1] * 1000)
                     if el['label']:
                        elements.append(f"{el['label']} ({el['tag']}) at ({norm_x}, {norm_y})")
                 
                 if elements:
                    interactables_text = "\nDetected Interactive Elements:\n" + "\n".join(elements)

            return {
                "url": result_state.url,
                "status": "success",
                "interactables": interactables_text, # Add this field
                "screenshot_base64": base64.b64encode(result_state.screenshot).decode('utf-8')
            }
        return {"status": "success"}

    def agent_loop(self):
        # Initial step: Open browser if not already open (though user might prompt it)
        # We rely on the model to call open_web_browser if needed or if it's the first sensible action.
        
        while True:
            self._prune_history()
            
            if self._verbose:
                with console.status("Waiting for model response...", spinner="dots"):
                    try:
                        response = self._client.chat.completions.create(
                            model=self._model_name,
                            messages=self._messages,
                            tools=TOOLS_SCHEMA,
                            tool_choice="auto"
                        )
                    except Exception as e:
                        termcolor.cprint(f"API Error: {e}", "red")
                        return

            message = response.choices[0].message
            # Append the assistant's response to history
            self._messages.append(message.model_dump())
            
            if message.tool_calls:
                self._consecutive_text_responses = 0
                for tool_call in message.tool_calls:
                    result = self.handle_tool_call(tool_call)
                    
                    # Tool output content (JSON without image)
                    content_dict = {k: v for k, v in result.items() if k != "screenshot_base64"}
                    self._messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": json.dumps(content_dict)
                    })
                    
                    # If we have a screenshot, append it as a new User message to show the state
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
            else:
                # No tool calls, just text response
                content = message.content or ""
                termcolor.cprint(f"\nAssistant: {content}\n", "green")
                
                # Check for explicit completion signal
                if "TASK_COMPLETED:" in content:
                    termcolor.cprint("Task marked as completed by agent. Handing over to user.", "green", attrs=["bold"])
                    break
                
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
                                args = {"url": clean_args[0]}
                            elif func_name == "type_text_at" and len(clean_args) >= 3:
                                try:
                                    args = {
                                        "x": int(clean_args[0]), 
                                        "y": int(clean_args[1]), 
                                        "text": clean_args[2]
                                    }
                                except: pass
                            elif func_name == "click_at" and len(clean_args) >= 2:
                                try:
                                    args = {"x": int(clean_args[0]), "y": int(clean_args[1])}
                                except: pass
                            elif func_name == "key_combination" and len(clean_args) >= 1:
                                args = {"keys": clean_args[0]}
                            elif func_name == "scroll_document" and len(clean_args) >= 1:
                                args = {"direction": clean_args[0]}
                        
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
