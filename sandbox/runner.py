import ast

SAFE_NAMES = {
    "navigate",
    "click_at",
    "hover_at",
    "type_text_at",
    "scroll_document",
    "scroll_at",
    "wait_5_seconds",
    "go_back",
    "go_forward",
    "key_combination",
    "drag_and_drop",
    "wait_for_selector",
    "scroll_into_view",
    "focus_selector",
    "click_selector",
    "type_selector",
    "js_eval_safe"
}

def _validate(code: str) -> None:
    tree = ast.parse(code or "")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.With, ast.Try, ast.ClassDef)):
            raise ValueError("Unsupported construct")
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            if node.value.id not in SAFE_NAMES:
                raise ValueError("Unsafe attribute access")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id not in SAFE_NAMES:
                raise ValueError("Unknown function")

def run(code: str, api: dict):
    if isinstance(code, str):
        code = code.replace("\\n", "\n")
    _validate(code)
    env = {k: v for k, v in api.items() if k in SAFE_NAMES}
    exec(code, {"__builtins__": {}}, env)
