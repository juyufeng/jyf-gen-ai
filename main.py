# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
from app_logging.context_logger import ContextLogger
from app_logging.dev_context_logger import DevContextLogger
from models.selector import select_model

from agent import BrowserAgent
from agent_qwen import QwenAgent
from computers import BrowserbaseComputer, PlaywrightComputer


PLAYWRIGHT_SCREEN_SIZE = (1440, 900)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the browser agent with a query.")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="The query for the browser agent to execute.",
    )

    parser.add_argument(
        "--env",
        type=str,
        choices=("playwright", "browserbase"),
        default="playwright",
        help="The computer use environment to use.",
    )
    parser.add_argument(
        "--initial_url",
        type=str,
        default="about:blank",
        help="The inital URL loaded for the computer.",
    )
    parser.add_argument(
        "--highlight_mouse",
        action="store_true",
        default=False,
        help="If possible, highlight the location of the mouse.",
    )
    parser.add_argument(
        "--model",
        default='gemini-2.5-computer-use-preview-10-2025',
        help="Set which main model to use.",
    )
    parser.add_argument(
        "--provider",
        type=str,
        choices=("gemini", "qwen"),
        default="gemini",
        help="The AI provider to use.",
    )
    parser.add_argument(
        "--region",
        type=str,
        choices=("beijing", "singapore"),
        default="beijing",
        help="DashScope region.",
    )
    parser.add_argument(
        "--log_dir",
        type=str,
        default="./logs/context",
        help="Directory to store context logs.",
    )
    parser.add_argument(
        "--force_model",
        type=str,
        default=None,
        help="Force a specific provider model id.",
    )
    parser.add_argument(
        "--allow_script",
        action="store_true",
        default=False,
        help="Allow sandbox script execution tool.",
    )
    parser.add_argument(
        "--dev_log_dir",
        type=str,
        default="./logs/dev",
        help="Directory to store programming-assistant logs.",
    )
    parser.add_argument(
        "--module",
        type=str,
        default="default",
        help="Current development module for dev logs.",
    )
    parser.add_argument(
        "--session_label",
        type=str,
        default="",
        help="Session label for dev logs.",
    )
    parser.add_argument(
        "--strict_redaction",
        action="store_true",
        default=True,
        help="Enable strict redaction in dev logs.",
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="The API key for the chosen provider. If not set, uses environment variables.",
    )
    args = parser.parse_args()

    api_key = args.api_key
    if isinstance(api_key, str) and api_key:
        if args.provider == "qwen":
            os.environ["DASHSCOPE_API_KEY"] = api_key
        else:
            os.environ["GEMINI_API_KEY"] = api_key

    if args.env == "playwright":
        env = PlaywrightComputer(
            screen_size=PLAYWRIGHT_SCREEN_SIZE,
            initial_url=args.initial_url,
            highlight_mouse=args.highlight_mouse,
        )
    elif args.env == "browserbase":
        env = BrowserbaseComputer(
            screen_size=PLAYWRIGHT_SCREEN_SIZE,
            initial_url=args.initial_url
        )
    else:
        raise ValueError("Unknown environment: ", args.env)

    with env as browser_computer:
        log_dir = args.log_dir if isinstance(args.log_dir, str) and args.log_dir else "./logs/context"
        logger = ContextLogger(log_dir)
        dev_log_dir = args.dev_log_dir if isinstance(args.dev_log_dir, str) and args.dev_log_dir else "./logs/dev"
        module_name = args.module if isinstance(args.module, str) else "default"
        session_label = args.session_label if isinstance(args.session_label, str) else ""
        strict_redaction = bool(args.strict_redaction)
        dev_logger = DevContextLogger(base_dir=dev_log_dir, module=module_name, session_label=session_label, strict_redaction=strict_redaction)
        dev_logger.log_context(
            module_desc=f"Module: {args.module}",
            business_logic="",
            architecture_notes="",
            external_services=[{"name": "DashScope", "purpose": "LLM", "doc_url": "https://help.aliyun.com/zh/model-studio/models", "version": ""}],
        )
        if args.provider == "qwen":
            agent = QwenAgent(
                browser_computer=browser_computer,
                query=args.query,
                model_name=args.model if args.model != 'gemini-2.5-computer-use-preview-10-2025' else 'qwen-vl-max',
                logger=logger,
                region=args.region,
                force_model=args.force_model,
                dev_logger=dev_logger,
            )
        else:
            agent = BrowserAgent(
                browser_computer=browser_computer,
                query=args.query,
                model_name=args.model,
            )
        agent.agent_loop()
        
        print("\nSession finished. The browser is still open for your review.")
        if sys.stdin.isatty():
            input("Press Enter to close the browser and exit...")
    return 0


if __name__ == "__main__":
    main()
