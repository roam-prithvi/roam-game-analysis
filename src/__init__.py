from beartype.claw import beartype_this_package

beartype_this_package()

import os
from pathlib import Path

from dotenv import load_dotenv

# 1) Locate and load the .env file (defaults to .env in CWD)
env_path: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(
    dotenv_path=env_path, override=False
)  # override=False keeps already-exported vars

# 2) Access the keys
GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
