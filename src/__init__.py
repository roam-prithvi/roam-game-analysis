from beartype.claw import beartype_this_package

beartype_this_package()

import os
from pathlib import Path

from dotenv import load_dotenv

env_path: Path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(
    dotenv_path=env_path, override=False
)  # override=False keeps already-exported vars

GEMINI_API_KEY: str = os.environ["GEMINI_API_KEY"]
OPENAI_API_KEY: str = os.environ["OPENAI_API_KEY"]
ANTHROPIC_API_KEY: str = os.environ["ANTHROPIC_API_KEY"]
SAM2_API_KEY: str = os.environ["SAM2_API_KEY"]
SIEVE_API_KEY: str = os.environ["SIEVE_API_KEY"]
