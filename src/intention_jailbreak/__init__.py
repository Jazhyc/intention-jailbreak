"""Intention Jailbreak - Research project for intention analysis and jailbreak studies."""

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

__version__ = "0.1.0"

# Make dataset utilities and common constants easily accessible
from . import dataset
from . import common
