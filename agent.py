from anthropic import Anthropic

from config import settings, SYSTEM_PROMPT

# direct API client — the "phone line" to Claude (no framework);
# the API key is read from the environment (config.py puts it there)
client = Anthropic()
