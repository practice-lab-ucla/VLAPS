from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()  # ✅ loads .env from current directory

# ===== Model and API key =====
MODEL = "gpt-4o-mini"   #
API_KEY = os.environ.get("CHATGPT_API_KEY")



# ===== Default prompt =====
DEFAULT_PROMPT = (
    "Compare the expert policy (green path) and novice policy (red path) shown in the image.\n\n"
    "First, output your decision strictly in this JSON format:\n"
    "{{\"response\": <1 or 0>}}\n\n"
    "Where:\n"
    " - 1 → The expert policy is **significantly better** than the novice policy.\n"
    " - 0 → The expert policy is **similar** to the novice policy.\n\n"
    "Immediately after the JSON object, provide a short one-sentence explanation for your decision.\n"
    "Do not add any extra formatting or text before the JSON."
)



def make_client():
    if not API_KEY or API_KEY.startswith("REPLACE_WITH"):
        raise RuntimeError("API key missing. Set OPENAI_API_KEY or edit vlm_config.py.")
    return OpenAI(api_key=API_KEY)
