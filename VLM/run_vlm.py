# run_vlm.py
import base64, mimetypes, json, re, os, datetime
from pathlib import Path
from .vlm_config import MODEL, DEFAULT_PROMPT, make_client
from typing import Optional

def run_vlm(image_path: str, prompt: Optional[str] = None) -> int:
    """
    Sends an image + text prompt to the model.
    Prints and logs the full raw response (JSON + explanation),
    but returns only the integer 0 or 1.
    """
    client = make_client()
    prompt = prompt or DEFAULT_PROMPT

    # --- Image encoding ---
    path = Path(image_path)
    if not path.exists():
        path = Path(os.path.dirname(__file__)) / image_path
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or "image/png"
    data_uri = f"data:{mime};base64,{base64.b64encode(path.read_bytes()).decode()}"

    # --- Build request ---
    messages = [
        {"role": "system", "content": "You are a strict classifier. Output valid JSON first."},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_uri}}
            ],
        }
    ]

    # --- API call ---
    try:
        response = client.chat.completions.create(model=MODEL, messages=messages)
    except Exception as e:
        raise RuntimeError(f"OpenAI API error when calling model '{MODEL}': {e}")

    try:
        raw_text = response.choices[0].message.content.strip()
    except Exception:
        raw_text = str(response)

    # --- Print raw response ---
    print("=== MODEL RAW RESPONSE ===")
    print(raw_text)
    print("==========================")

    # --- Log to vlm_log folder ---
    log_dir = Path(__file__).parent / "vlm_log"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"vlm_response_{timestamp}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Model: {MODEL}\n")
        f.write(f"Prompt: {prompt}\n\n")
        f.write("=== RAW MODEL RESPONSE ===\n")
        f.write(raw_text + "\n")
    print(f"📝 Saved raw response to: {log_file}")

    # --- Parse JSON ---
    m = re.search(r"\{.*?\}", raw_text, flags=re.DOTALL)
    if not m:
        raise ValueError(f"No JSON object found in model output: {raw_text!r}")
    json_text = m.group(0)
    try:
        parsed = json.loads(json_text)
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON returned: {json_text!r}")

    if "response" not in parsed:
        raise ValueError(f"Missing 'response' key: {parsed!r}")

    val = parsed["response"]
    try:
        intval = int(val)
    except Exception:
        raise ValueError(f"'response' is not numeric: {val!r}")
    if intval not in (0, 1):
        raise ValueError(f"'response' must be 0 or 1, got {intval!r}")

    # --- Return only integer result ---
    return intval


if __name__ == "__main__":
    img = "test.png"
    result = run_vlm(img)
    print("Returned result:", result)
