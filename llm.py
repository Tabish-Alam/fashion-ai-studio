import json
import os
from typing import Any

from openai import OpenAI


XAI_API_KEY = os.getenv("XAI_API_KEY")

XAI_BASE_URL = "https://api.x.ai/v1"

GROK_MODEL = os.getenv(
    "GROK_MODEL",
    "grok-4.1-fast-non-reasoning",
)


client = None

if XAI_API_KEY:
    client = OpenAI(
        api_key=XAI_API_KEY,
        base_url=XAI_BASE_URL,
    )


SYSTEM_PROMPT = """
You are the intent router for a Fashion AI Studio.

The application has these tools:

1. search
   Search the user's LOCAL fashion image catalog using CLIP.

2. analyze
   Analyze the currently selected fashion image.

3. similar
   Find visually/semantically similar catalog items.

4. outfit
   Recommend complementary catalog items for the selected item.

5. edit
   Apply a deterministic image edit.

6. analyze_and_similar
   Analyze selected image and find similar catalog items.

7. analyze_and_outfit
   Analyze selected image and build an outfit.

8. search_and_outfit
   Search the catalog based on a fashion description.

9. help
   Explain what the Fashion AI Studio can do.

10. unsupported
    Use when the user asks for a capability the current application
    cannot perform.

Supported edit operations:
- bright
- contrast
- black_and_white
- sharpen
- blur
- rotate_right
- rotate_left
- flip_horizontal
- flip_vertical

IMPORTANT:

The catalog is local. Never claim that the application searches the
internet or external stores.

CLIP retrieves existing catalog images. It does NOT generate clothing.

Pillow edits are basic deterministic image transformations.

The application CANNOT currently:
- change clothing color semantically
- replace clothing
- remove objects
- change backgrounds
- put clothes on a person
- generate new fashion images
- perform virtual try-on

Return ONLY valid JSON.

Schema:

{
  "intent": "search",
  "query": "",
  "edit_operation": null,
  "needs_selected_image": false,
  "response": ""
}

Examples:

User:
"show me casual summer clothes"

Output:
{
  "intent": "search",
  "query": "casual summer clothes",
  "edit_operation": null,
  "needs_selected_image": false,
  "response": ""
}

User:
"find black elegant office clothes"

Output:
{
  "intent": "search",
  "query": "black elegant office clothes",
  "edit_operation": null,
  "needs_selected_image": false,
  "response": ""
}

User:
"analyze this image"

Output:
{
  "intent": "analyze",
  "query": "",
  "edit_operation": null,
  "needs_selected_image": true,
  "response": ""
}

User:
"analyze this and find similar pieces"

Output:
{
  "intent": "analyze_and_similar",
  "query": "",
  "edit_operation": null,
  "needs_selected_image": true,
  "response": ""
}

User:
"build an outfit around this"

Output:
{
  "intent": "outfit",
  "query": "",
  "edit_operation": null,
  "needs_selected_image": true,
  "response": ""
}

User:
"make this brighter"

Output:
{
  "intent": "edit",
  "query": "",
  "edit_operation": "bright",
  "needs_selected_image": true,
  "response": ""
}

User:
"turn this black and white"

Output:
{
  "intent": "edit",
  "query": "",
  "edit_operation": "black_and_white",
  "needs_selected_image": true,
  "response": ""
}

User:
"change this jacket to red leather"

Output:
{
  "intent": "unsupported",
  "query": "",
  "edit_operation": null,
  "needs_selected_image": true,
  "response": "The current editor cannot regenerate clothing or change garment materials. It supports brightness, contrast, black and white, sharpening, blur, rotation and flipping."
}

Extract the cleanest possible fashion retrieval query.

Do not include conversational filler such as:
"show me", "please", "can you", "I want", or "find me"
inside the query.

Do not answer the fashion question yourself.
Your job is routing.
"""


VALID_INTENTS = {
    "search",
    "analyze",
    "similar",
    "outfit",
    "edit",
    "analyze_and_similar",
    "analyze_and_outfit",
    "search_and_outfit",
    "help",
    "unsupported",
}


VALID_EDIT_OPERATIONS = {
    "bright",
    "contrast",
    "black_and_white",
    "sharpen",
    "blur",
    "rotate_right",
    "rotate_left",
    "flip_horizontal",
    "flip_vertical",
}


def llm_available() -> bool:
    return client is not None


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")

        if start == -1 or end == -1:
            raise

        return json.loads(
            text[start:end + 1]
        )


def _clean_result(
    result: dict[str, Any],
) -> dict[str, Any]:

    intent = str(
        result.get(
            "intent",
            "search",
        )
    ).strip().lower()

    if intent not in VALID_INTENTS:
        intent = "search"

    query = str(
        result.get(
            "query",
            "",
        )
        or ""
    ).strip()

    edit_operation = result.get(
        "edit_operation"
    )

    if edit_operation is not None:
        edit_operation = (
            str(edit_operation)
            .strip()
            .lower()
        )

        if (
            edit_operation
            not in VALID_EDIT_OPERATIONS
        ):
            edit_operation = None

    needs_selected_image = bool(
        result.get(
            "needs_selected_image",
            False,
        )
    )

    response = str(
        result.get(
            "response",
            "",
        )
        or ""
    ).strip()

    return {
        "intent": intent,
        "query": query,
        "edit_operation": edit_operation,
        "needs_selected_image":
            needs_selected_image,
        "response": response,
    }


def fallback_router(
    message: str,
) -> dict[str, Any]:

    text = message.lower().strip()

    if any(
        phrase in text
        for phrase in [
            "what can you do",
            "help me",
            "your capabilities",
            "what do you do",
        ]
    ):
        return {
            "intent": "help",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": False,
            "response": "",
        }

    if (
        "analy" in text
        and
        any(
            word in text
            for word in [
                "similar",
                "match",
                "like this",
            ]
        )
    ):
        return {
            "intent":
                "analyze_and_similar",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": True,
            "response": "",
        }

    if (
        "analy" in text
        and
        "outfit" in text
    ):
        return {
            "intent":
                "analyze_and_outfit",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": True,
            "response": "",
        }

    if "analy" in text:
        return {
            "intent": "analyze",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": True,
            "response": "",
        }

    if any(
        word in text
        for word in [
            "similar",
            "similar item",
            "similar clothes",
            "like this",
        ]
    ):
        return {
            "intent": "similar",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": True,
            "response": "",
        }

    if "outfit" in text:
        return {
            "intent": "outfit",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": True,
            "response": "",
        }

    edit_map = {
        "brighter":
            "bright",

        "brighten":
            "bright",

        "brightness":
            "bright",

        "contrast":
            "contrast",

        "black and white":
            "black_and_white",

        "grayscale":
            "black_and_white",

        "grey scale":
            "black_and_white",

        "sharpen":
            "sharpen",

        "blur":
            "blur",

        "rotate right":
            "rotate_right",

        "rotate left":
            "rotate_left",

        "flip horizontal":
            "flip_horizontal",

        "flip horizontally":
            "flip_horizontal",

        "flip vertical":
            "flip_vertical",

        "flip vertically":
            "flip_vertical",
    }

    for phrase, operation in edit_map.items():

        if phrase in text:

            return {
                "intent": "edit",
                "query": "",
                "edit_operation":
                    operation,
                "needs_selected_image":
                    True,
                "response": "",
            }

    cleaned = text

    filler = [
        "can you show me",
        "could you show me",
        "please show me",
        "can you find me",
        "could you find me",
        "please find me",
        "show me",
        "find me",
        "search for",
        "i want",
        "i need",
        "please",
    ]

    for phrase in filler:

        if cleaned.startswith(
            phrase
        ):
            cleaned = cleaned[
                len(phrase):
            ].strip()

    return {
        "intent": "search",
        "query":
            cleaned or message,
        "edit_operation": None,
        "needs_selected_image": False,
        "response": "",
    }


def understand_request(
    message: str,
) -> dict[str, Any]:

    if not message.strip():

        return {
            "intent": "help",
            "query": "",
            "edit_operation": None,
            "needs_selected_image": False,
            "response": "",
        }

    if client is None:

        return fallback_router(
            message
        )

    try:

        completion = (
            client.chat.completions.create(
                model=GROK_MODEL,

                temperature=0,

                messages=[
                    {
                        "role":
                            "system",

                        "content":
                            SYSTEM_PROMPT,
                    },
                    {
                        "role":
                            "user",

                        "content":
                            message,
                    },
                ],
            )
        )

        content = (
            completion
            .choices[0]
            .message
            .content
        )

        if not content:
            raise ValueError(
                "Empty Grok response."
            )

        result = _extract_json(
            content
        )

        return _clean_result(
            result
        )

    except Exception as exc:

        print(
            "Grok routing warning:",
            exc,
        )

        return fallback_router(
            message
        )
