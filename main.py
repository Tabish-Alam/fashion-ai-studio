import hashlib
import json
from pathlib import Path
from urllib.parse import quote

import numpy as np
import torch

from PIL import (
    Image,
    ImageEnhance,
    ImageFilter,
    ImageOps,
)

from transformers import (
    CLIPModel,
    CLIPProcessor,
)

from llm import (
    understand_request,
    llm_available,
)


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = (
    Path(__file__)
    .resolve()
    .parent
)

STATIC_DIR = (
    BASE_DIR /
    "static"
)

CATALOG_DIR = (
    STATIC_DIR /
    "catalog"
)

EDITED_DIR = (
    STATIC_DIR /
    "edited"
)

CACHE_FILE = (
    BASE_DIR /
    "embedding_cache.json"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
}

MODEL_NAME = (
    "openai/clip-vit-base-patch32"
)


CATALOG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

EDITED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    f"Using device: {device}"
)


# ============================================================
# CLIP
# ============================================================

processor = (
    CLIPProcessor
    .from_pretrained(
        MODEL_NAME
    )
)

model = (
    CLIPModel
    .from_pretrained(
        MODEL_NAME
    )
    .to(device)
)

model.eval()


# ============================================================
# LABELS
# ============================================================

STYLE_LABELS = [
    "Streetwear Minimal",
    "Modern Luxury",
    "Casual Chic",
    "Sporty",
    "Classic Elegant",
    "Bohemian",
    "Vintage",
    "Business Formal",
]


CATEGORY_LABELS = [
    "dress",
    "jacket",
    "coat",
    "shirt",
    "t-shirt",
    "blouse",
    "sweater",
    "hoodie",
    "trousers",
    "jeans",
    "skirt",
    "shorts",
    "shoes",
    "bag",
    "fashion accessory",
]


COLOR_LABELS = [
    "black",
    "white",
    "grey",
    "beige",
    "brown",
    "red",
    "blue",
    "green",
    "yellow",
    "pink",
    "purple",
    "orange",
]


SEASON_LABELS = [
    "spring",
    "summer",
    "autumn",
    "winter",
]


OCCASION_LABELS = [
    "casual everyday",
    "office work",
    "formal event",
    "party",
    "vacation",
    "sport",
    "evening event",
]


# ============================================================
# CATALOG ITEM
# ============================================================

class CatalogItem:

    def __init__(
        self,
        item_id,
        filename,
        filepath,
        image_url,
        embedding,
    ):

        self.id = item_id

        self.filename = (
            filename
        )

        self.filepath = str(
            filepath
        )

        self.image_url = (
            image_url
        )

        self.embedding = (
            np.asarray(
                embedding,
                dtype=np.float32,
            )
        )


    def to_dict(self):

        return {
            "id":
                self.id,

            "filename":
                self.filename,

            "image_url":
                self.image_url,
        }


catalog = []


# ============================================================
# EMBEDDING HELPERS
# ============================================================

def _norm(
    tensor,
):

    tensor = tensor / (
        tensor.norm(
            dim=-1,
            keepdim=True,
        )
        + 1e-12
    )

    return (
        tensor
        .squeeze(0)
        .detach()
        .cpu()
        .numpy()
        .astype(
            np.float32
        )
    )


@torch.inference_mode()
def get_text_embedding(
    text,
):

    inputs = processor(
        text=[text],
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    input_ids = (
        inputs["input_ids"]
        .to(device)
    )

    attention_mask = (
        inputs["attention_mask"]
        .to(device)
    )


    outputs = (
        model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
    )


    embedding = (
        model.text_projection(
            outputs.pooler_output
        )
    )


    return _norm(
        embedding
    )


@torch.inference_mode()
def get_image_embedding(
    image,
):

    should_close = False


    if isinstance(
        image,
        (str, Path),
    ):

        image = Image.open(
            image
        )

        should_close = True


    try:

        image = (
            image
            .convert("RGB")
        )


        inputs = processor(
            images=image,
            return_tensors="pt",
        )


        pixel_values = (
            inputs["pixel_values"]
            .to(device)
        )


        outputs = (
            model.vision_model(
                pixel_values=
                    pixel_values
            )
        )


        embedding = (
            model.visual_projection(
                outputs.pooler_output
            )
        )


        return _norm(
            embedding
        )

    finally:

        if should_close:
            image.close()


# ============================================================
# CACHE
# ============================================================

def load_cache():

    if not CACHE_FILE.exists():
        return {}

    try:

        with CACHE_FILE.open(
            "r",
            encoding="utf-8",
        ) as file:

            data = json.load(
                file
            )

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception as exc:

        print(
            "Cache load warning:",
            exc,
        )

    return {}


def save_cache(
    cache,
):

    with CACHE_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            cache,
            file,
        )


def file_signature(
    path,
):

    stat = path.stat()

    raw = (
        f"{path.name}|"
        f"{stat.st_size}|"
        f"{stat.st_mtime_ns}"
    )

    return (
        hashlib
        .sha256(
            raw.encode(
                "utf-8"
            )
        )
        .hexdigest()
    )


# ============================================================
# CATALOG SYNC
# ============================================================

def sync_catalog():

    cache = load_cache()

    new_catalog = []

    existing_names = set()


    files = sorted(
        [
            path
            for path
            in CATALOG_DIR.iterdir()

            if (
                path.is_file()
                and
                path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        ],

        key=lambda p:
            p.name.lower(),
    )


    for index, path in enumerate(
        files,
        start=1,
    ):

        existing_names.add(
            path.name
        )

        signature = (
            file_signature(
                path
            )
        )

        cached = cache.get(
            path.name
        )


        embedding = None


        if (
            isinstance(
                cached,
                dict,
            )
            and
            cached.get(
                "signature"
            )
            == signature
            and
            cached.get(
                "embedding"
            )
            is not None
        ):

            try:

                embedding = (
                    np.asarray(
                        cached[
                            "embedding"
                        ],
                        dtype=np.float32,
                    )
                )

            except Exception:
                embedding = None


        if embedding is None:

            print(
                "Creating embedding:",
                path.name,
            )

            embedding = (
                get_image_embedding(
                    path
                )
            )

            cache[path.name] = {
                "signature":
                    signature,

                "embedding":
                    embedding.tolist(),
            }


        image_url = (
            "/static/catalog/"
            + quote(
                path.name
            )
        )


        new_catalog.append(
            CatalogItem(
                item_id=index,
                filename=path.name,
                filepath=path,
                image_url=image_url,
                embedding=embedding,
            )
        )


    # Remove stale cache records.

    stale_names = [
        filename
        for filename
        in cache.keys()

        if filename
        not in existing_names
    ]


    for filename in stale_names:

        cache.pop(
            filename,
            None,
        )


    # IMPORTANT:
    # mutate instead of reassigning
    # because api.py imports catalog.

    catalog.clear()

    catalog.extend(
        new_catalog
    )


    save_cache(
        cache
    )


    print(
        "Catalog synchronized:",
        len(catalog),
        "items",
    )


    return catalog


# ============================================================
# GET ITEM
# ============================================================

def get_item(
    item_id,
):

    for item in catalog:

        if (
            int(item.id)
            ==
            int(item_id)
        ):

            return item

    return None


# ============================================================
# CLIP LABEL RANKING
# ============================================================

def rank_prompts(
    vector,
    labels,
    template="a fashion photograph of {}",
):

    scores = []


    for label in labels:

        text_vector = (
            get_text_embedding(
                template.format(
                    label
                )
            )
        )


        similarity = float(
            np.dot(
                vector,
                text_vector,
            )
        )


        scores.append(
            (
                label,
                similarity,
            )
        )


    scores.sort(
        key=lambda x:
            x[1],

        reverse=True,
    )


    return scores


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_embedding(
    vector,
):

    styles = rank_prompts(
        vector,
        STYLE_LABELS,
    )

    categories = rank_prompts(
        vector,
        CATEGORY_LABELS,
    )

    colors = rank_prompts(
        vector,
        COLOR_LABELS,
    )

    seasons = rank_prompts(
        vector,
        SEASON_LABELS,
    )

    occasions = rank_prompts(
        vector,
        OCCASION_LABELS,
    )


    return {
        "style":
            styles[0][0],

        "style_score":
            round(
                styles[0][1]
                * 100,
                1,
            ),

        "category":
            categories[0][0],

        "category_score":
            round(
                categories[0][1]
                * 100,
                1,
            ),

        "color":
            colors[0][0],

        "season":
            seasons[0][0],

        "occasion":
            occasions[0][0],

        "top_styles": [
            {
                "label":
                    label,

                "score":
                    round(
                        score * 100,
                        1,
                    ),
            }

            for label, score
            in styles[:3]
        ],
    }


# ============================================================
# IMAGE SIMILARITY
# ============================================================

def find_similar(
    target_embedding,
    exclude_id=None,
    limit=4,
    min_similarity=0.45,
):

    matches = []


    for item in catalog:

        if (
            exclude_id is not None
            and
            int(item.id)
            ==
            int(exclude_id)
        ):

            continue


        similarity = float(
            np.dot(
                target_embedding,
                item.embedding,
            )
        )


        if (
            similarity
            <
            min_similarity
        ):
            continue


        matches.append(
            {
                "id":
                    item.id,

                "filename":
                    item.filename,

                "image_url":
                    item.image_url,

                "score":
                    round(
                        similarity
                        * 100,
                        1,
                    ),

                "_similarity":
                    similarity,
            }
        )


    matches.sort(
        key=lambda x:
            x["_similarity"],

        reverse=True,
    )


    matches = (
        matches[:limit]
    )


    for match in matches:

        match.pop(
            "_similarity",
            None,
        )


    return matches


# ============================================================
# TEXT SEARCH
# ============================================================

def text_search(
    query,
    limit=8,
    min_score=0.22,
):

    query = query.strip()


    if not query:
        return []


    # Multiple CLIP prompts improve retrieval
    # compared with sending conversational
    # language directly.

    prompts = [
        f"a fashion photograph of {query}",
        f"a clothing item featuring {query}",
        f"a stylish outfit with {query}",
    ]


    vectors = [
        get_text_embedding(
            prompt
        )

        for prompt
        in prompts
    ]


    query_embedding = (
        np.mean(
            vectors,
            axis=0,
        )
    )


    norm = np.linalg.norm(
        query_embedding
    )


    if norm > 0:

        query_embedding = (
            query_embedding
            / norm
        )


    matches = []


    for item in catalog:

        similarity = float(
            np.dot(
                query_embedding,
                item.embedding,
            )
        )


        if (
            similarity
            <
            min_score
        ):

            continue


        matches.append(
            {
                "id":
                    item.id,

                "filename":
                    item.filename,

                "image_url":
                    item.image_url,

                "score":
                    round(
                        similarity
                        * 100,
                        1,
                    ),

                "_similarity":
                    similarity,
            }
        )


    matches.sort(
        key=lambda x:
            x["_similarity"],

        reverse=True,
    )


    matches = (
        matches[:limit]
    )


    for match in matches:

        match.pop(
            "_similarity",
            None,
        )


    return matches


# ============================================================
# OUTFIT BUILDER
# ============================================================

def build_outfit(
    item,
):

    analysis = (
        analyze_embedding(
            item.embedding
        )
    )


    category = (
        analysis["category"]
    )

    style = (
        analysis["style"]
    )

    color = (
        analysis["color"]
    )


    if category in {
        "shirt",
        "t-shirt",
        "blouse",
        "sweater",
        "hoodie",
    }:

        query = (
            f"{style} trousers skirt "
            f"jeans shoes that complement "
            f"a {color} {category}"
        )


    elif category in {
        "trousers",
        "jeans",
        "skirt",
        "shorts",
    }:

        query = (
            f"{style} shirt blouse jacket "
            f"top shoes that complement "
            f"{color} {category}"
        )


    elif category in {
        "dress",
    }:

        query = (
            f"{style} shoes bag jacket "
            f"accessories for a "
            f"{color} dress"
        )


    elif category in {
        "jacket",
        "coat",
    }:

        query = (
            f"{style} shirt trousers jeans "
            f"shoes that complement "
            f"a {color} {category}"
        )


    elif category == "shoes":

        query = (
            f"{style} clothing outfit "
            f"that complements "
            f"{color} shoes"
        )


    elif category == "bag":

        query = (
            f"{style} outfit dress clothing "
            f"that complements "
            f"a {color} bag"
        )


    else:

        query = (
            f"{style} coordinated outfit "
            f"for {color} {category}"
        )


    recommendations = (
        text_search(
            query,
            limit=6,
            min_score=0.20,
        )
    )


    recommendations = [
        recommendation

        for recommendation
        in recommendations

        if (
            int(
                recommendation[
                    "id"
                ]
            )
            !=
            int(
                item.id
            )
        )
    ]


    return {
        "analysis":
            analysis,

        "query":
            query,

        "recommendations":
            recommendations[:4],
    }


# ============================================================
# IMAGE EDITING
# ============================================================

def edit_image(
    item,
    operation,
):

    operation = (
        operation
        .lower()
        .strip()
    )


    with Image.open(
        item.filepath
    ) as source:

        image = (
            source
            .convert("RGB")
        )


        if operation == "bright":

            result = (
                ImageEnhance
                .Brightness(
                    image
                )
                .enhance(
                    1.35
                )
            )


        elif operation == "contrast":

            result = (
                ImageEnhance
                .Contrast(
                    image
                )
                .enhance(
                    1.35
                )
            )


        elif operation in {
            "black_and_white",
            "grayscale",
        }:

            result = (
                ImageOps
                .grayscale(
                    image
                )
                .convert(
                    "RGB"
                )
            )


        elif operation == "sharpen":

            result = (
                image.filter(
                    ImageFilter.SHARPEN
                )
            )


        elif operation == "blur":

            result = (
                image.filter(
                    ImageFilter.GaussianBlur(
                        radius=2
                    )
                )
            )


        elif operation == "rotate_right":

            result = (
                image.rotate(
                    -90,
                    expand=True,
                )
            )


        elif operation == "rotate_left":

            result = (
                image.rotate(
                    90,
                    expand=True,
                )
            )


        elif operation == "flip_horizontal":

            result = (
                ImageOps.mirror(
                    image
                )
            )


        elif operation == "flip_vertical":

            result = (
                ImageOps.flip(
                    image
                )
            )


        else:

            return {
                "success":
                    False,

                "message":
                    (
                        "That image edit is not "
                        "supported yet."
                    ),
            }


        safe_stem = (
            Path(
                item.filename
            )
            .stem
        )


        output_name = (
            f"{safe_stem}_"
            f"{operation}.jpg"
        )


        output_path = (
            EDITED_DIR /
            output_name
        )


        result.save(
            output_path,
            format="JPEG",
            quality=95,
        )


    return {
        "success":
            True,

        "message":
            (
                f"Applied {operation.replace('_', ' ')}."
            ),

        "image_url":
            (
                "/static/edited/"
                +
                quote(
                    output_name
                )
            ),
    }


# ============================================================
# HELP
# ============================================================

def help_response():

    return {
        "type":
            "help",

        "message":
            (
                "I can search your fashion catalog "
                "using natural language, analyze a "
                "selected item's style, category, "
                "color, season and occasion, find "
                "similar pieces, build outfit "
                "suggestions, and apply basic image "
                "edits such as brightness, contrast, "
                "black and white, sharpening, blur, "
                "rotation and flipping."
            ),

        "items": [],
    }


# ============================================================
# NO SELECTION
# ============================================================

def selection_required_response():

    return {
        "type":
            "selection_required",

        "message":
            (
                "Select an image from your catalog "
                "first, then send that request again."
            ),

        "items": [],
    }


# ============================================================
# CHAT / GROK ORCHESTRATION
# ============================================================

def chat_reply(
    message,
    item_id=None,
):

    message = (
        message
        .strip()
    )


    if not message:

        return {
            "type":
                "error",

            "message":
                "Please enter a request.",

            "items": [],
        }


    route = (
        understand_request(
            message
        )
    )


    print(
        "Fashion Copilot route:",
        route,
    )


    intent = (
        route.get(
            "intent",
            "search",
        )
    )


    requires_image = (
        route.get(
            "needs_selected_image",
            False,
        )
    )


    item = None


    if item_id is not None:

        item = get_item(
            item_id
        )


    if (
        requires_image
        and
        item is None
    ):

        return (
            selection_required_response()
        )


    # --------------------------------------------------------
    # HELP
    # --------------------------------------------------------

    if intent == "help":

        return help_response()


    # --------------------------------------------------------
    # UNSUPPORTED
    # --------------------------------------------------------

    if intent == "unsupported":

        return {
            "type":
                "unsupported",

            "message":
                route.get(
                    "response"
                )
                or
                (
                    "That requires generative image "
                    "editing, which is not enabled "
                    "in the current version."
                ),

            "items": [],
        }


    # --------------------------------------------------------
    # ANALYZE
    # --------------------------------------------------------

    if intent == "analyze":

        analysis = (
            analyze_embedding(
                item.embedding
            )
        )


        return {
            "type":
                "analysis",

            "message":
                (
                    "I analyzed the selected "
                    "fashion image using CLIP."
                ),

            "analysis":
                analysis,

            "items": [],
        }


    # --------------------------------------------------------
    # SIMILAR
    # --------------------------------------------------------

    if intent == "similar":

        matches = (
            find_similar(
                item.embedding,
                exclude_id=item.id,
                limit=6,
            )
        )


        if not matches:

            return {
                "type":
                    "similar",

                "message":
                    (
                        "I couldn't find a sufficiently "
                        "similar item in the current "
                        "catalog."
                    ),

                "items": [],
            }


        return {
            "type":
                "similar",

            "message":
                (
                    f"I found {len(matches)} "
                    "similar catalog "
                    "item"
                    + (
                        "s."
                        if len(matches) != 1
                        else "."
                    )
                ),

            "items":
                matches,
        }


    # --------------------------------------------------------
    # ANALYZE + SIMILAR
    # --------------------------------------------------------

    if (
        intent
        ==
        "analyze_and_similar"
    ):

        analysis = (
            analyze_embedding(
                item.embedding
            )
        )


        matches = (
            find_similar(
                item.embedding,
                exclude_id=item.id,
                limit=6,
            )
        )


        return {
            "type":
                "analysis_and_similar",

            "message":
                (
                    "I analyzed the selected item "
                    "and searched the catalog for "
                    "similar pieces."
                ),

            "analysis":
                analysis,

            "items":
                matches,
        }


    # --------------------------------------------------------
    # OUTFIT
    # --------------------------------------------------------

    if intent == "outfit":

        result = (
            build_outfit(
                item
            )
        )


        recommendations = (
            result[
                "recommendations"
            ]
        )


        if recommendations:

            message_text = (
                "I analyzed the selected piece "
                "and found catalog items that may "
                "work with it."
            )

        else:

            message_text = (
                "I analyzed the selected piece, "
                "but your current catalog does not "
                "contain strong complementary "
                "matches yet."
            )


        return {
            "type":
                "outfit",

            "message":
                message_text,

            "analysis":
                result[
                    "analysis"
                ],

            "items":
                recommendations,
        }


    # --------------------------------------------------------
    # ANALYZE + OUTFIT
    # --------------------------------------------------------

    if (
        intent
        ==
        "analyze_and_outfit"
    ):

        result = (
            build_outfit(
                item
            )
        )


        return {
            "type":
                "analysis_and_outfit",

            "message":
                (
                    "I analyzed the selected item "
                    "and built the closest outfit "
                    "combination available in your "
                    "catalog."
                ),

            "analysis":
                result[
                    "analysis"
                ],

            "items":
                result[
                    "recommendations"
                ],
        }


    # --------------------------------------------------------
    # EDIT
    # --------------------------------------------------------

    if intent == "edit":

        operation = (
            route.get(
                "edit_operation"
            )
        )


        if not operation:

            return {
                "type":
                    "edit",

                "message":
                    (
                        "I understood that you want "
                        "to edit the image, but the "
                        "requested edit is not "
                        "supported."
                    ),

                "items": [],
            }


        result = (
            edit_image(
                item,
                operation,
            )
        )


        return {
            "type":
                "edit",

            "message":
                result.get(
                    "message",
                    "Edit complete.",
                ),

            "image_url":
                result.get(
                    "image_url"
                ),

            "items": [],
        }


    # --------------------------------------------------------
    # SEARCH
    # --------------------------------------------------------

    query = (
        route.get(
            "query"
        )
        or message
    )


    results = (
        text_search(
            query,
            limit=8,
            min_score=0.22,
        )
    )


    if not results:

        return {
            "type":
                "no_match",

            "message":
                (
                    f'I could not find a strong '
                    f'match for "{query}" in your '
                    f'current catalog. The catalog '
                    f'may not contain that type of '
                    f'fashion item yet.'
                ),

            "query":
                query,

            "items": [],
        }


    best_score = (
        results[0][
            "score"
        ]
    )


    if best_score >= 30:

        confidence_text = (
            "I found relevant catalog matches."
        )

    else:

        confidence_text = (
            "I found some possible catalog matches, "
            "but their CLIP similarity is moderate."
        )


    return {
        "type":
            "search",

        "message":
            (
                f'{confidence_text} '
                f'Search: "{query}".'
            ),

        "query":
            query,

        "items":
            results,
    }


# ============================================================
# INITIAL SYNC
# ============================================================

sync_catalog()


print(
    "Grok LLM:",
    (
        "enabled"
        if llm_available()
        else
        "not configured - fallback router active"
    ),
)
