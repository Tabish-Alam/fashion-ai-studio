import json
import re
import shutil

from pathlib import Path

from fastapi import (
    FastAPI,
    UploadFile,
    File,
    HTTPException,
)

from fastapi.middleware.cors import (
    CORSMiddleware,
)

from fastapi.responses import (
    FileResponse,
)

from fastapi.staticfiles import (
    StaticFiles,
)

from pydantic import (
    BaseModel,
)

from main import (
    BASE_DIR,
    CACHE_FILE,
    CATALOG_DIR,
    catalog,
    sync_catalog,
    get_item,
    analyze_embedding,
    find_similar,
    text_search,
    build_outfit,
    chat_reply,
)


# ============================================================
# PATHS
# ============================================================

STATIC_DIR = (
    BASE_DIR /
    "static"
)


# ============================================================
# APP
# ============================================================

app = FastAPI(
    title="Fashion AI Studio",
    description=(
        "Grok + CLIP Fashion Copilot"
    ),
    version="4.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount(
    "/static",
    StaticFiles(
        directory=STATIC_DIR
    ),
    name="static",
)


# ============================================================
# REQUEST MODELS
# ============================================================

class ChatRequest(
    BaseModel
):

    message: str

    item_id: int | None = None


class SearchRequest(
    BaseModel
):

    query: str

    limit: int = 8


# ============================================================
# FRONTEND
# ============================================================

@app.get("/")
def home():

    index_file = (
        STATIC_DIR /
        "index.html"
    )


    if not index_file.exists():

        raise HTTPException(
            status_code=404,
            detail=(
                "index.html not found"
            ),
        )


    return FileResponse(
        index_file
    )


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status":
            "ok",

        "catalog_items":
            len(catalog),
    }


# ============================================================
# CATALOG
# ============================================================

@app.get("/api/items")
def items():

    sync_catalog()


    return [
        item.to_dict()

        for item
        in catalog
    ]


# ============================================================
# UPLOAD
# ============================================================

@app.post("/api/upload")
async def upload(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="Filename missing.",
        )


    extension = (
        Path(
            file.filename
        )
        .suffix
        .lower()
    )


    allowed = {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }


    if extension not in allowed:

        raise HTTPException(
            status_code=400,
            detail=(
                "Only JPG, JPEG, PNG "
                "and WEBP images are supported."
            ),
        )


    safe_filename = re.sub(
        r"[^\w\.-]",
        "_",
        file.filename,
    )


    destination = (
        CATALOG_DIR /
        safe_filename
    )


    # Avoid overwriting existing images.

    if destination.exists():

        stem = (
            destination.stem
        )

        suffix = (
            destination.suffix
        )

        counter = 1


        while destination.exists():

            destination = (
                CATALOG_DIR /
                f"{stem}_{counter}{suffix}"
            )

            counter += 1


    try:

        with destination.open(
            "wb"
        ) as buffer:

            shutil.copyfileobj(
                file.file,
                buffer,
            )

    finally:

        await file.close()


    sync_catalog()


    uploaded_item = None


    for item in catalog:

        if (
            item.filename
            ==
            destination.name
        ):

            uploaded_item = item

            break


    if uploaded_item is None:

        raise HTTPException(
            status_code=500,
            detail=(
                "Image was saved but "
                "could not be indexed."
            ),
        )


    return {
        "success":
            True,

        "message":
            "Image uploaded and indexed.",

        "item":
            uploaded_item.to_dict(),
    }


# ============================================================
# DELETE
# ============================================================

@app.delete(
    "/api/items/{item_id}"
)
def delete_item(
    item_id: int
):

    sync_catalog()


    item = get_item(
        item_id
    )


    if item is None:

        raise HTTPException(
            status_code=404,
            detail=(
                "Catalog item not found."
            ),
        )


    filepath = Path(
        item.filepath
    )

    filename = (
        item.filename
    )


    # Ensure the path actually belongs
    # to the catalog directory.

    try:

        resolved_file = (
            filepath.resolve()
        )

        resolved_catalog = (
            CATALOG_DIR.resolve()
        )


        resolved_file.relative_to(
            resolved_catalog
        )

    except ValueError:

        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid catalog file path."
            ),
        )


    if filepath.exists():

        try:

            filepath.unlink()

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=(
                    f"Could not delete image: "
                    f"{exc}"
                ),
            )


    # Clean embedding cache.

    if CACHE_FILE.exists():

        try:

            with CACHE_FILE.open(
                "r",
                encoding="utf-8",
            ) as file:

                cache = (
                    json.load(
                        file
                    )
                )


            cache.pop(
                filename,
                None,
            )


            with CACHE_FILE.open(
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    cache,
                    file,
                )

        except Exception as exc:

            print(
                "Cache cleanup warning:",
                exc,
            )


    sync_catalog()


    return {
        "success":
            True,

        "message":
            "Image removed from catalog.",

        "deleted_filename":
            filename,

        "catalog_items":
            len(catalog),
    }


# ============================================================
# CHAT
# ============================================================

@app.post("/api/chat")
def chat(
    req: ChatRequest
):

    sync_catalog()


    try:

        return chat_reply(
            req.message,
            req.item_id,
        )

    except Exception as exc:

        print(
            "CHAT ERROR:",
            exc,
        )


        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# RECOMMEND
# ============================================================

@app.post(
    "/api/recommend/{item_id}"
)
def recommend(
    item_id: int
):

    sync_catalog()


    item = get_item(
        item_id
    )


    if item is None:

        raise HTTPException(
            status_code=404,
            detail="Item not found.",
        )


    analysis = (
        analyze_embedding(
            item.embedding
        )
    )


    matches = (
        find_similar(
            item.embedding,
            exclude_id=item.id,
            limit=4,
        )
    )


    return {
        "item":
            item.to_dict(),

        "analysis":
            analysis,

        "recommendations":
            matches,

        "commentary":
            (
                f"This item is closest to "
                f"{analysis['style']} and "
                f"is classified as "
                f"{analysis['category']}."
            ),
    }


# ============================================================
# SEARCH
# ============================================================

@app.post("/api/search")
def search(
    req: SearchRequest
):

    sync_catalog()


    query = (
        req.query
        .strip()
    )


    if not query:

        raise HTTPException(
            status_code=400,
            detail=(
                "Search query is required."
            ),
        )


    limit = max(
        1,
        min(
            req.limit,
            20,
        ),
    )


    results = (
        text_search(
            query,
            limit=limit,
        )
    )


    return {
        "query":
            query,

        "items":
            results,
    }


# ============================================================
# OUTFIT
# ============================================================

@app.post(
    "/api/outfit/{item_id}"
)
def outfit(
    item_id: int
):

    sync_catalog()


    item = get_item(
        item_id
    )


    if item is None:

        raise HTTPException(
            status_code=404,
            detail="Item not found.",
        )


    return build_outfit(
        item
    )
