# Fashion AI Studio v2 — Features Added

- Text-to-fashion semantic search (`GET /api/search`)
- Temporary uploaded-image visual search (`POST /api/visual-search`)
- Rich CLIP zero-shot analysis: style, category, color, occasion, season
- Smart Outfit Builder (`POST /api/outfit/{item_id}`)
- Per-recommendation explanations
- Find Similar flow retained and improved
- Persistent embedding cache (`embedding_cache.npz`) to avoid recomputing unchanged catalog images
- Updated responsive UI for all new workflows

## Run
From the project directory:

    uvicorn api:app --reload

The first run still needs access to the HuggingFace CLIP model unless it is already cached locally.
