# Fashion AI Studio v3

Free conversational upgrade with no paid API and no additional generative model.

## Added
- Fashion Copilot chat interface
- Natural-language intent routing
- Chat-driven CLIP catalog search
- Chat-driven image analysis
- Chat-driven similar-item retrieval
- Chat-driven outfit recommendations
- Lightweight image edits using Pillow: brightness, contrast, B&W, sharpen, blur, rotate, flip
- Modern glass/dark responsive UI
- Active-image conversational context

## Run
`source venv/bin/activate` then `uvicorn api:app --reload`

Note: CLIP still requires its existing Hugging Face model files. Image editing here is deterministic Pillow processing, not generative editing.
