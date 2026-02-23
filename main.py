import json
import tempfile
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from faster_whisper import WhisperModel


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = WhisperModel("medium", device="auto", compute_type="default")
    yield


app = FastAPI(title="Speech-to-Text Service", lifespan=lifespan)


def save_upload_to_temp(upload: UploadFile) -> str:
    suffix = os.path.splitext(upload.filename or ".wav")[1]
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(upload.file.read())
    tmp.close()
    return tmp.name


@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    tmp_path = save_upload_to_temp(file)
    try:
        segments_gen, info = app.state.model.transcribe(tmp_path)
        segments = []
        full_text_parts = []
        for seg in segments_gen:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
            full_text_parts.append(seg.text.strip())
        return {
            "text": " ".join(full_text_parts),
            "segments": segments,
            "language": info.language,
            "duration": info.duration,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe/stream")
async def transcribe_stream(file: UploadFile = File(...)):
    tmp_path = save_upload_to_temp(file)

    def generate():
        try:
            segments_gen, info = app.state.model.transcribe(tmp_path)
            for seg in segments_gen:
                data = json.dumps({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
                yield f"data: {data}\n\n"
            done = json.dumps({"done": True, "language": info.language, "duration": info.duration})
            yield f"data: {done}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8004)
