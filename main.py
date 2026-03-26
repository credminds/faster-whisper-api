import json
import tempfile
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
import nemo.collections.asr as nemo_asr
import soundfile as sf


@asynccontextmanager
async def lifespan(app: FastAPI):
    asr_model = nemo_asr.models.EncDecRNNTBPEModel.from_pretrained(
        model_name="nvidia/parakeet-tdt-1.1b"
    )
    asr_model.change_attention_model(
        "rel_pos_local_attn", att_context_size=[256, 256]
    )
    app.state.model = asr_model
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
        output = app.state.model.transcribe([tmp_path], timestamps=True)
        result = output[0]

        full_text = result.text
        segments = []
        if result.timestamp and "segment" in result.timestamp:
            for ts in result.timestamp["segment"]:
                segments.append({
                    "start": ts["start"],
                    "end": ts["end"],
                    "text": ts["segment"].strip(),
                })

        audio_info = sf.info(tmp_path)
        duration = audio_info.duration

        return {
            "text": full_text,
            "segments": segments,
            "language": "en",
            "duration": duration,
        }
    finally:
        os.unlink(tmp_path)


@app.post("/transcribe/stream")
async def transcribe_stream(file: UploadFile = File(...)):
    tmp_path = save_upload_to_temp(file)

    def generate():
        try:
            output = app.state.model.transcribe([tmp_path], timestamps=True)
            result = output[0]

            audio_info = sf.info(tmp_path)
            duration = audio_info.duration

            if result.timestamp and "segment" in result.timestamp:
                for ts in result.timestamp["segment"]:
                    data = json.dumps({
                        "start": ts["start"],
                        "end": ts["end"],
                        "text": ts["segment"].strip(),
                    })
                    yield f"data: {data}\n\n"

            done = json.dumps({"done": True, "language": "en", "duration": duration})
            yield f"data: {done}\n\n"
        finally:
            os.unlink(tmp_path)

    return StreamingResponse(generate(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8010)
