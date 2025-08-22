import os
import subprocess
import tempfile
import logging
import uuid
import io
import sys
from fastapi import FastAPI, Form, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import APIKeyHeader
from concurrent.futures import ThreadPoolExecutor
import threading
from pydub import AudioSegment
from text_preprocess import preprocess_text, _NLP  # Import _NLP for segmentation

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_api_key(api_key: str = Depends(api_key_header)):
    expected_key = os.getenv("API_KEY")
    if not expected_key or api_key != expected_key:
        raise HTTPException(status_code=401, detail="Invalid or missing API Key")
    return api_key

# Thread pool for CPU-intensive operations
executor = ThreadPoolExecutor(max_workers=2)

# In-memory job storage (since no Redis)
jobs = {}
jobs_lock = threading.Lock()

# Predefined voices mapping
VOICES_DIR = "voices"  # Directory where voice files are stored
PREDEFINED_VOICES = {
    "voice_one": {
        "wav_path": os.path.join(VOICES_DIR, "voice_one.wav"),
        "transcription": "Some call me nature. Others call me mother nature. I've been here for over 4.5 billions years. Twenty-two-thousand-five-hundred times longer than you."
    },
    "voice_two": {
        "wav_path": os.path.join(VOICES_DIR, "voice_two.wav"),
        "transcription": "This is the transcription for voice two"
    },
    # Add more voices as needed
}

def run_inference(cmd):
    """Run the inference command and handle output"""
    logger.info(f"Executing command: {' '.join(cmd)}")  # Added log for command
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise Exception(f"Inference failed: {result.stderr}")
    elif result.stderr:
        logger.warning(f"Inference warnings: {result.stderr}")
    logger.info("Inference command completed successfully")

def segment_text(text, max_chars=400):
    """
    Segment text into chunks based on natural boundaries using spaCy for accurate splitting.
    Returns list of chunks, preserving paragraph structure for pause decisions.
    max_chars: approximate character limit per chunk (adjust based on TTS performance; ~400 chars ≈ 20-30s audio)
    """
    doc = _NLP(text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sent in doc.sents:
        sentence = sent.text.strip()
        if not sentence:
            continue
        
        if current_length + len(sentence) <= max_chars:
            current_chunk.append(sentence)
            current_length += len(sentence) + 1  # +1 for space
        else:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_length = len(sentence)
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    logger.debug(f"Segmented text into {len(chunks)} chunks")
    return chunks

def inference_task(job_id, voice_info, text, model_name, use_onnx, voice_name):
    """Task function to run in thread with chunking support"""
    logger.info(f"[{job_id}] Entered inference_task function")  # Added: Log entry to task

    try:
        logger.info(f"[{job_id}] Attempting to import torch")
        import torch
        logger.info(f"[{job_id}] Imported torch successfully")

        logger.info(f"[{job_id}] Attempting to import preprocess_text (this may load models)")
        from text_preprocess import preprocess_text
        logger.info(f"[{job_id}] Imported preprocess_text successfully (models should be loaded)")
    except Exception as e:
        logger.error(f"[{job_id}] Import failure: {str(e)}")
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['exc_info'] = str(e)
        return  # Exit task early

    logger.info(f"[{job_id}] Starting inference task with chunking")  # Existing log

    with jobs_lock:
        jobs[job_id]['status'] = 'started'
        jobs[job_id]['progress'] = 'Preprocessing text'
    logger.info(f"[{job_id}] Job status set to 'started'")

    # Preprocess the entire text first
    try:
        original_text = text
        logger.info(f"[{job_id}] Starting text preprocessing")
        text = preprocess_text(text)
        logger.info(f"[{job_id}] Preprocessed text: '{text[:50]}...' (original: '{original_text[:50]}...')")
    except Exception as e:
        logger.warning(f"[{job_id}] Text preprocessing failed: {str(e)}. Using original text.")
        text = original_text

    # Segment the text into chunks
    chunks = segment_text(text)
    logger.info(f"[{job_id}] Split text into {len(chunks)} chunks")
    
    with jobs_lock:
        jobs[job_id]['total_chunks'] = len(chunks)
        jobs[job_id]['progress'] = f"0/{len(chunks)} chunks processed"

    # Generate audio for each chunk
    audio_chunks = []
    
    for i, chunk in enumerate(chunks):
        logger.info(f"[{job_id}] Processing chunk {i+1}/{len(chunks)}: '{chunk[:50]}...'")
        
        fd, res_wav_path = tempfile.mkstemp(suffix='.wav')
        try:
            # Build the inference command for this chunk
            module = "zipvoice.bin.infer_zipvoice_onnx" if use_onnx else "zipvoice.bin.infer_zipvoice"
            cmd = [
                sys.executable, "-m", module,
                "--model-name", model_name,
                "--prompt-text", voice_info["transcription"],
                "--prompt-wav", voice_info["wav_path"],
                "--text", chunk,
                "--res-wav-path", res_wav_path
            ]

            cuda_available = torch.cuda.is_available()
            if cuda_available:
                if use_onnx:
                    logger.info(f"[{job_id}] CUDA available, but ONNX path defaults to CPU (no provider flag supported)")
                else:
                    logger.info(f"[{job_id}] CUDA available; PyTorch path should use it automatically")
            else:
                logger.info(f"[{job_id}] CUDA not available, using CPU")

            logger.info(f"[{job_id}] Running inference for chunk {i+1}")
            run_inference(cmd)

            logger.info(f"[{job_id}] Loading generated WAV for chunk {i+1}")
            audio_segment = AudioSegment.from_wav(res_wav_path)
            audio_chunks.append(audio_segment)
            
            # Update progress
            with jobs_lock:
                jobs[job_id]['progress'] = f"{i+1}/{len(chunks)} chunks processed"
                
        except Exception as e:
            logger.error(f"[{job_id}] Chunk {i+1} failed: {str(e)}")
            # If a chunk fails, use a short silence as placeholder
            audio_chunks.append(AudioSegment.silent(duration=1000))  # 1 second silence
        finally:
            os.close(fd)
            if os.path.exists(res_wav_path):
                os.remove(res_wav_path)
            logger.debug(f"[{job_id}] Cleaned up temp file for chunk {i+1}")

    # Combine all audio chunks with pauses (250ms between sentences for flow)
    logger.info(f"[{job_id}] Combining {len(audio_chunks)} audio chunks")
    
    if not audio_chunks:
        logger.error(f"[{job_id}] No audio chunks generated")
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['exc_info'] = "No audio chunks generated"
        return
        
    # Start with the first chunk
    combined_audio = audio_chunks[0]
    
    # Add subsequent chunks with pauses
    for i in range(1, len(audio_chunks)):
        pause_duration = 250  # milliseconds; could vary if paragraph detection added
        logger.debug(f"[{job_id}] Adding {pause_duration}ms pause before chunk {i+1}")
        combined_audio += AudioSegment.silent(duration=pause_duration)
        combined_audio += audio_chunks[i]
    
    # Export to bytes
    try:
        buffer = io.BytesIO()
        combined_audio.export(buffer, format="wav")
        wav_bytes = buffer.getvalue()
    except Exception as e:
        logger.error(f"[{job_id}] Audio export failed: {str(e)}")
        with jobs_lock:
            jobs[job_id]['status'] = 'failed'
            jobs[job_id]['exc_info'] = str(e)
        return

    with jobs_lock:
        jobs[job_id]['status'] = 'finished'
        jobs[job_id]['wav_bytes'] = wav_bytes
        jobs[job_id]['voice_name'] = voice_name
    logger.info(f"[{job_id}] Audio generation completed successfully")


@app.post("/tts", status_code=status.HTTP_202_ACCEPTED)
async def generate_tts(
    voice_name: str = Form(..., description="Name of the predefined voice"),
    text: str = Form(..., description="Text to synthesize"),
    model_name: str = Form(default="zipvoice_distill", description="Model name: zipvoice or zipvoice_distill"),
    use_onnx: bool = Form(default=True, description="Use ONNX for faster CPU inference"),
    api_key: str = Depends(get_api_key)
):
    request_id = str(uuid.uuid4())[:8]
    logger.info(f"[{request_id}] Request started: voice_name={voice_name}, text='{text[:50]}...', model={model_name}, onnx={use_onnx}")

    # Check if voice exists
    if voice_name not in PREDEFINED_VOICES:
        available_voices = list(PREDEFINED_VOICES.keys())
        raise HTTPException(
            status_code=400, 
            detail=f"Voice '{voice_name}' not found. Available voices: {available_voices}"
        )
    
    voice_info = PREDEFINED_VOICES[voice_name]
    
    # Verify the voice file exists
    if not os.path.exists(voice_info["wav_path"]):
        raise HTTPException(
            status_code=500, 
            detail=f"Voice file not found: {voice_info['wav_path']}"
        )

    # Create job in memory
    job_id = str(uuid.uuid4())
    with jobs_lock:
        jobs[job_id] = {'status': 'queued', 'progress': 'Waiting to start', 'total_chunks': 0}

    # Submit to executor
    executor.submit(inference_task, job_id, voice_info, text, model_name, use_onnx, voice_name)
    logger.info(f"[{request_id}] Job enqueued: {job_id}")

    return JSONResponse({"job_id": job_id, "status": "queued"})

@app.get("/jobs/{job_id}")
async def get_job_result(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    status = job['status']
    if status == 'failed':
        raise HTTPException(status_code=500, detail=f"Job failed: {job.get('exc_info', 'Unknown error')}")

    if status != 'finished':
        progress = job.get('progress', 'Unknown')
        total_chunks = job.get('total_chunks', 0)
        return JSONResponse({
            "status": status,
            "progress": progress,
            "total_chunks": total_chunks
        })

    # Job is finished: stream the result and cleanup
    wav_bytes = job['wav_bytes']
    voice_name = job.get('voice_name', 'output')

    def file_generator():
        yield wav_bytes
        # Cleanup job after successful fetch
        with jobs_lock:
            if job_id in jobs:
                del jobs[job_id]

    return StreamingResponse(
        file_generator(),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="{voice_name}_output.wav"'}
    )

# Keep the custom endpoint for URL-based voices if needed
@app.post("/tts/custom")
async def generate_tts_custom(
    model_name: str = Form(default="zipvoice_distill", description="Model name: zipvoice or zipvoice_distill"),
    prompt_text: str = Form(..., description="Transcription of the prompt WAV"),
    prompt_wav_url: str = Form(..., description="URL to the prompt WAV file"),
    text: str = Form(..., description="Text to synthesize"),
    use_onnx: bool = Form(default=True, description="Use ONNX for faster CPU inference"),
    api_key: str = Depends(get_api_key)
):
    # Implementation for custom voices (similar to your original code)
    raise HTTPException(status_code=501, detail="Custom TTS endpoint not implemented yet")

@app.get("/voices")
def list_voices():
    """Endpoint to list available voices"""
    return {
        "voices": list(PREDEFINED_VOICES.keys()),
        "details": PREDEFINED_VOICES
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))