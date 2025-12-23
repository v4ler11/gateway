import asyncio
import time

from typing import AsyncIterator, AsyncGenerator, Any

import numpy as np

from core.logger import error, info
from stt.inference.schemas import ParakeetEvent, SpeechStart, SpeechStop, SpeechTranscription


def _check_silero_speech(vad_model: Any, chunk: np.ndarray, sample_rate: int) -> bool:
    try:
        out = vad_model(chunk, sample_rate)
        if hasattr(out, 'item'):
            prob = out.item()
        else:
            prob = out
        return prob > 0.5
    except Exception:
        rms = np.sqrt(np.mean(chunk ** 2))
        return rms > 0.01


async def stream_parakeet_with_vad(
        loop: asyncio.AbstractEventLoop,
        audio_stream: AsyncIterator[np.ndarray],
        model: Any,
        vad_model: Any,
        sample_rate: int = 16000,
        min_silence_duration: float = 0.3,
        max_duration: float = 180.0,
) -> AsyncGenerator[ParakeetEvent, None]:
    buffer = []
    buffer_duration = 0.0

    is_speech_active = False
    silence_counter = 0.0

    vad_chunk_size = 512
    vad_buffer = np.array([], dtype=np.float32)

    async for chunk in audio_stream:
        buffer.append(chunk)

        chunk_duration = len(chunk) / sample_rate
        buffer_duration += chunk_duration

        vad_buffer = np.concatenate([vad_buffer, chunk])

        while len(vad_buffer) >= vad_chunk_size:
            process_chunk = vad_buffer[:vad_chunk_size]
            vad_buffer = vad_buffer[vad_chunk_size:]

            is_speech = _check_silero_speech(vad_model, process_chunk, sample_rate)

            if is_speech:
                if not is_speech_active:
                    yield SpeechStart()

                is_speech_active = True
                silence_counter = 0.0
            else:
                if is_speech_active:
                    chunk_duration_vad = vad_chunk_size / sample_rate
                    silence_counter += chunk_duration_vad

        if buffer_duration >= max_duration:
            if is_speech_active:
                yield SpeechStop()

                full_audio = np.concatenate(buffer)

                process_audio = full_audio

                try:
                    t0 = time.time()
                    result = await loop.run_in_executor(None, model.recognize, process_audio)
                    info(f"Inference took {time.time() - t0:.2f} seconds")

                    text = ""
                    if isinstance(result, str):
                        text = result.strip()
                    elif hasattr(result, 'text') and result.text:
                        text = result.text.strip()

                    if text:
                        yield SpeechTranscription(text=text)

                except Exception as e:
                    error(f"Inference failed during force flush: {e}")

            buffer = []
            buffer_duration = 0.0
            is_speech_active = False
            silence_counter = 0.0

            continue

        if is_speech_active and silence_counter >= min_silence_duration:
            yield SpeechStop()

            full_audio = np.concatenate(buffer)
            buffer = []
            buffer_duration = 0.0
            is_speech_active = False
            silence_counter = 0.0

            trim_samples = int((min_silence_duration - 0.1) * sample_rate)
            if trim_samples > 0 and len(full_audio) > trim_samples:
                process_audio = full_audio[:-trim_samples]
            else:
                process_audio = full_audio

            try:
                result = await loop.run_in_executor(None, model.recognize, process_audio)

                text = ""
                if isinstance(result, str):
                    text = result.strip()
                elif hasattr(result, 'text') and result.text:
                    text = result.text.strip()

                if text:
                    yield SpeechTranscription(text=text)

            except Exception as e:
                error(f"Inference failed: {e}")

    if buffer and is_speech_active:
        yield SpeechStop()

        full_audio = np.concatenate(buffer)

        try:
            result = await loop.run_in_executor(None, model.recognize, full_audio)

            text = ""
            if isinstance(result, str):
                text = result.strip()
            elif hasattr(result, 'text') and result.text:
                text = result.text.strip()

            if text:
                yield SpeechTranscription(text=text)

        except Exception as e:
            error(f"Inference failed during final flush: {e}")
