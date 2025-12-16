import torch
import nemo.collections.asr as nemo_asr
import numpy as np
from typing import List, Any, Optional
from pydantic import BaseModel


class WordStamp(BaseModel):
    word: str
    start: float
    end: float


def load_global_model(model_name="nvidia/parakeet-tdt-0.6b-v3"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = nemo_asr.models.ASRModel.from_pretrained(model_name=model_name)
    model.freeze()
    model.to(device)
    model.eval()

    model.change_attention_model(
        self_attention_model="rel_pos_local_attn",
        att_context_size=[128, 0]
    )
    return model


class StreamingParakeet:
    def __init__(self, model: Any):
        self.model = model
        self.device = next(self.model.parameters()).device

        subsampling_factor = self.model.encoder._cfg.subsampling_factor
        self.frame_duration = 0.01 * subsampling_factor


        self.cache = None
        self.global_time_offset = 0.0
        self.partial_word_tokens: List[int] = []
        self.partial_word_start: Optional[float] = None


        self.reset_state()

    def reset_state(self):
        self.cache = None
        self.global_time_offset = 0.0

        self.partial_word_tokens: List[int] = []
        self.partial_word_start: Optional[float] = None

    def transcribe_chunk(self, new_audio_chunk: np.array) -> List[WordStamp]:
        audio_tensor = torch.tensor(new_audio_chunk, dtype=torch.float32).unsqueeze(0).to(self.device)
        audio_length = torch.tensor([audio_tensor.shape[1]], dtype=torch.long).to(self.device)

        word_stamps: List[WordStamp] = []

        with torch.no_grad():
            processed_signal, processed_len, new_cache = self.model.encoder(
                audio_signal=audio_tensor,
                length=audio_length,
                cache=self.cache
            )
            self.cache = new_cache

            best_hyp = self.model.decoding.rnnt_decoder_predictions_tensor(
                encoder_output=processed_signal,
                encoded_lengths=processed_len,
                return_hypotheses=True
            )

            hyp = best_hyp[0]

            if hyp.y_sequence is not None and len(hyp.y_sequence) > 0:
                token_ids = hyp.y_sequence.cpu().tolist()
                timesteps = hyp.timestep

                for i, token_id in enumerate(token_ids):
                    token_str = self.model.tokenizer.ids_to_text([token_id])

                    frame_index = timesteps[i] if i < len(timesteps) else 0
                    token_time = (frame_index * self.frame_duration) + self.global_time_offset

                    is_new_word = token_str.startswith(" ")

                    if is_new_word:
                        if self.partial_word_tokens:
                            full_word = self.model.tokenizer.ids_to_text(self.partial_word_tokens).strip()
                            if full_word:
                                word_stamps.append(WordStamp(
                                    word=full_word,
                                    start=round(self.partial_word_start, 3),
                                    end=round(token_time, 3)
                                ))

                        self.partial_word_tokens = [token_id]
                        self.partial_word_start = token_time
                    else:
                        self.partial_word_tokens.append(token_id)
                        if self.partial_word_start is None:
                            self.partial_word_start = token_time

            chunk_duration = processed_len[0].item() * self.frame_duration
            self.global_time_offset += chunk_duration

            return word_stamps

    def finish(self) -> List[WordStamp]:
        final_stamps = []
        if self.partial_word_tokens:
            full_word = self.model.tokenizer.ids_to_text(self.partial_word_tokens).strip()
            if full_word:
                final_stamps.append(WordStamp(
                    word=full_word,
                    start=round(self.partial_word_start, 3),
                    end=round(self.global_time_offset, 3)
                ))
        self.reset_state()
        return final_stamps
