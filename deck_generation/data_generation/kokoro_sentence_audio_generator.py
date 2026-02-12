from pathlib import Path


import pandas
import soundfile as sf  # type: ignore[import-untyped]
import torch
import tqdm
from kokoro import KPipeline  # type: ignore[import-untyped]
import logging

from deck_generation.bin.config import AudioGenerationConfig

_logger = logging.getLogger(name=__file__)
_logger.setLevel(level=logging.DEBUG)

# TODO: Try out https://github.com/Kugelaudio/kugelaudio-open


class KokoroSentenceAudioGenerator:
    def __init__(self, config: AudioGenerationConfig) -> None:
        self.config = config
        self._pipeline: KPipeline | None = None

    @property
    def pipeline(self) -> KPipeline:
        if self._pipeline is None:
            _logger.info("   Loading audio model...")
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._pipeline = KPipeline(
                lang_code=self.config.language_code,
                device=device,
                repo_id="hexgrad/Kokoro-82M",
            )

        return self._pipeline

    def generate_sentences_audio(
        self,
        sentences: list[str],
        sentences_ids: list[int],
        sentences_paths: list[Path],
        overwrite_existing_files: bool,
    ) -> None:
        _logger.info(f"Received {len(sentences)} sentences to generate audio for...")
        assert len(sentences) == len(
            sentences_paths
        ), "Please provide an equal number of sentences and paths for audio generation."

        sentences_df = pandas.DataFrame(
            data={"sentences": sentences, "paths": sentences_paths, "ids": sentences_ids}
        )
        sentences_df["voice"] = sentences_df["ids"] % len(self.config.voices)

        if not overwrite_existing_files:
            existing_sentences_file_mask = sentences_df["paths"].apply(
                lambda sentence_file_path: Path(sentence_file_path).exists()
            )
            if existing_sentences_file_mask.sum() > 0:
                _logger.info(
                    f"{existing_sentences_file_mask.sum()} out of {len(sentences_df)} audio files already generated, skipping them."
                )

            if sum(existing_sentences_file_mask) == len(sentences):
                _logger.info("All sentence audio files already exists.")
                return

            sentences_df = sentences_df[~existing_sentences_file_mask]

        with tqdm.tqdm(
            total=len(sentences_df), desc="Generating audio sentences"
        ) as pbar:
            for voice_idx, voice in enumerate(self.config.voices):
                sentences_df_chunk = sentences_df[sentences_df["voice"] == voice_idx]
                generator = self.pipeline(
                    sentences_df_chunk["sentences"].values,
                    voice=voice,
                    speed=1,
                    split_pattern=r"\n+",
                )
                for sentence_idx, result in enumerate(generator):
                    sf.write(
                        file=sentences_df_chunk["paths"].iloc[sentence_idx],
                        data=result.audio
                        / result.audio.max(),
                        samplerate=24000,
                        compression_level=0.7,
                    )
                    pbar.update(1)
        
        _logger.info("Audio generation done.")
