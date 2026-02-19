from pathlib import Path

TARGET_SENTENCE_COL_NAME = "target_sentence"
TARGET_ID_COL_NAME = "target_id"
TRANSLATED_SENTENCE_COL_NAME = "translated_sentence"
TRANSLATED_ID_COL_NAME = "translated_id"

AUDIO_FILE_COL_NAME = "audio_file_path"

SENTENCES_WORDS_COL_NAME = "sentences_words"

RAREST_WORD_COL_NAME = "rarest_word"
RAREST_WORD_FREQ_COL_NAME = "rarest_word_freq"

ROOT_DIR = Path(__file__).parent.parent
GENERATED_DECK_DATA_DIR = ROOT_DIR / "generated_deck_data"
