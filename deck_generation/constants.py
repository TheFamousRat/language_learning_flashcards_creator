from pathlib import Path

ORIGINAL_SENTENCE_COL_NAME = "original_sentence"
ORIGINAL_ID_COL_NAME = "original_id"
TRANSLATED_SENTENCE_COL_NAME = "translated_sentence"
TRANSLATED_ID_COL_NAME = "translated_id"

AUDIO_FILE_COL_NAME = "audio_file_path"

ROOT_DIR = Path(__file__).parent.parent
GENERATED_DECK_DATA_DIR = ROOT_DIR / "generated_deck_data"
