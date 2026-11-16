# type: ignore
import logging
from pathlib import Path

import pandas
import regex
import tqdm

from deck_generation.constants import TARGET_ID_COL_NAME, TARGET_SENTENCE_COL_NAME
from deck_generation.deck_generation.config import SentenceFilteringConfig
from deck_generation.preprocessing.sentence_filterer import SentenceFilterer

logging.basicConfig()

_logger = logging.getLogger(name=__name__)
_logger.setLevel(level=logging.DEBUG)


def _save_deu_word_ranking_file(
    all_tokens_file_path: Path, dest_path: Path, max_tokens_count: int
) -> None:
    chunks: list[pandas.DataFrame] = []

    current_token_count = 0
    with tqdm.tqdm(total=max_tokens_count) as pbar:
        for chunk in pandas.read_csv(
            all_tokens_file_path, sep="\\t", header=None, chunksize=100000
        ):
            chunk = chunk[~chunk[0].isna()]
            token_has_characters_mask = chunk[0].apply(
                lambda word: regex.search(r"\p{L}", word) is not None
            )
            chunk = chunk[token_has_characters_mask]

            chunks.append(chunk)

            pbar.update(len(chunk))
            current_token_count += len(chunk)
            if current_token_count >= max_tokens_count:
                break

    word_to_ranking_df = pandas.concat(chunks)
    word_to_ranking_df[1] = word_to_ranking_df[1].max() - word_to_ranking_df[1]
    word_to_ranking_df.columns = ["word", "freq"]
    word_to_ranking_df.to_csv(dest_path, sep="\t", index=False)


def main() -> None:
    data_path = Path("/home/thefamousrat/Documents/Projects/Anki_ITA_FRA/data/de")

    word_ranking_file_path = data_path / "tokens_ranking_filtered.csv"
    if not word_ranking_file_path.exists():
        _save_deu_word_ranking_file(
            all_tokens_file_path=data_path / "token_keys.all.tsv",
            dest_path=word_ranking_file_path,
            max_tokens_count=2 * 1e6,
        )

    word_to_ranking_df = pandas.read_csv(
        filepath_or_buffer=word_ranking_file_path, sep="\\t"
    )
    word_to_ranking_df.columns = ["word", "freq"]
    word_to_ranking = dict(zip(word_to_ranking_df["word"], word_to_ranking_df["freq"]))

    sentences_loader = lambda: pandas.read_csv(
        data_path / "deu_sentences_detailed.tsv",
        sep="\t",
        header=None,
        names=[
            TARGET_ID_COL_NAME,
            "lang",
            TARGET_SENTENCE_COL_NAME,
            "user",
            "created",
            "modified",
        ],
    )[[TARGET_ID_COL_NAME, TARGET_SENTENCE_COL_NAME]]

    filterer = SentenceFilterer(
        word_to_freq=word_to_ranking, sentences_loader=sentences_loader
    )

    sentences_filtered_and_ranked_df = filterer.get_filtered_sentences_df(
        SentenceFilteringConfig(
            min_word_count=4,
            max_word_count=1000000,
            max_sentences_count_per_new_word=2,
            only_proper_nouns_capitalized=False,
        )
    )

    sentences_filtered_and_ranked_df.to_csv("deu_ranked.csv", index=False)


if __name__ == "__main__":
    main()
