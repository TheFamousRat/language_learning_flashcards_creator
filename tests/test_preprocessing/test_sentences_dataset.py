from pathlib import Path

import pytest

from deck_generation.preprocessing.sentences_dataset import SentencesDataset


@pytest.fixture()
def sentences_dataset_with_translation() -> SentencesDataset:
    file_path = Path(__file__).parent / "sentences_with_translation.csv"
    dataset = SentencesDataset.from_sentences_file(
        sentences_file_path=file_path,
        separator=";",
        has_header=True,
        original_sentences_column="original",
        translated_sentences_column="translated",
    )

    return dataset

@pytest.fixture()
def sentences_dataset_with_relative_audio() -> SentencesDataset:
    file_path = Path(__file__).parent / "sentences_with_relative_audio_files.csv"
    dataset = SentencesDataset.from_sentences_file(
        sentences_file_path=file_path,
        separator="|",
        has_header=False,
        original_sentences_column=1,
        audio_files_column=2,
    )

    return dataset


def test_it_lazy_loads_additional_columns_to_the_internal_dataframe(
    sentences_dataset_with_translation: SentencesDataset,
) -> None:
    assert sentences_dataset_with_translation.original_sentences is not None
    assert (
        SentencesDataset.ORIGINAL_SENTENCES_COLUMN
        in sentences_dataset_with_translation.sentences_df
    )
    assert (
        sentences_dataset_with_translation.original_sentences
        == sentences_dataset_with_translation.sentences_df[
            SentencesDataset.ORIGINAL_SENTENCES_COLUMN
        ]
    ).all()

    assert (
        SentencesDataset.TRANSLATED_SENTENCES_COLUMN
        not in sentences_dataset_with_translation.sentences_df
    )
    assert sentences_dataset_with_translation.translated_sentences is not None
    assert (
        sentences_dataset_with_translation.translated_sentences
        == sentences_dataset_with_translation.sentences_df[
            SentencesDataset.TRANSLATED_SENTENCES_COLUMN
        ]
    ).all()

def test_it_generates_audio_files_from_a_generator(sentences_dataset_with_relative_audio: SentencesDataset) -> None:
    pass