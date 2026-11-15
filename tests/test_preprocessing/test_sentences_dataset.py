from pathlib import Path

import pandas
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

    def append_relative_audio_column(dataset: SentencesDataset) -> pandas.Series:
        relative_audio_column = dataset.load_new_dataframe_columns([1])
        absolute_audio_path = relative_audio_column.apply(
            lambda rel_audio: dataset.source_sentences_file_path.parent / rel_audio
        )
        return absolute_audio_path

    dataset = SentencesDataset.from_sentences_file(
        sentences_file_path=file_path,
        separator="|",
        has_header=False,
        original_sentences_column=1,
        audio_files_generator=append_relative_audio_column,
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


def test_it_generates_audio_files_from_a_generator(
    sentences_dataset_with_relative_audio: SentencesDataset,
) -> None:
    assert (
        SentencesDataset.SENTENCES_AUDIO_COLUMN
        not in sentences_dataset_with_relative_audio.sentences_df.columns
    )
    assert sentences_dataset_with_relative_audio.sentences_audio is not None
    assert (
        sentences_dataset_with_relative_audio.sentences_audio
        == sentences_dataset_with_relative_audio.sentences_df[
            SentencesDataset.SENTENCES_AUDIO_COLUMN
        ]
    ).all()
    assert len(sentences_dataset_with_relative_audio.sentences_audio) == len(
        sentences_dataset_with_relative_audio.sentences_df
    )
