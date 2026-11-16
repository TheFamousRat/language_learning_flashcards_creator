from pathlib import Path

import pandas
import pytest
from pandas.testing import assert_frame_equal

from deck_generation.preprocessing.sentences_dataset import (
    DatasetIndex,
    DatasetLoadingConfig,
    SentencesDataset,
)


@pytest.fixture()
def sentences_dataset_with_translation() -> SentencesDataset:
    file_path = Path(__file__).parent / "sentences_with_translation.csv"
    dataset = SentencesDataset.from_loading_config(
        loading_config=DatasetLoadingConfig(
            source_sentences_file_path=file_path,
            separator=";",
            has_header=True,
            original_sentences_column="original",
            translated_sentences_column="translated",
        )
    )

    return dataset


@pytest.fixture()
def sentences_dataset_with_relative_audio() -> SentencesDataset:
    file_path = Path(__file__).parent / "sentences_with_relative_audio_files.csv"

    def append_relative_audio_column(
        dataset_index: DatasetIndex, dataset: SentencesDataset
    ) -> pandas.Series:
        relative_audio_column = dataset.load_new_dataset_dataframe_columns(
            dataset_index=dataset_index, columns_to_load=[1]
        )[1]
        sub_dataset_config = dataset.loading_config_by_dataset_index[dataset_index]
        absolute_audio_path = relative_audio_column.apply(
            lambda rel_audio: sub_dataset_config.source_sentences_file_path.parent
            / rel_audio
        )
        return absolute_audio_path

    dataset = SentencesDataset.from_loading_config(
        loading_config=DatasetLoadingConfig(
            source_sentences_file_path=file_path,
            separator="|",
            has_header=False,
            original_sentences_column=0,
            audio_files_generator=append_relative_audio_column,
        )
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


def test_it_correctly_concatenates_dataset_data(
    sentences_dataset_with_translation: SentencesDataset,
    sentences_dataset_with_relative_audio: SentencesDataset,
) -> None:
    merged_dataset = (
        sentences_dataset_with_translation + sentences_dataset_with_relative_audio
    )

    assert len(merged_dataset.sentences_df) == len(
        sentences_dataset_with_relative_audio.sentences_df
    ) + len(sentences_dataset_with_translation.sentences_df)
    assert set(
        merged_dataset.sentences_df[merged_dataset._DATASET_INDEX_COLUMN].unique()
    ) == {0, 1}

    dataset_1 = merged_dataset.get_dataset_rows(index=0)
    dataset_2 = merged_dataset.get_dataset_rows(index=1)
    recovered_datasets = [dataset_1, dataset_2]
    original_datasets = [
        sentences_dataset_with_translation.sentences_df,
        sentences_dataset_with_relative_audio.sentences_df,
    ]
    if len(dataset_1) == len(sentences_dataset_with_relative_audio.sentences_df):
        original_datasets = original_datasets[::-1]

    for recovered_dataset, original_dataset in zip(
        recovered_datasets, original_datasets
    ):
        all_columns_but_index_set = set(recovered_dataset.columns)
        all_columns_but_index_set.remove(SentencesDataset._DATASET_INDEX_COLUMN)
        all_columns_but_index = list(all_columns_but_index_set)
        assert_frame_equal(
            left=recovered_dataset[all_columns_but_index].reset_index(drop=True),
            right=original_dataset[all_columns_but_index].reset_index(drop=True),
        )


def test_it_correctly_loads_data_on_concatenated_datasets(
    sentences_dataset_with_translation: SentencesDataset,
    sentences_dataset_with_relative_audio: SentencesDataset,
) -> None:
    merged_dataset = (
        sentences_dataset_with_translation + sentences_dataset_with_relative_audio
    )

    merged_dataset.sentences_audio
