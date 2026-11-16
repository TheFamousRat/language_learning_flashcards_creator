from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas

DatasetIndex = int
ColumnGenerator = Callable[[DatasetIndex, "SentencesDataset"], pandas.Series]


@dataclass(frozen=True)
class DatasetLoadingConfig:
    source_sentences_file_path: Path
    separator: str
    has_header: bool
    original_sentences_column: Hashable
    translated_sentences_column: Hashable | None = None
    audio_files_column: Hashable | None = None
    audio_files_generator: ColumnGenerator | None = None


class SentencesDataset:
    ORIGINAL_SENTENCES_COLUMN = "original_sentences"
    TRANSLATED_SENTENCES_COLUMN = "translated_sentences"
    SENTENCES_AUDIO_COLUMN = "original_sentences_audio"
    _DATASET_INDEX_COLUMN = "dataset_index"
    _ORIGINAL_FILE_ROWS_COLUMN = "original_file_row"

    def __init__(
        self,
        sentences_df: pandas.DataFrame,
        loading_config_by_dataset_index: dict[DatasetIndex, DatasetLoadingConfig],
    ) -> None:
        self.sentences_df = sentences_df
        self.loading_config_by_dataset_index = loading_config_by_dataset_index

    @property
    def original_sentences(self) -> pandas.Series[str]:
        return self.sentences_df[self.ORIGINAL_SENTENCES_COLUMN]

    @property
    def translated_sentences(self) -> pandas.Series[str] | None:
        def get_translations_column(config: DatasetLoadingConfig) -> Hashable | None:
            return config.translated_sentences_column

        return self._get_column_or_load_it_from_subdatasets(
            column_name=self.TRANSLATED_SENTENCES_COLUMN,
            column_in_file_field=get_translations_column,
        )

    @property
    def sentences_audio(self) -> pandas.Series[str] | None:
        def get_audio_column(config: DatasetLoadingConfig) -> Hashable | None:
            return config.audio_files_column

        def get_audio_generator(config: DatasetLoadingConfig) -> ColumnGenerator | None:
            return config.audio_files_generator

        return self._get_column_or_load_it_from_subdatasets(
            column_name=self.SENTENCES_AUDIO_COLUMN,
            column_in_file_field=get_audio_column,
            column_generator_field=get_audio_generator,
        )

    def _get_column_or_load_it_from_subdatasets(
        self,
        column_name: Hashable,
        column_in_file_field: (
            Callable[[DatasetLoadingConfig], Hashable | None] | None
        ) = None,
        column_generator_field: (
            Callable[[DatasetLoadingConfig], ColumnGenerator | None] | None
        ) = None,
    ) -> pandas.Series | None:
        # TODO: Cache already loaded series by index
        loaded_series: list[pandas.Series] = []
        unique_dataset_indices = self.sentences_df[self._DATASET_INDEX_COLUMN].unique()
        for dataset_index in unique_dataset_indices:
            dataset_config = self.loading_config_by_dataset_index[dataset_index]

            column_in_file = (
                None
                if column_in_file_field is None
                else column_in_file_field(dataset_config)
            )
            column_generator = (
                None
                if column_generator_field is None
                else column_generator_field(dataset_config)
            )

            if column_in_file is not None:
                loaded_column = self.load_new_dataset_dataframe_columns(
                    dataset_index=dataset_index, columns_to_load=[column_in_file]
                )[column_in_file]
                loaded_series.append(loaded_column)
            elif column_generator is not None:
                generated_column = column_generator(dataset_index, self)
                loaded_series.append(generated_column)

        if len(loaded_series) > 0:
            self.sentences_df[column_name] = pandas.concat(loaded_series)

        column_content = (
            None
            if column_name not in self.sentences_df.columns
            else self.sentences_df[column_name]
        )
        return column_content

    def __len__(self) -> int:
        return len(self.sentences_df)

    def load_new_dataset_dataframe_columns(
        self, dataset_index: DatasetIndex, columns_to_load: list[Hashable]
    ) -> pandas.DataFrame:
        loading_config = self.loading_config_by_dataset_index[dataset_index]

        dataset_of_index = self.get_dataset_rows(index=dataset_index)
        dataset_rows = dataset_of_index[self._ORIGINAL_FILE_ROWS_COLUMN].sort_values()

        loaded_rows = self._load_columns_from_dataframe_file(
            dataframe_file=loading_config.source_sentences_file_path,
            separator=loading_config.separator,
            has_header=loading_config.has_header,
            columns_to_load=columns_to_load,
            rows_to_load=dataset_rows,
        )

        loaded_rows.index = dataset_rows.index

        return loaded_rows

    @classmethod
    def _load_columns_from_dataframe_file(
        cls,
        dataframe_file: Path,
        separator: str,
        has_header: bool,
        columns_to_load: list[Hashable],
        rows_to_load: pandas.Series[int] | None = None,
    ) -> pandas.DataFrame:
        header: Literal["infer"] | None = "infer" if has_header else None
        if rows_to_load is None:
            return pandas.read_csv(
                filepath_or_buffer=dataframe_file,
                usecols=columns_to_load,
                sep=separator,
                header=header,
            )

        parts: list[pandas.DataFrame] = []
        for chunk in pandas.read_csv(
            filepath_or_buffer=dataframe_file,
            usecols=columns_to_load,
            sep=separator,
            header=header,
            chunksize=200_000,
        ):
            filtered_chunk = chunk.loc[chunk.index.isin(rows_to_load)]
            parts.append(filtered_chunk)

        new_columns_df = pandas.concat(parts)

        return new_columns_df

    @classmethod
    def from_loading_config(
        cls,
        loading_config: DatasetLoadingConfig,
    ) -> SentencesDataset:
        sentences_df = cls._load_columns_from_dataframe_file(
            dataframe_file=loading_config.source_sentences_file_path,
            separator=loading_config.separator,
            has_header=loading_config.has_header,
            columns_to_load=[loading_config.original_sentences_column],
        )
        sentences_df.columns = [cls.ORIGINAL_SENTENCES_COLUMN]
        sentences_df[cls._ORIGINAL_FILE_ROWS_COLUMN] = sentences_df.index
        sentences_df[cls._DATASET_INDEX_COLUMN] = 0

        return SentencesDataset(
            sentences_df=sentences_df,
            loading_config_by_dataset_index={0: loading_config},
        )

    def get_dataset_rows(self, index: DatasetIndex) -> pandas.DataFrame:
        return self.sentences_df[self.sentences_df[self._DATASET_INDEX_COLUMN] == index]

    def __add__(self, other: SentencesDataset) -> SentencesDataset:
        reindex_sentences_df: list[pandas.DataFrame] = []
        merged_loading_configs_by_index: dict[DatasetIndex, DatasetLoadingConfig] = {}

        max_idx: int = 0
        datasets_to_merge: list[SentencesDataset] = [self, other]
        for dataset_to_merge in datasets_to_merge:
            for (
                sub_dataset_index,
                sub_dataset_config,
            ) in dataset_to_merge.loading_config_by_dataset_index.items():
                merged_loading_configs_by_index[max_idx] = sub_dataset_config

                subdataset_df = dataset_to_merge.sentences_df[
                    dataset_to_merge.sentences_df[self._DATASET_INDEX_COLUMN]
                    == sub_dataset_index
                ]
                subdataset_df[self._DATASET_INDEX_COLUMN] = max_idx
                reindex_sentences_df.append(subdataset_df)

                max_idx += 1

        merged_sentences_df = pandas.concat(reindex_sentences_df, ignore_index=True)

        merged_dataset = SentencesDataset(
            sentences_df=merged_sentences_df,
            loading_config_by_dataset_index=merged_loading_configs_by_index,
        )

        return merged_dataset
