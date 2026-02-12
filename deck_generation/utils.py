def split_string_at_indices(
    string_to_split: str, split_indices: list[int]
) -> list[str]:
    split_string: list[str] = [
        string_to_split[start_split_idx:end_split_idx]
        for start_split_idx, end_split_idx in zip(
            split_indices, split_indices[1:] + [None]
        )
    ]

    return split_string
