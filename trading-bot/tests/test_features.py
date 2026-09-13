from src.features import FEATURE_COLUMNS, add_features, add_labels, build_dataset


def test_add_features_produces_expected_columns(synthetic_price_df):
    out = add_features(synthetic_price_df)
    for col in FEATURE_COLUMNS:
        assert col in out.columns


def test_add_labels_is_binary_and_correct(synthetic_price_df):
    out = add_labels(synthetic_price_df)
    non_null = out["target"].iloc[:-1]
    assert set(non_null.unique()) <= {0, 1}

    for i in range(len(out) - 1):
        expected = int(out["Close"].iloc[i + 1] > out["Close"].iloc[i])
        assert out["target"].iloc[i] == expected


def test_build_dataset_drops_warmup_nans(synthetic_price_df):
    dataset = build_dataset(synthetic_price_df, with_labels=True)
    assert not dataset[FEATURE_COLUMNS + ["target"]].isna().any().any()
    assert len(dataset) < len(synthetic_price_df)
    assert len(dataset) > 0
