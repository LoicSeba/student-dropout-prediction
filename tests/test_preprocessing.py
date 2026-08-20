import pandas as pd
import pytest
from preprocessing import RareCategoryGrouper, build_preprocessing_pipeline


def make_df(n=100):
    return pd.DataFrame({
        "Course": ["A"] * 60 + ["B"] * 30 + ["rare"] * 10,
        "Father's occupation": [1] * n,
        "Father's qualification": [1] * n,
        "Mother's qualification": [1] * n,
        "Mother's occupation": [1] * n,
        "Application mode": [1] * n,
        "Marital status": [1] * n,
        "Nacionality": [1] * n,
        "Previous qualification": [1] * n,
        "Age": list(range(n)),
    })


def test_rare_category_grouper_learns_only_on_fit_data():
    df = make_df()
    grouper = RareCategoryGrouper(threshold=15)
    grouper.fit(df)
    # "rare" appears 10 times (< 15) -> should be grouped
    assert "rare" in grouper.rare_categories_["Course"]
    # "A" appears 60 times (>= 15) -> should NOT be grouped
    assert "A" not in grouper.rare_categories_["Course"]


def test_transform_does_not_recompute_statistics():
    train_df = make_df(n=100)
    # A category that's common in "test" but was rare in train
    test_df = pd.DataFrame({**{c: [1] * 5 for c in train_df.columns if c != "Course"},
                             "Course": ["rare"] * 5})

    grouper = RareCategoryGrouper(threshold=15)
    grouper.fit(train_df)
    result = grouper.transform(test_df)
    assert (result["Course"] == "99").all()


def test_pipeline_handles_unseen_category_without_crashing():
    train_df = make_df()
    pipeline = build_preprocessing_pipeline()
    pipeline.named_steps["rare_grouping"].threshold = 15
    pipeline.fit(train_df)

    unseen_row = train_df.iloc[[0]].copy()
    unseen_row["Course"] = "totally_new_category_never_seen"

    result = pipeline.transform(unseen_row)
    assert result.shape[0] == 1  # doesn't crash, doesn't drop the row


def test_single_row_inference_matches_batch_column_count():
    """Transforming one row must produce
    the same number of columns as transforming the full batch."""
    df = make_df()
    pipeline = build_preprocessing_pipeline()
    pipeline.named_steps["rare_grouping"].threshold = 15
    pipeline.fit(df)

    batch_result = pipeline.transform(df)
    single_result = pipeline.transform(df.iloc[[0]])

    assert batch_result.shape[1] == single_result.shape[1]