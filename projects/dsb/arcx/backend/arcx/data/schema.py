"""Data schema definitions for Parquet logging

Defines the schema for decision logs stored in Parquet format.
"""

import polars as pl
from typing import List


# Schema for decision logs
DECISION_LOG_SCHEMA = {
    "run_id": pl.Utf8,  # Unique run identifier
    "decision_idx": pl.Int32,  # Decision index within run
    "t_sec": pl.Float32,  # Time since raid start (seconds)
    "map_id": pl.Utf8,  # Map identifier
    "action": pl.Utf8,  # Action taken: "stay" or "extract"
    "final_loot_value": pl.Float32,  # Final loot value from run
    "total_time_sec": pl.Float32,  # Total raid duration
    "success": pl.Boolean,  # Whether extraction was successful
    "z_seq": pl.List(pl.Float32),  # Flattened latent sequence (L*D)
    # YOLO valuation metadata (optional, may be null)
    "num_items_detected": pl.Int32,  # Number of items detected by YOLO
    "avg_detection_confidence": pl.Float32,  # Average YOLO confidence
    "value_breakdown_json": pl.Utf8,  # JSON string of value breakdown
    "rarity_counts_json": pl.Utf8,  # JSON string of rarity counts
}


# Schema for feedback logs
FEEDBACK_LOG_SCHEMA = {
    "run_id": pl.Utf8,
    "decision_idx": pl.Int32,
    "timestamp": pl.Float64,
    "rating": pl.Utf8,  # "bad" or "good"
    "context": pl.Utf8,  # JSON string of additional context
}


def create_decision_log_df() -> pl.DataFrame:
    """Create an empty decision log DataFrame with correct schema"""
    return pl.DataFrame(schema=DECISION_LOG_SCHEMA)


def create_feedback_log_df() -> pl.DataFrame:
    """Create an empty feedback log DataFrame with correct schema"""
    return pl.DataFrame(schema=FEEDBACK_LOG_SCHEMA)


def validate_decision_log(df: pl.DataFrame) -> bool:
    """
    Validate that a DataFrame matches the decision log schema.

    Args:
        df: DataFrame to validate

    Returns:
        True if valid, False otherwise
    """
    expected_cols = set(DECISION_LOG_SCHEMA.keys())
    actual_cols = set(df.columns)

    if expected_cols != actual_cols:
        missing = expected_cols - actual_cols
        extra = actual_cols - expected_cols
        if missing:
            print(f"Missing columns: {missing}")
        if extra:
            print(f"Extra columns: {extra}")
        return False

    # Check types (basic check)
    for col, expected_type in DECISION_LOG_SCHEMA.items():
        actual_type = df[col].dtype
        # Polars type checking is more complex, so just check name
        if str(actual_type) != str(expected_type):
            print(f"Type mismatch for {col}: {actual_type} != {expected_type}")
            return False

    return True


def flatten_latent_sequence(z_seq_list: List[List[float]]) -> List[float]:
    """
    Flatten a 2D latent sequence to 1D list.

    Args:
        z_seq_list: [L, D] nested list

    Returns:
        [L*D] flat list
    """
    return [val for row in z_seq_list for val in row]


def unflatten_latent_sequence(z_flat: List[float], seq_len: int, latent_dim: int) -> List[List[float]]:
    """
    Unflatten 1D latent list to 2D sequence.

    Args:
        z_flat: [L*D] flat list
        seq_len: L
        latent_dim: D

    Returns:
        [L, D] nested list
    """
    assert len(z_flat) == seq_len * latent_dim
    return [z_flat[i * latent_dim : (i + 1) * latent_dim] for i in range(seq_len)]


def test_schema():
    """Test schema definitions"""
    print("Testing schema...")

    # Test creating empty DataFrames
    df_decisions = create_decision_log_df()
    print(f"Decision log schema: {df_decisions.schema}")
    assert len(df_decisions) == 0

    df_feedback = create_feedback_log_df()
    print(f"Feedback log schema: {df_feedback.schema}")
    assert len(df_feedback) == 0

    # Test flatten/unflatten
    z_seq = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
    z_flat = flatten_latent_sequence(z_seq)
    print(f"Flattened: {z_flat}")
    assert z_flat == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    z_seq_restored = unflatten_latent_sequence(z_flat, seq_len=2, latent_dim=3)
    print(f"Unflattened: {z_seq_restored}")
    assert z_seq_restored == z_seq

    print("✓ Schema tests passed")


if __name__ == "__main__":
    test_schema()
