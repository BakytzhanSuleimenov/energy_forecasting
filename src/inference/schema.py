import pandas as pd


class SchemaValidationError(ValueError):
    pass


def build_input_schema(feature_cols, sequence_length):
    return {
        "required_columns": feature_cols,
        "sequence_length": sequence_length,
        "input_type": "history_window",
    }


def validate_history_frame(history, feature_cols, sequence_length):
    if isinstance(history, pd.DataFrame):
        df = history.copy()
    else:
        df = pd.DataFrame(history)

    missing_columns = [column for column in feature_cols if column not in df.columns]
    if missing_columns:
        raise SchemaValidationError(f"Missing required columns: {missing_columns}")

    if len(df) < sequence_length:
        raise SchemaValidationError(
            f"Expected at least {sequence_length} rows but received {len(df)}"
        )

    df = df.loc[:, feature_cols].copy()
    for column in feature_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    invalid_columns = [column for column in feature_cols if df[column].isna().any()]
    if invalid_columns:
        raise SchemaValidationError(f"Invalid or missing values found in: {invalid_columns}")

    return df.tail(sequence_length).reset_index(drop=True)
