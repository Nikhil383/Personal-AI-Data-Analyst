"""AI Data Analyst - Data Loader Module"""
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple
import io


class DataLoader:
    """Handles loading and initial processing of data files."""

    @staticmethod
    def load_csv(file_path: str) -> pd.DataFrame:
        """Load CSV file with auto-detection of delimiter."""
        # Try common delimiters
        delimiters = [',', ';', '\t', '|']

        for delimiter in delimiters:
            try:
                df = pd.read_csv(file_path, sep=delimiter, nrows=5)
                # If we got more than 1 column, this delimiter works
                if len(df.columns) > 1:
                    # Reload with correct delimiter
                    df = pd.read_csv(file_path, sep=delimiter)
                    return df
            except Exception:
                continue

        # Fallback to default
        return pd.read_csv(file_path)

    @staticmethod
    def load_excel(file_path: str, sheet_name: Optional[str] = None) -> pd.DataFrame:
        """Load Excel file."""
        return pd.read_excel(file_path, sheet_name=sheet_name)

    @staticmethod
    def load_file(file_path: str) -> pd.DataFrame:
        """Load data file based on extension."""
        path = Path(file_path)
        suffix = path.suffix.lower()

        loaders = {
            '.csv': DataLoader.load_csv,
            '.xlsx': DataLoader.load_excel,
            '.xls': DataLoader.load_excel,
        }

        if suffix not in loaders:
            raise ValueError(f"Unsupported file format: {suffix}")

        return loaders[suffix](file_path)

    @staticmethod
    def get_data_info(df: pd.DataFrame) -> dict:
        """Get basic information about the dataframe."""
        return {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': list(df.columns),
            'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
            'missing_values': df.isnull().sum().to_dict(),
            'numeric_columns': list(df.select_dtypes(include=['number']).columns),
            'categorical_columns': list(df.select_dtypes(include=['object', 'category']).columns),
        }

    @staticmethod
    def get_summary_stats(df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics for numeric columns."""
        return df.describe()

    @staticmethod
    def get_preview(df: pd.DataFrame, n_rows: int = 100) -> pd.DataFrame:
        """Get preview of dataframe."""
        return df.head(n_rows)
