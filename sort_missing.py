import os

import polars as pl

def read_excel_to_polars(file_path):
    """
    Reads an Excel file into a Polars DataFrame.

    Parameters:
    - file_path (str): Path to the input Excel file.

    Returns:
    - pl.DataFrame: Polars DataFrame containing the Excel data.
    """
    # Read Excel file directly with Polars
    polars_df = pl.read_excel(file_path)
    return polars_df

def calculate_missing_percentages(df: pl.DataFrame):
    total_rows = df.height
    total_columns = df.width

    # Calculate missing percentage for each column
    missing_cols = df.select([
        (pl.col(col).is_null().sum() / total_rows * 100).alias(col) 
        for col in df.columns
    ])
    # Transpose and sort columns by missing percentage descending
    missing_cols_df = missing_cols.transpose(include_header=True).sort(by="column_0", descending=True)
    missing_cols_df = missing_cols_df.rename({"column": "Column", "column_0": "Missing_Percentage"})

    # Reorder DataFrame columns based on sorted missing percentages
    sorted_columns = missing_cols_df["Column"].to_list()
    df_sorted_cols = df.select(sorted_columns)

    # Calculate missing percentage for each row
    new_column_name = "Row_Missing_Percentage"
    df_with_missing = df_sorted_cols.with_columns([
        (pl.all().is_null().sum() / total_columns * 100).alias(new_column_name)
    ])

    # Sort rows by missing percentage descending
    df_sorted = df_with_missing.sort(new_column_name, descending=True)

    return df_sorted, missing_cols_df

    total_rows = df.height
    total_columns = df.width

    # Calculate missing percentage for each column
    missing_cols = df.select([
        (pl.col(col).is_null().sum() / total_rows * 100).alias(col) 
        for col in df.columns
    ])
    # Transpose and sort columns by missing percentage descending
    missing_cols_df = missing_cols.transpose(include_header=True).sort(by="column_0", descending=True)
    missing_cols_df = missing_cols_df.rename({"column": "Column", "column_0": "Missing_Percentage"})

    # Reorder DataFrame columns based on sorted missing percentages
    sorted_columns = missing_cols_df["Column"].to_list()
    df_sorted_cols = df.select(sorted_columns)

    # Calculate missing percentage for each row
    df_with_missing = df_sorted_cols.with_columns([
        (pl.all().is_null().sum() / total_columns * 100).alias("Missing_Percentage")
    ])

    # Sort rows by missing percentage descending
    df_sorted = df_with_missing.sort("Missing_Percentage", descending=True)

    return df_sorted, missing_cols_df

    """
    Calculates the percentage of missing values for each column and each row.

    Parameters:
    - df (pl.DataFrame): Input Polars DataFrame.

    Returns:
    - pl.DataFrame: DataFrame with missing percentages added.
    - pl.DataFrame: DataFrame containing missing percentage for each column.
    """
    total_rows = df.height
    total_columns = df.width

    # Calculate missing percentage for each column
    missing_cols = df.select([
        (pl.col(col).is_null().sum() / total_rows * 100).alias(col) 
        for col in df.columns
    ])
    # Transpose and sort columns by missing percentage descending
    missing_cols_df = missing_cols.transpose(include_header=True).sort(by="column_0", descending=True)
    missing_cols_df = missing_cols_df.rename({"column": "Column", "column_0": "Missing_Percentage"})

    # Reorder DataFrame columns based on sorted missing percentages
    sorted_columns = missing_cols_df["Column"].to_list()
    df_sorted_cols = df.select(sorted_columns)

    # Calculate missing percentage for each row
    df_with_missing = df_sorted_cols.with_columns([
        ((pl.concat_list(pl.all()).arr.eval(pl.element().is_null()).list.sum()) / total_columns * 100).alias("Missing_Percentage")
    ])

    # Sort rows by missing percentage descending
    df_sorted = df_with_missing.sort("Missing_Percentage", descending=True)

    return df_sorted, missing_cols_df

def write_polars_to_excel(df: pl.DataFrame, output_path):
    """
    Writes a Polars DataFrame to an Excel file.

    Parameters:
    - df (pl.DataFrame): Polars DataFrame to write.
    - output_path (str): Path to save the output Excel file.
    """
    # Write to Excel using Polars
    df.write_excel(output_path)
    print(f"Processed file saved to: {output_path}")

def process_excel_file(file_path, output_path):
    """
    Processes an Excel file: calculates missing percentages, sorts columns and rows,
    and writes the sorted DataFrame to a new Excel file.

    Parameters:
    - file_path (str): Path to the input Excel file.
    - output_path (str): Path to save the processed Excel file.
    """
    print(f"Reading file: {file_path}")
    df = read_excel_to_polars(file_path)

    print("Calculating missing percentages and sorting...")
    df_sorted, missing_cols = calculate_missing_percentages(df)

    print("Writing sorted DataFrame to Excel...")
    write_polars_to_excel(df_sorted, output_path)

def main():
    # List of files to process
    files = [
        ("file1", "/home/aricept094/mydata/savede/merged_chat.xlsx"),
        ("file2", "/home/aricept094/mydata/savede/merged_claude.xlsx"),
    ]

    for label, path in files:
        if not os.path.isfile(path):
            print(f"File not found: {path}. Skipping...")
            continue

        # Define the output path, e.g., appending '_sorted' to the original filename
        base, ext = os.path.splitext(path)
        output_path = f"{base}_sorted{ext}"
        print(f"\nProcessing {label}: {path}")
        process_excel_file(path, output_path)

if __name__ == "__main__":
    main()
