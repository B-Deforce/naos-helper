import pandas as pd
import sys
from pathlib import Path


def csv_to_excel(csv_path: str):
    csv_file = Path(csv_path)
    if not csv_file.exists():
        raise FileNotFoundError(f"{csv_file} does not exist")

    # Output Excel file with same name but .xlsx extension
    excel_file = csv_file.with_suffix(".xlsx")

    # Read CSV and write to Excel
    df = pd.read_csv(csv_file, sep="|", encoding="utf-8-sig")
    df.to_excel(excel_file, index=False)

    print(f"[done] Converted {csv_file} -> {excel_file}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert.py <file.csv>")
    else:
        csv_to_excel(sys.argv[1])
