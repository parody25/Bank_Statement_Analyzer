import os
import pandas as pd

def filter_transactions(final_csv_path, output_dir=None):
    if not os.path.exists(final_csv_path):
        raise FileNotFoundError(f"CSV not found at: {final_csv_path}")

    df = pd.read_csv(final_csv_path)

    df['Debit'] = pd.to_numeric(df['Debit'], errors='coerce').fillna(0)
    df['Credit'] = pd.to_numeric(df['Credit'], errors='coerce').fillna(0)

    credit_df = df[df['Credit'] > 0]
    debit_df = df[df['Debit'] > 0]

    if output_dir is None:
        output_dir = os.path.dirname(final_csv_path)
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(final_csv_path))[0]
    if base_name.startswith("combined_"):
        uuid_part = base_name[len("combined_"):]
    else:
        uuid_part = base_name

    credit_filename = f"credit_{uuid_part}.csv"
    debit_filename = f"debit_{uuid_part}.csv"

    credit_path = os.path.join(output_dir, credit_filename)
    debit_path = os.path.join(output_dir, debit_filename)

    credit_df.to_csv(credit_path, index=False)
    debit_df.to_csv(debit_path, index=False)

    print(f"Credits saved to: {credit_path}")
    print(f"Debits saved to: {debit_path}")

    return {
        "credit_csv": credit_path,
        "debit_csv": debit_path
    }

# if __name__ == "__main__":
#     BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     FINAL_CSV_DIR = os.path.join(BASE_DIR, "uploads", "final_csv_uploads")
#     final_csv_path = os.path.join(FINAL_CSV_DIR, "combined_1366e2ca8ab045bab33433a71fc18376.csv")
    
#     filter_transactions(final_csv_path)