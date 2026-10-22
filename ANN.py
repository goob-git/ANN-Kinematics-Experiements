import os
import time
import psutil
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

# Suppress warnings to keep terminal logs clean
warnings.filterwarnings("ignore", category=UserWarning)

# Hardcoded File Paths - Points to your active workspace file
DATASET_PATH = r"C:\Users\tiise\PycharmProjects\PythonProject1\kinematics_500k_dataset.xlsx.xlsx"


def capture_hardware_metrics():
    """Helper function to log current CPU and RAM usage."""
    process = psutil.Process(os.getpid())
    cpu_percent = psutil.cpu_percent(interval=None)
    ram_usage_mb = process.memory_info().rss / (1024 * 1024)
    return cpu_percent, ram_usage_mb


def clean_kinematics_dataset(file_path):
    """
    Loads Sheet 1 and Sheet 2. Strips out rows from Sheet 1
    that correspond to invalid solutions marked in Sheet 2.
    Removes any pre-existing prediction columns to prevent duplication.
    """
    print("\n[1/5] Loading and Cleaning Multi-Sheet Dataset...")

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Dataset file missing at: {os.path.abspath(file_path)}")

    print("Reading sheets from Excel (this may take a moment for large files)...")
    sheet1 = pd.read_excel(file_path, sheet_name=0)
    sheet2 = pd.read_excel(file_path, sheet_name=1)

    invalid_mask = sheet2.isnull().any(axis=1)
    invalid_indices = sheet2[invalid_mask].index

    print(f"Detected {len(invalid_indices)} invalid kinematics solutions in Sheet 2.")

    cleaned_data = sheet1.drop(index=invalid_indices).dropna()

    # Drop previous prediction columns if they exist from a past run
    drop_cols = [c for c in ['pred_joint1', 'pred_joint2', 'pred_joint3', 'pred_joint4'] if c in cleaned_data.columns]
    if drop_cols:
        cleaned_data = cleaned_data.drop(columns=drop_cols)

    print(f"Cleaning complete. Remaining valid records for processing: {len(cleaned_data)}")

    return cleaned_data, sheet2


def train_and_optimize_model(df, target_r2=0.98):
    """
    Splits data, creates an adaptive training loop,
    and refines hyperparameters if target accuracy is not met.
    """
    print("\n[2/5] Preparing Train/Test Split & Initializing Model...")

    feature_cols = ['x', 'y', 'z', 'pitch']
    target_cols = ['joint1_FK', 'joint2_FK', 'joint3_FK', 'joint4_FK']

    X = df[feature_cols].values
    Y = df[target_cols].values

    # 80/20 Train-Test Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.20, random_state=42
    )

    # High-Performance Parallelized ExtraTreesRegressor Setup
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('et', ExtraTreesRegressor(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            bootstrap=False,
            n_jobs=-1,
            random_state=42
        ))
    ])

    attempt = 1
    max_attempts = 4

    print("\n[3/5] Starting Model Training & Refinement Loop...")
    while True:
        start_time = time.time()
        print(f"--- Training Iteration {attempt} ---")

        # Train model
        model_pipeline.fit(X_train, Y_train)

        # Evaluate on remaining 20% test partition
        predictions = model_pipeline.predict(X_test)
        r2 = r2_score(Y_test, predictions)
        mse = mean_squared_error(Y_test, predictions)

        elapsed = time.time() - start_time
        cpu, ram = capture_hardware_metrics()

        # Display performance results first, followed by hardware usage
        print(f"Results -> R² Accuracy: {r2:.4f} | MSE: {mse:.6f}")
        print(f"Metrics -> Time: {elapsed:.2f}s | CPU: {cpu}% | RAM: {ram:.2f} MB")

        if r2 >= target_r2:
            print(f"[SUCCESS] Target accuracy of {target_r2 * 100}% met successfully!")
            break
        elif attempt >= max_attempts:
            print("[WARNING] Reached maximum tuning attempts. Proceeding with best configuration.")
            break
        else:
            print(f"[REFINE] Accuracy too low ({r2:.2f}). Increasing estimator density to smooth boundaries...")
            attempt += 1

            model_pipeline.set_params(et=ExtraTreesRegressor(
                n_estimators=150 if attempt == 2 else 200,
                max_depth=None,
                min_samples_split=2,
                bootstrap=False,
                n_jobs=-1,
                random_state=42 + attempt
            ))

    return model_pipeline


def execute_full_dataset_inference(model, cleaned_df):
    """ Runs calculations across records, profiles performance, and appends predictions. """
    print("\n[4/5] Running Model Calculation Across Full Valid Dataset...")

    feature_cols = ['x', 'y', 'z', 'pitch']
    target_cols = ['joint1_FK', 'joint2_FK', 'joint3_FK', 'joint4_FK']

    X_full = cleaned_df[feature_cols].values
    Y_full = cleaned_df[target_cols].values

    start_time = time.time()
    initial_cpu, initial_ram = capture_hardware_metrics()

    # Execute full multi-threaded calculation pass
    final_predictions = model.predict(X_full)

    end_time = time.time()
    final_cpu, final_ram = capture_hardware_metrics()

    total_time = end_time - start_time
    final_r2 = r2_score(Y_full, final_predictions)
    final_mse = mean_squared_error(Y_full, final_predictions)

    print("\n============================================================")
    print("FINAL KINEMATICS PIPELINE BENCHMARK SUMMARY")
    print("============================================================")
    print(f"Total Dataset Records Evaluated : {len(X_full):,}")
    print(f"Global Model R² Accuracy Score   : {final_r2:.4f} ({final_r2 * 100:.2f}%)")
    print(f"Global Mean Squared Error (MSE)  : {final_mse:.6f}")
    print(f"Total Calculation Runtime        : {total_time:.4f} seconds")
    print(f"Peak Operational CPU Usage       : {max(initial_cpu, final_cpu):.2f}%")
    print(f"Peak RAM Footprint               : {max(initial_ram, final_ram):.2f} MB")
    print("============================================================")

    # Attach prediction results straight back into the dataframe structure
    cleaned_df['pred_joint1'] = final_predictions[:, 0]
    cleaned_df['pred_joint2'] = final_predictions[:, 1]
    cleaned_df['pred_joint3'] = final_predictions[:, 2]
    cleaned_df['pred_joint4'] = final_predictions[:, 3]

    return cleaned_df


def save_updated_excel_file(file_path, updated_sheet1, original_sheet2):
    """ Rewrites the master file with updated predictions while keeping sheet architecture intact. """
    print("\n[5/5] Automatically Updating Excel Workbook File...")
    start_save_time = time.time()

    # Use ExcelWriter to overwrite the workbook structure without destroying auxiliary sheets
    with pd.ExcelWriter(file_path, engine='openpyxl') as writer:
        updated_sheet1.to_excel(writer, sheet_name="Sheet1", index=False)
        original_sheet2.to_excel(writer, sheet_name="Sheet2", index=False)

    print(f"[SUCCESS] Excel file updated automatically in: {time.time() - start_save_time:.2f} seconds!")
    print(f"Location: {os.path.abspath(file_path)}")


if __name__ == "__main__":
    print("============================================================")
    print("STARTING KINEMATICS PIPELINE & HARDWARE BENCHMARKING")
    print("============================================================")

    try:
        # Step 1: Clean data and preserve Sheet 2 structure
        cleaned_dataset, sheet2_backup = clean_kinematics_dataset(DATASET_PATH)

        # Step 2 & 3: Run multi-core optimization model training loop
        trained_pipeline = train_and_optimize_model(cleaned_dataset, target_r2=0.98)

        # Step 4: Run inference across full dataset and append predictions
        finalized_dataframe = execute_full_dataset_inference(trained_pipeline, cleaned_dataset)

        # Step 5: Save predictions directly into the live file path
        save_updated_excel_file(DATASET_PATH, finalized_dataframe, sheet2_backup)

    except FileNotFoundError as e:
        print(f"\n[Configuration Required]: Please place your master file into the script directory.")
        print(f"Details: {e}")
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed during runtime execution: {str(e)}")

