import pandas as pd
import os
import matplotlib.pyplot as plt


def extract_and_save_success_rates(csv_path, output_folder):
    # Load the CSV
    df = pd.read_csv(csv_path)

    # Filter columns
    mean_cols = [col for col in df.columns if col.endswith("/success/rate/mean")]
    best10_cols = [col for col in df.columns if col.endswith("/success/rate/best@10")]

    # Helper to extract base name and non-NaN value
    def extract_values(columns):
        data = {}
        for col in columns:
            base_name = col.split(",")[0]  # extract the <anything> part
            value = df[col].dropna().iloc[0]  # only one non-NaN value per column
            data[base_name] = value
        return data

    # Extract values
    mean_data = extract_values(mean_cols)
    best10_data = extract_values(best10_cols)

    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Save to CSV and Plot
    def save_and_plot(data_dict, filename):
        # Save CSV
        csv_path = os.path.join(output_folder, f"{filename}.csv")
        with open(csv_path, "w") as f:
            f.write(",".join(data_dict.keys()) + "\n")
            f.write(",".join(str(v) for v in data_dict.values()) + "\n")

        # Plot and Save PNG
        plt.figure(figsize=(10, 6))
        plt.bar(data_dict.keys(), data_dict.values(), color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.title(filename.replace("_", " ").title())
        plt.ylabel("Success Rate")
        plt.tight_layout()
        png_path = os.path.join(output_folder, f"{filename}.png")
        plt.savefig(png_path)
        plt.close()

    save_and_plot(mean_data, "success_rate_mean")
    save_and_plot(best10_data, "success_rate_best@10")


if __name__ == "__main__":
    log_path = input("Enter the path to the run logs: ")
    if not log_path:
        log_path = "logs/full_config_20250606_054055"
    extract_and_save_success_rates(
        f"{log_path}/scalars.csv",
        f"{log_path}/results_metrics",
    )
