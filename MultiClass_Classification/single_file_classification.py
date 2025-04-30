import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter

# Select Model Type & Dataset Folder
model_type = "transformer" # Options: rnn, bi_rnn, gru, bi_gru, lstm, bi_lstm, transformer
csv_folder_path = "../Datasets/HII_New_Data/test"


# Define Results Folders
base_results_folder = "../Results"
conf_matrix_folder = os.path.join(base_results_folder, "confusion_matrix_evaluation")
report_folder = os.path.join(base_results_folder, "classification_report_evaluation")
intent_graph_folder = os.path.join(base_results_folder, "intent_graphs")
xy_plot_folder = os.path.join(base_results_folder, "xy_color_plots")
os.makedirs(conf_matrix_folder, exist_ok=True)
os.makedirs(report_folder, exist_ok=True)
os.makedirs(intent_graph_folder, exist_ok=True)
os.makedirs(xy_plot_folder, exist_ok=True)


# Accuracy Log File
accuracy_log_file = os.path.join(base_results_folder, "model_accuracies.csv")

# Prepare to append results
log_columns = ["filename", "model_type", "num_predictions", "accuracy (%)"]
log_rows = []

# Behavior Label Mapping
behavior_mapping = {
    0: "benign",
    1: "block",
    2: "ram",
    3: "cross",
    4: "headon",
    5: "herd",
    6: "overtake"
}
num_classes = len(behavior_mapping)
all_class_indices = list(range(num_classes))
behavior_labels = [behavior_mapping[i] for i in all_class_indices]

# Load Model
model_path = f"../Models/{model_type}_model_Data_1.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load(model_path, map_location=device, weights_only=False)
model.to(device)
model.eval()

# Load CSV Data with Sliding Windows
def load_data_from_csv(csv_file, window_size=20, step_size=5):
    sequences, labels = [], []
    try:
        df = pd.read_csv(csv_file)
        if df.empty or 'Label' not in df.columns:
            print(f"Skipping {csv_file}: File is empty or missing 'Label'.")
            return np.array([]), np.array([])

        feature_columns = [col for col in df.columns if col != 'Label']
        full_sequence = df[feature_columns].values.astype(np.float32)

        for start in range(0, len(full_sequence) - window_size + 1, step_size):
            window = full_sequence[start:start + window_size]
            if window.shape != (window_size, len(feature_columns)):
                continue

            window_labels = df['Label'][start:start + window_size].values
            if (window_labels == -1).any():
                continue

            majority_label = Counter(window_labels).most_common(1)[0][0]
            sequences.append(window)
            labels.append(majority_label)
    except Exception as e:
        print(f"Error processing {csv_file}: {e}")
    return np.array(sequences), np.array(labels)


# intent over time graph generation
def plot_intent_over_time(filename, all_preds, all_labels, behavior_labels, save_path):
    import matplotlib.pyplot as plt
    time_steps = range(len(all_preds))
    plt.figure(figsize=(12, 4))
    plt.plot(time_steps, all_labels, label="Ground Truth", color='black', linestyle='--', alpha=0.7)
    plt.plot(time_steps, all_preds, label="Predicted", color='blue', alpha=0.8)
    plt.yticks(ticks=range(len(behavior_labels)), labels=behavior_labels)
    plt.xlabel("Time Step")
    plt.ylabel("Intent")
    plt.title(f"Intent over Time: {filename}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# color coded graph generation
def plot_color_coded_xy(filename, df, all_preds, behavior_labels, save_path,
                        lat_col="abs_b_r", lon_col="abs_r_b",
                        window_size=20, step_size=5):
    import matplotlib.pyplot as plt
    import numpy as np

    if lat_col not in df.columns or lon_col not in df.columns:
        print(f"⚠️ Could not plot XY data for {filename} — missing {lat_col}/{lon_col} columns.")
        return

    try:
        latitudes = df[lat_col].values
        longitudes = df[lon_col].values
        position_steps = [i + window_size // 2 for i in range(0, len(df) - window_size + 1, step_size)]

        if len(position_steps) != len(all_preds):
            print(f"⚠️ Mismatch between position steps and predictions for {filename}.")
            return

        plot_lats = latitudes[position_steps]
        plot_lons = longitudes[position_steps]

        plt.figure(figsize=(8, 8))
        scatter = plt.scatter(plot_lons, plot_lats, c=all_preds, cmap="tab10", s=15)
        cbar = plt.colorbar(scatter, ticks=range(len(behavior_labels)))
        cbar.ax.set_yticklabels(behavior_labels)
        plt.xlabel(lon_col)
        plt.ylabel(lat_col)
        plt.title(f"XY Color Plot: {filename}")
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
    except Exception as e:
        print(f"⚠️ Error generating XY plot for {filename}: {e}")



# Loop through CSV files in the folder
for filename in os.listdir(csv_folder_path):
    if filename.endswith(".csv"):
        file_path = os.path.join(csv_folder_path, filename)
        print(f"Evaluating {filename}...")

        X_test, y_test = load_data_from_csv(file_path)
        if len(X_test) == 0:
            print(f"No valid sequences found in {filename}. Skipping.")
            continue

        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)
        test_loader = DataLoader(TensorDataset(X_test_tensor, y_test_tensor), batch_size=64, shuffle=False)

        all_preds, all_labels = [], []
        with torch.no_grad():
            for sequences, labels in test_loader:
                outputs = model(sequences)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
        print(f"Test Accuracy for {filename}: {test_accuracy * 100:.2f}%")
        print(f"  - Labels present in ground truth: {np.unique(all_labels)}")
        print(f"  - Labels present in predictions:   {np.unique(all_preds)}")

        # Log accuracy row
        log_rows.append({
            "filename": filename,
            "model_type": model_type,
            "num_predictions": len(all_preds),
            "accuracy (%)": round(test_accuracy * 100, 2)
        })

        # Confusion Matrix (include all labels explicitly)
        conf_matrix = confusion_matrix(
            all_labels,
            all_preds,
            labels=all_class_indices  # ensures 7x7 matrix
        )

        # Plot Confusion Matrix
        plt.figure(figsize=(10, 7))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                    xticklabels=behavior_labels, yticklabels=behavior_labels)
        plt.title(f"Confusion Matrix ({model_type.upper()}): {filename}")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        cm_filename = os.path.join(conf_matrix_folder, f"confusion_matrix_{filename.replace('.csv', '')}.png")
        plt.savefig(cm_filename)
        plt.close()

        # Classification Report
        classification_report_text = classification_report(
            all_labels,
            all_preds,
            labels=all_class_indices,
            target_names=behavior_labels,
            zero_division=0
        )
        report_filename = os.path.join(report_folder, f"classification_report_{filename.replace('.csv', '')}.txt")
        with open(report_filename, "w") as f:
            f.write(classification_report_text)


        # Load original CSV again for plotting purposes
        df_orig = pd.read_csv(file_path)

        # Intent over Time Plot
        intent_graph_path = os.path.join(intent_graph_folder,
                                         f"intent_over_time_{filename.replace('.csv', '')}.png")
        plot_intent_over_time(filename, all_preds, all_labels, behavior_labels, intent_graph_path)

        # XY Color Plot
        # XY Color Plot
        xy_plot_path = os.path.join(xy_plot_folder, f"xy_color_plot_{filename.replace('.csv', '')}.png")
        plot_color_coded_xy(filename, df_orig, all_preds, behavior_labels, xy_plot_path,
                            lat_col="abs_b_r", lon_col="abs_r_b",
                            window_size=20, step_size=5)

# Save accuracy log
df_log = pd.DataFrame(log_rows)

# If file exists, append without writing headers
if os.path.exists(accuracy_log_file):
    df_log.to_csv(accuracy_log_file, mode='a', header=False, index=False)
else:
    df_log.to_csv(accuracy_log_file, mode='w', header=True, index=False)


print(f"\n✅ All results saved in:\n- {conf_matrix_folder}\n- {report_folder}")
