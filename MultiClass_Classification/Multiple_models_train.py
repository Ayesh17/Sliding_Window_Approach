import subprocess
import torch


# Define dataset-specific configurations
dataset_specific_configs = {
    "Data_1000": [
        ("transformer", 0, 0.001, 16),
        ("transformer", 0, 0.001, 64),
        ("transformer", 0, 0.0001, 16),
        ("transformer", 0.2, 0.001, 32),
        ("transformer", 0, 0.001, 32),
    ],
    "Data": [
        ("transformer", 0, 0.001, 16),
        ("transformer", 0, 0.001, 64),
        ("transformer", 0, 0.0001, 16),
        ("transformer", 0.2, 0.001, 32),
        ("transformer", 0, 0.001, 32),
    ],
    "Binary_Data_1000": [
        ("transformer", 0, 0.001, 16),
        ("transformer", 0, 0.0001, 16),
        ("transformer", 0.2, 0.001, 16),
        ("transformer", 0, 0.001, 32),
        ("transformer", 0, 0.001, 64),
    ],
    "Binary_Data": [
        ("transformer", 0, 0.001, 16),
        ("transformer", 0, 0.0001, 16),
        ("transformer", 0.2, 0.001, 16),
        ("transformer", 0, 0.001, 32),
        ("transformer", 0, 0.001, 64),
    ],
}

# Generate training configurations
training_configs = [
    {
        "dataset_variant": dataset_variant,
        "model_type": model_type,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }
    for dataset_variant, model_configs in dataset_specific_configs.items()
    for model_type, dropout, learning_rate, batch_size in model_configs
]


# Path to the training script (Fixing backslashes for Windows)
training_script = r"Multiclass_Classification/Multiclass_model_train.py"

# Python environment executable
python_executable = r"C:\Users\ameepaganithage\Documents\GitHub\Sliding_Window_Approach\.venv\Scripts\python.exe"

# Run each configuration sequentially
for config in training_configs:
    print(f"\n🚀 Starting training for: {config['model_type']} with {config['dataset_variant']} 🚀\n")

    # Construct command dynamically
    cmd = [
        python_executable, training_script,
        "--model_type", config["model_type"],
        "--dataset_variant", config["dataset_variant"],
        "--dropout", str(config["dropout"]),
        "--learning_rate", str(config["learning_rate"]),
        "--batch_size", str(config["batch_size"]),
    ]

    # Add hidden_size only if it exists in the config
    if "hidden_size" in config:
        cmd.extend(["--hidden_size", str(config["hidden_size"])])

    try:
        # Run the training script
        subprocess.run(cmd, check=True)
        print(f"\n✅ Finished training for: {config['model_type']} ✅\n")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training failed for: {config['model_type']} (Error: {e}) ❌\n")
        continue  # Continue with the next configuration

print("\n🎉 All training runs completed! 🎉")
