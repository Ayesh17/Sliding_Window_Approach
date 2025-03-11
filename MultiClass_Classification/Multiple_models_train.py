import subprocess
import torch

# List of training configurations
training_configs = [
    {
        "dataset_variant": dataset_variant,
        "model_type": model_type,
        "hidden_size": hidden_size,
        "dropout": dropout,
        "learning_rate": learning_rate,
        "batch_size": batch_size
    }
    # for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
    for dataset_variant in ["Data_1000"]
    for model_type, hidden_size, dropout, learning_rate, batch_size in [
        # # RNN
        # ("rnn", 128, 0, 0.0001, 32),
        # ("rnn", 128, 0.2, 0.0001, 16),
        # ("rnn", 128, 0.4, 0.0001, 64),
        # ("rnn", 128, 0.4, 0.0001, 32),
        # ("rnn", 128, 0.4, 0.0001, 16),
        #
        # # Bi-RNN
        # ("bi_rnn", 128, 0.2, 0.0001, 16),
        # ("bi_rnn", 128, 0.2, 0.0001, 64),
        # ("bi_rnn", 128, 0.2, 0.0001, 32),
        # ("bi_rnn", 128, 0.4, 0.0001, 32),
        # ("bi_rnn", 128, 0.4, 0.0001, 16),
        #
        # # GRU
        # ("gru", 128, 0, 0.0001, 32),
        # ("gru", 128, 0.4, 0.0001, 64),
        # ("gru", 128, 0.4, 0.0001, 16),
        # ("gru", 128, 0.4, 0.0001, 32),
        # ("gru", 128, 0.2, 0.0001, 32),

        # # Bi-GRU
        # ("bi_gru", 128, 0.2, 0.0001, 16),
        # ("bi_gru", 128, 0, 0.0001, 16),
        # ("bi_gru", 128, 0.4, 0.0001, 16),
        # ("bi_gru", 128, 0.2, 0.0001, 32),
        # ("bi_gru", 128, 0.4, 0.0001, 32),
        #
        # # LSTM
        # ("lstm", 128, 0, 0.001, 64),
        # ("lstm", 128, 0.4, 0.001, 16),
        # ("lstm", 128, 0.2, 0.001, 64),
        # ("lstm", 128, 0.2, 0.001, 32),
        # ("lstm", 64, 0.2, 0.001, 64),

        # # Bi-LSTM
        # ("bi_lstm", 64, 0.2, 0.001, 32),
        # ("bi_lstm", 128, 0, 0.001, 16),
        # ("bi_lstm", 128, 0.2, 0.001, 16),
        # ("bi_lstm", 64, 0, 0.001, 64),
        # ("bi_lstm", 64, 0.4, 0.001, 64),
        #
        # # Transformer
        # ("transformer", None, 0, 0.001, 16),
        # ("transformer", None, 0, 0.0001, 16),
        # ("transformer", None, 0.2, 0.001, 16),
        # ("transformer", None, 0, 0.001, 32),
        # ("transformer", None, 0, 0.001, 64)

        ("bi_lstm", 128, 0.4, 0.0001, 128),
    ]
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
