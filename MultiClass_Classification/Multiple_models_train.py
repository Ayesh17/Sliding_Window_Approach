import subprocess
import torch

# List of training configurations
training_configs = [


    # BI-GRU Cases
    *[
        {
            "model_type": "bi_gru",
            "dataset_variant": dataset_variant,
            "hidden_size": 128,
            "dropout": dropout,
            "learning_rate": 0.0001,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Data"]
        for dropout, batch_size in [(0.0, 64)]
    ],

    # LSTM Cases
    *[
        {
            "model_type": "lstm",
            "dataset_variant": dataset_variant,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Data"]
        for hidden_size, dropout, learning_rate, batch_size in [
            (128, 0.0, 0.001, 32),
            (64, 0.2, 0.001, 32),
            (128, 0.0, 0.001, 64),
            (128, 0.2, 0.001, 64),
            (64, 0.2, 0.0001, 64),
        ]
    ],

    # BI-LSTM Cases
    *[
        {
            "model_type": "bi_lstm",
            "dataset_variant": dataset_variant,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Data_1000", "Data"]
        for hidden_size, dropout, learning_rate, batch_size in [
            (64, 0.2, 0.001, 32),
            (128, 0.4, 0.0001, 64),
            (128, 0.2, 0.001, 32),
            (32, 0.4, 0.001, 16),
            (128, 0.4, 0.001, 32),
        ]
    ],

    # Transformer Cases (No hidden_size)
    *[
        {
            "model_type": "transformer",
            "dataset_variant": dataset_variant,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Data_1000", "Data"]
        for dropout, learning_rate, batch_size in [
            (0.0, 0.001, 16),
            (0.0, 0.001, 64),
            (0.0, 0.0001, 16),
            (0.2, 0.001, 32),
            (0.0, 0.001, 32),
        ]
    ],




    ###############################
    # Binary Classification
    # RNN Cases
    *[
        {
            "model_type": "rnn",
            "dataset_variant": dataset_variant,
            "hidden_size": 128,
            "dropout": dropout,
            "learning_rate": 0.0001,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for dropout, batch_size in [(0.2, 64), (0.2, 32), (0.4, 32), (0.4, 64), (0.0, 64)]
    ],

    # BI_RNN Cases
    *[
        {
            "model_type": "bi_rnn",
            "dataset_variant": dataset_variant,
            "hidden_size": 128,
            "dropout": dropout,
            "learning_rate": 0.0001,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for dropout, batch_size in [(0.4, 16), (0.2, 32), (0.2, 16), (0.2, 64), (0.0, 64)]
    ],

    # GRU Cases
    *[
        {
            "model_type": "gru",
            "dataset_variant": dataset_variant,
            "hidden_size": 128,
            "dropout": dropout,
            "learning_rate": 0.0001,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for dropout, batch_size in [(0.2, 16), (0.4, 16), (0.2, 32), (0.2, 64), (0.4, 64)]
    ],

    # BI-GRU Cases
    *[
        {
            "model_type": "bi_gru",
            "dataset_variant": dataset_variant,
            "hidden_size": 128,
            "dropout": dropout,
            "learning_rate": 0.0001,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for dropout, batch_size in [(0.4, 32), (0.4, 64), (0.2, 64), (0.0, 32), (0.0, 64)]
    ],

    # LSTM Cases
    *[
        {
            "model_type": "lstm",
            "dataset_variant": dataset_variant,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for hidden_size, dropout, learning_rate, batch_size in [
            (128, 0.0, 0.001, 32),
            (64, 0.2, 0.001, 32),
            (128, 0.0, 0.001, 64),
            (128, 0.2, 0.001, 64),
            (64, 0.2, 0.0001, 64),
        ]
    ],

    # BI-LSTM Cases
    *[
        {
            "model_type": "bi_lstm",
            "dataset_variant": dataset_variant,
            "hidden_size": hidden_size,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for hidden_size, dropout, learning_rate, batch_size in [
            (64, 0.2, 0.001, 32),
            (128, 0.4, 0.0001, 64),
            (128, 0.2, 0.001, 32),
            (32, 0.4, 0.001, 16),
            (128, 0.4, 0.001, 32),
        ]
    ],

    # Transformer Cases (No hidden_size)
    *[
        {
            "model_type": "transformer",
            "dataset_variant": dataset_variant,
            "dropout": dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
        }
        for dataset_variant in ["Binary_Data_1000", "Binary_Data"]
        for dropout, learning_rate, batch_size in [
            (0.0, 0.001, 16),
            (0.0, 0.001, 64),
            (0.0, 0.0001, 16),
            (0.2, 0.001, 32),
            (0.0, 0.001, 32),
        ]
    ],

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
