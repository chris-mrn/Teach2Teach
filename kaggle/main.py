import os
import subprocess

# Clone the repository
subprocess.run(
    ["git", "clone", "--branch", "text", "--single-branch", "https://github.com/LounesMD/LLaDA_Arithmetic.git"],
    check=True,
)
os.chdir("LLaDA_Arithmetic")


# Define hyperparameters

device = "cuda"
# Run the training script
subprocess.run(
    [
        "python",
        "main.py",
    ],
    check=True,
)
