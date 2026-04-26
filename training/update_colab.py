import json
import os

with open("training/trajectory_optimizer.py", "r", encoding="utf-8") as f:
    traj_code = f.read()

notebook = {
 "nbformat": 4,
 "nbformat_minor": 0,
 "metadata": {
  "colab": {
   "provenance": [],
   "name": "Policy-to-Logic Training"
  },
  "kernelspec": {
   "name": "python3",
   "display_name": "Python 3"
  },
  "language_info": {
   "name": "python"
  }
 },
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Policy-to-Logic RL Environment \u2014 Training Notebook\n",
    "\n",
    "This notebook runs the **reward-guided trajectory optimization loop** against the deployed environment.\n",
    "\n",
    "**What it does:**\n",
    "1. Connects to the live HF Spaces environment\n",
    "2. Runs 8 episodes per task (3 tasks = 24 total episodes)\n",
    "3. Accumulates high-reward trajectories as few-shot examples\n",
    "4. Generates training evidence plots (reward curve, accuracy curve, improvement chart)\n",
    "5. Logs everything to Weights & Biases"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 1: Install dependencies\n",
    "!pip install openai requests matplotlib numpy wandb"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 2: Configuration\n",
    "import os\n",
    "\n",
    "# SET THESE BEFORE RUNNING\n",
    "HF_TOKEN = \"\"  # Your Hugging Face token with inference access\n",
    "ENV_URL = \"https://godreign-policy2logic.hf.space\"  # Your deployed environment URL\n",
    "WANDB_API_KEY = \"\" # Your Wandb API key\n",
    "\n",
    "os.environ[\"HF_TOKEN\"] = HF_TOKEN\n",
    "os.environ[\"ENV_BASE_URL\"] = ENV_URL\n",
    "if WANDB_API_KEY:\n",
    "    os.environ[\"WANDB_API_KEY\"] = WANDB_API_KEY\n",
    "\n",
    "print(f\"Environment URL: {ENV_URL}\")\n",
    "print(f\"HF Token set: {'Yes' if HF_TOKEN else 'NO - MUST SET THIS'}\")\n",
    "print(f\"Wandb Token set: {'Yes' if WANDB_API_KEY else 'NO - WILL PROMPT'}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 3: Verify environment is reachable\n",
    "import requests\n",
    "\n",
    "r = requests.get(f\"{ENV_URL}/health\")\n",
    "print(f\"Status: {r.status_code}\")\n",
    "print(f\"Response: {r.json()}\")\n",
    "\n",
    "r2 = requests.get(f\"{ENV_URL}/tasks\")\n",
    "tasks = r2.json()\n",
    "print(f\"\\nAvailable tasks: {list(tasks['tasks'].keys())}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 4: Wandb login — run this before training\n",
    "import wandb\n",
    "wandb.login()  # Will prompt for API key if WANDB_API_KEY is not set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 5: Training loop implementation (full trajectory_optimizer.py)\n"
   ] + [line + "\n" for line in traj_code.split("\n")[:-1]] + [traj_code.split("\n")[-1]]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 6: Run training loop\n",
    "loop = TrainingLoop(ENV_URL, HF_TOKEN)\n",
    "metrics = loop.run()\n",
    "print(f\"\\nTotal episodes run: {len(metrics)}\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 7: Generate plots and display inline\n",
    "save_plots(metrics)\n",
    "\n",
    "from IPython.display import Image, display\n",
    "display(Image(\"training/plots/reward_curve.png\"))\n",
    "display(Image(\"training/plots/accuracy_curve.png\"))\n",
    "display(Image(\"training/plots/improvement_chart.png\"))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 8: Display wandb run link\n",
    "print(f\"Wandb run: https://wandb.ai/YOUR_USERNAME/policy-to-logic-rl\")\n",
    "print(\"Add this link to your README under Deliverables.\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Cell 9: Download plots to commit to repo\n",
    "from google.colab import files\n",
    "\n",
    "files.download(\"training/plots/reward_curve.png\")\n",
    "files.download(\"training/plots/accuracy_curve.png\")\n",
    "files.download(\"training/plots/improvement_chart.png\")\n",
    "files.download(\"training/plots/metrics_latest.json\")\n",
    "\n",
    "print(\"Downloaded. Now commit these files to: training/plots/ in your repo.\")"
   ]
  }
 ]
}

with open("training/colab_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)

print("Colab Notebook updated successfully with UTF-8 encoding")
