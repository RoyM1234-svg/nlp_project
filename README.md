# Run Verifier Script Notebook

This README provides exact instructions on how to run the `run_verifier_script.ipynb` notebook in a Google Colab environment (or a local Jupyter setup) to train the verifier model.

---

## Prerequisites

* **Environment**: Google Colab (recommended) or local Jupyter Notebook
* **Git**: Installed and available in your PATH
* **Python**: Version 3.7 or higher
* **W\&B Account**: You need an API key from [Weights & Biases](https://wandb.ai/) to log experiments

---

## Instructions

1. **Open the Notebook**
   Open `run_verifier_script.ipynb` in Colab or your local Jupyter interface.

2. **Mount Google Drive** (Colab only)

   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   ```

3. **Navigate to Your Drive**

   ```bash
   %cd /content/drive/MyDrive/
   ```

4. **Clone the Repository**

   ```bash
   !git clone https://github.com/RoyM1234-svg/nlp_project.git
   ```

5. **Enter the Project Directory**

   ```bash
   %cd nlp_project
   ```

6. **Pull Latest Changes & Switch to Main**

   ```bash
   !git pull
   !git checkout main
   ```

7. **Install Dependencies & Login to W\&B**

   ```bash
   !pip install wandb datasets transformers
   !wandb login <YOUR_WANDB_API_KEY>
   ```

   * Replace `<YOUR_WANDB_API_KEY>` with your actual W\&B key.

8. **Change into the `musr` Folder**

   ```bash
   %cd musr
   ```

9. **Run the Verifier Training Script**

    ```bash
    !python train_verifier.py \
      --model_name google/bigbird-roberta-base \
      --lr 5e-6 \
      --batch_size 2 \
      --save_model_path verifier_model \
      --output_dir verifier_model \
      --num_train_epochs 3 \
      --bf16
    ```

    * Adjust any flags (e.g., `--num_train_epochs`, `--lr`, `--batch_size`) to your needs.

---
