# WHODUNIT: A new method for solving mystery stories using self consistency and a verifier

## Project Overview

The project implements a multi-stage pipeline for solving detective puzzles:

1. **Chain-of-Thought (CoT) Generation**: LLMs generate step-by-step reasoning for detective mysteries
2. **Verification**: A trained verifier model scores the quality of the reasoning 
3. **Self-Consistency**: Multiple generations are created and the best one is selected using verifier scores
4. **Final Answer Extraction**: The selected reasoning is used to determine the guilty suspect

### Supported Models
- ** LLaMA (Qunatized)* .We evaluated with meta-llama/Llama-3.1-8B-Instruct(quntaized for memory efficiency). If you have larger memory you are welcomed to try a bigger model.

## Repository Structure

```
nlp_project/
├── analysis/                    # Result analysis scripts
│   ├── analyze_results.py      # Comprehensive result analysis
│   ├── analyze_test_cases.py   # Test case verification
│   └── prediction_analysis_plot.py # Visualization generation
├── custom_datasets/            # Custom dataset implementations
├── data/                       # Dataset files
│   ├── detective-puzzles.csv   # Main detective puzzle dataset
│   └── test_cases.csv         # Test cases for verification
├── data_loaders/              # Data loading utilities
├── models/                    # Model implementations
│   ├── cot_models/           # Chain-of-thought generation models
│   ├── final_answer_models/  # Final answer generation models
│   ├── detective_model.py    # Base detective model class
│   └── verifier_model.py     # Verifier model implementation
├── musr/                     # MUSR dataset and verifier training
│   ├── train_verifier.py     # Verifier training script
│   └── verifier_training_data/ # Training data for verifier
├── results/                  # Experimental results
│   ├── k_1/                 # Results with k=1 self-consistency
│   ├── k_3/                 # Results with k=3 self-consistency
├── plots/                   # Generated visualizations
├── evaluate_model.py        # Main evaluation pipeline
└── utils.py                # Utility functions
```

## Reproducing Project Results

### Step 1: Setup Environment

1. **Open a new Google Colab notebook** and ensure you're using a GPU runtime:
   - Go to `Runtime` → `Change runtime type` → Select `A100 GPU` as Hardware accelerator

2. **Mount Google Drive**:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   %cd /content/drive/MyDrive/
   ```

3. **Clone the repository and install dependencies**:
   ```python
   !git clone https://github.com/YOUR_USERNAME/nlp_project.git
   %cd nlp_project
   
   # Install dependencies
   !pip install wandb datasets transformers accelerate bitsandbytes

   # Setup Weights & Biases
   import wandb
   wandb.login()  
   ```

### Step 2: Train the Verifier Model

```python
# Navigate to musr directory
%cd musr

# Train the verifier model
!python train_verifier.py 
--model_name google/bigbird-roberta-base 
--lr 5e-6 
--batch_size 2 
--save_model_path verifier_model 
--output_dir verifier_model 
--num_train_epochs 3 
--bf16

# Return to project root
%cd ..
```

### Step 3: Run Main Evaluation Pipeline

1. **Login to Hugging Face** and ensure you have the license for "meta-llama/Llama-3.1-8B-Instruct":
   ```python
   from huggingface_hub import login
   login() 
   ```

2. **Configure memory allocation** to avoid fragmentation:
   ```python
   %env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
   ```

3. **Run the evaluation pipeline**:
   ```python
   # k=1 (baseline)
   !python evaluate_model.py \
      --model_type "llama" \
      --model_path "meta-llama/Llama-3.1-8B-Instruct" \
      --batch_size 20 \
      --k 1

   # k=3 (self-consistency)
   !python evaluate_model.py \
      --model_type "llama" \
      --model_path "meta-llama/Llama-3.1-8B-Instruct" \
      --batch_size 20 \
      --k 3
   ```

### Step 4: Generate Analysis and Visualizations

```python
# Comprehensive result analysis
!python analysis/analyze_results.py results/k_1/results_llama_final_answer_k_1.csv --model-name "LLaMA k=1"
!python analysis/analyze_results.py results/k_3/results_llama_final_answer_k_3.csv --model-name "LLaMA k=3"

# Generate prediction analysis plots for k=1
!python analysis/prediction_analysis_plot.py \
  results/k_1/results_llama_final_answer_k_1.csv \
  --verifier results/k_1/results_llama_verifier_k_1.csv \
  --output plots/llama_k_1/analysis

# Generate prediction analysis plots for k=3
!python analysis/prediction_analysis_plot.py \
  results/k_3/results_llama_final_answer_k_3.csv \
  --verifier results/k_3/results_llama_verifier_k_3.csv \
  --output plots/llama_k_3/analysis
```

### Generated Files

After running the complete pipeline, you will have:

1. **Model Results**: 
   - `results_<model>_cot_k_<k>.csv` - Chain-of-thought generations
   - `results_<model>_verifier_k_<k>.csv` - Verifier scores  
   - `results_<model>_final_answer_k_<k>.csv` - Final predictions

2. **Analysis Plots**:
   - Prediction distribution plots (Unknown vs Correct vs Incorrect)
   - Mean verifier probability plots
   - Self-consistency improvement plots

3. **Performance Metrics**:
   - Accuracy scores
   - Error analysis
   - Verifier probability distributions
