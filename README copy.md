# Medical Lab Result Fine-Tuning

This repository contains a Jupyter notebook and supporting files for fine-tuning the
`unsloth/DeepSeek-R1-Distill-Llama-8B` model on a custom medical lab test dataset. The
notebook is optimized to run on a local machine with limited GPU memory (e.g., RTX 2060
6GB VRAM) using memory-saving techniques such as 4-bit quantization, LoRA adapters, and
sequence length reduction.

## Contents

- `fine_tune_unsloth.ipynb` - Main notebook demonstrating setup, model loading, dataset
  preparation, training, inference, and saving the fine-tuned model.
- `fine_tuning_lab_tests.jsonl` - Example dataset of lab test conversations in JSONL format.
- `.env` - (not checked in) Contains `HUGGINGFACE_TOKEN` and optionally `WANDB_API_KEY`.
- `myenv/` - Python virtual environment used for dependencies.
- `outputs/` - Directory where training outputs are stored.

The dataset consists of **1000 conversational records** spread evenly across **10 common and vital laboratory tests**:

1.  Glucose (Fasting & Random)
2.  Hemoglobin (Hgb)
3.  Creatinine (Serum)
4.  Sodium (Serum)
5.  Potassium (Serum)
6.  Cholesterol (Total)
7.  White Blood Cell (WBC) Count
8.  Platelets
9.  Alanine Aminotransferase (ALT)
10. Calcium (Serum)


## Getting Started

1. **Clone the repository**:
   ```bash
   git clone <repo-url> medlab
   cd medlab
   ```

2. **Create and activate a Python environment** (Python 3.11 recommended):
   ```bash
   python -m venv myenv
   source myenv/bin/activate
   pip install --upgrade pip
   ```

3. **Install dependencies** (notebook cell also handles this):
   ```bash
   pip install -r requirements.txt
   # or manually: unsloth, unsloth_zoo, datasets, wandb, trl, peft, transformers, bitsandbytes, python-dotenv
   ```

4. **Create a `.env` file** with your Hugging Face token:
   ```ini
   HUGGINGFACE_TOKEN=hf_...
   WANDB_API_KEY=...
   ```

5. **Run the notebook**:
   ```bash
   jupyter lab fine_tune_unsloth.ipynb
   ```
   Follow the cells sequentially. The model will automatically adjust sequence length
   to fit your GPU.

## Notes

- The notebook includes logic to progressively reduce `max_seq_length` (512 → 256 → 128 → 64)
  to fit within VRAM constraints. For RTX 2060 (6GB), a maximum of 64 tokens is required when
  using the fused cross-entropy loss.
- Training uses LoRA adapters (rank 8) and the `adamw_8bit` optimizer to minimize memory usage.
- It is possible to save only adapter weights (~150MB) or merge into a full 16-bit model (~16GB).

## License

This project is provided for educational purposes. Ensure compliance with the licensing terms of
models and libraries used (e.g., UnsLoth, Transformers).

## Acknowledgements

- [UnsLoth](https://github.com/unslothai/unsloth) for memory-efficient model wrappers.
- Hugging Face Transformers and Datasets.
- The RTX 2060 experiments were adapted from local-4GB-finetuning examples.
