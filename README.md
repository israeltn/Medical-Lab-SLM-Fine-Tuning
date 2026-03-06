# Towards Efficient Clinical Reasoning: Adapting Distilled Reasoning Models for Laboratory Diagnostics in Resource-Constrained Healthcare Environments


**Background:** Clinical decision support in African healthcare settings is often limited by a lack of specialized personnel and the high computational costs associated with modern AI. While Large Language Models (LLMs) offer reasoning capabilities, their deployment is hindered by hardware constraints and data privacy concerns in remote regions. This study evaluates the performance and efficiency of a distilled reasoning model tailored for automated laboratory result analysis in the Nigerian health infrastructure.

**Design/Methods:** We developed Med-Lab-FineTuned-Qwen2.5-1.5B by adapting the Qwen2.5-1.5B-Instruct model using Low-Rank Adaptation (LoRA) and 4-bit NormalFloat quantization. The model was trained on a structured dataset of laboratory diagnostics to identify abnormalities and provide clinical recommendations using a Short-Chain-of-Thought (Short-CoT) strategy. To ensure deployment scalability in constrained environments such as lab software and hospital edge devices, the model was converted to GGUF format (q4_k_m). This allows for offline, CPU-based inference on standard consumer-grade hardware (typically 4GB-16GB RAM) without requiring specialized GPU accelerators.

**Results:** The fine-tuned 1.5B model demonstrated high fidelity in clinical reasoning, updating only 1.18% of total parameters while maintaining the ability to categorize severity and provide detailed clinical summaries. Resource metrics indicated that the model operates effectively under a ~900MB RAM footprint during GGUF-based inference. This ensures full compatibility with legacy hardware common in local health centers and seamless integration into local laboratory information systems (LIS). The approach successfully aligned with FAIR principles by providing a secure, reusable diagnostic tool that functions without high-bandwidth internet dependency.

**Conclusion:** This AI-aided diagnostic tool meets the 'efficient is essential' requirement for resource-constrained environments. By bridging the gap between high-end research and on-the-ground deployment, the Qwen2.5-1.5B quantized model reinforces the potential of specialized reasoning agents to support health equity and national diagnostic programs in Nigeria.

**Keywords:** AI-aided screening, Clinical Reasoning, Resource-Constrained AI, Qwen2.5-1.5B, LoRA, GGUF Quantization, CPU Inference, Hospital Edge Devices, Lab Software Optimization.

---

## Technical Details

This repository contains a Jupyter notebook and supporting files for fine-tuning the `unsloth/Qwen2.5-1.5B-Instruct` model on a custom medical lab test dataset. The notebook is optimized to run on a local machine with limited GPU memory (e.g., RTX 2060 6GB VRAM) using memory-saving techniques such as 4-bit quantization, LoRA adapters, and Unsloth optimizations.

### Contents

- `fine_tune_unsloth.ipynb` & `fine_tune_unslothQwen2.5-1.5B_colab.ipynb` - Main notebooks demonstrating setup, model loading, dataset preparation, training, inference, and saving the fine-tuned model (including merging to GGUF).
- `fine_tuning_lab_tests.jsonl` - Example dataset of lab test conversations in JSONL format, complete with expected clinical reasoning.
- `.env` - (not checked in) Contains `HUGGINGFACE_TOKEN` and optionally `WANDB_API_KEY`.
- `outputs/` - Directory where training outputs are stored.

The dataset consists of **conversational records** spread across **10 common and vital laboratory tests**:

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

### Getting Started

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
   pip install "unsloth @ git+https://github.com/unslothai/unsloth.git"
   pip install --upgrade --no-cache-dir --no-deps unsloth_zoo
   pip install datasets wandb trl peft transformers bitsandbytes python-dotenv
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
   Follow the cells sequentially.

### Notes

- The target model is `Qwen2.5-1.5B-Instruct` due to its excellent balance of reasoning capabilities and small parameter count, allowing for comfortable fine-tuning on 6GB VRAM.
- The notebook sets `max_seq_length` to 1024, `per_device_train_batch_size` to 2, and `gradient_accumulation_steps` to 4. 
- The training dataset uses a custom `alpaca_prompt` format that forces the model to generate a `Reasoning:` thought-process block before delivering a medical conclusion.
- The notebook fully supports converting the final trained model into an optimized `GGUF (q4_k_m)` format. This lets you serve the model offline on a hospital CPU using legacy hardware (e.g. using `llama-cpp-python`).
- Training uses LoRA adapters (rank 16) and the `adamw_8bit` optimizer to minimize memory usage.

### License

This project is provided for educational purposes. Ensure compliance with the licensing terms of models and libraries used (e.g., Unsloth, Transformers, Qwen).

### Acknowledgements

- [Unsloth](https://github.com/unslothai/unsloth) for memory-efficient model wrappers.
- Hugging Face Transformers and Datasets.
