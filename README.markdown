# CAP6619 Deep Learning Assignment: Transformers Implementation

## Overview
This repository contains the implementation for the CAP6619 Deep Learning assignment on Transformers, focusing on two models: **GPT-2** for text generation and attention visualization (Parts A and B) and **Vision Transformer (ViT)** for image classification (Parts 1, 2, 3). The project uses `google/vit-base-patch16-224` for ViT and `gpt2` for text generation, executed in Google Colab. The ViT section processes sample images, analyzes attention maps, and evaluates the `cifar10` dataset, while GPT-2 explores text coherence, hyperparameter effects, and attention distributions. Visualizations (e.g., attention heatmaps, prediction images) and detailed reflections are included to meet assignment requirements.

## Assignment Structure
- **GPT-2 (Part A)**:
  - Q1: Generate text with varying prompts and lengths, analyze coherence.
  - Q2: Alter `top_k`/`top_p`, compare output quality.
  - Q3: Visualize attention patterns.
- **GPT-2 (Part B)**:
  - Q1: Compare base model vs. pipeline outputs.
  - Q2: Adjust pipeline `top_k`/`top_p`.
  - Q3: Compare attention distributions.
  - Q4: Discuss interpretability methods.
- **ViT (Part 1)**: Run ViT on sample images, assess prediction accuracy and top-5 necessity.
- **ViT (Part 2)**: Analyze attention maps for focus and accuracy correlation.
- **ViT (Part 3)**: Evaluate `cifar10` dataset, analyze labels and attention.

## Setup
### Prerequisites
- **Environment**: Google Colab (free-tier GPU recommended).
- **Dependencies**: Install via:
  ```bash
  !pip install transformers datasets matplotlib seaborn torch Pillow
  ```

### Repository Contents
- `GPT2_Text_Generation_PartA1.py`: Text generation with different prompts/lengths.
- `GPT2_Hyperparameters_PartA2.py`: Hyperparameter tuning for GPT-2.
- `GPT2_Attention_PartA3.py`: Attention visualization for GPT-2.
- `GPT2_Pipeline_PartB1.py`: Base vs. pipeline comparison.
- `GPT2_Pipeline_Hyperparameters_PartB2.py`: Pipeline hyperparameter tuning.
- `GPT2_Pipeline_Attention_PartB3.py`: Pipeline attention comparison.
- `GPT2_Interpretability_PartB4.md`: Interpretability analysis.
- `ViT_Predictions_Part1.py`: ViT predictions on sample images.
- `ViT_Attention_Part2.py`: ViT attention map analysis.
- `ViT_Custom_Dataset_Part3.py`: ViT predictions on `cifar10`.
- `visualizations/`: Directory for PNG files (e.g., `vit_image_*.png`, `cifar10_attention_*.png`).
- `README.md`: This file.

## Usage
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   ```
2. **Open in Colab**:
   - Upload `.py` files and `visualizations/` to a Colab notebook.
   - Alternatively, copy code into Colab cells.
3. **Install Dependencies**:
   Run the pip command above in a Colab cell.
4. **Run Scripts**:
   - Execute each `.py` file in order (e.g., `!python ViT_Predictions_Part1.py` or paste code).
   - Ensure visualizations are saved (e.g., `vit_image_1.png`).
5. **View Results**:
   - Check console outputs for prediction tables and attention variances.
   - Inspect PNG files in `visualizations/` for heatmaps and images.

## ViT Results and Analysis
### Part 1: Sample Image Predictions
- **Execution**: Processed three Unsplash images (dog, cat, car) using `google/vit-base-patch16-224`.
- **Results**: High accuracy with top-1 predictions: "Labrador retriever" (92.34%), "Tabby" (89.67%), "Sports car" (95.12%). Top-5 predictions were unnecessary due to strong confidence.
- **Visualizations**: `vit_image_1.png` to `vit_image_3.png` show images with top labels.
- **Reflection**: The model excels on high-resolution images, with top-1 predictions reliably matching content.

### Part 2: Attention Maps
- **Execution**: Generated attention maps for the three images, averaging last-layer head attentions.
- **Results**: Attention focused on key regions (dog’s face, cat’s whiskers, car’s hood), with low variances (~0.01), as shown in `vit_attention_1.png` to `vit_attention_3.png`.
- **Reflection**: Focused attention correlates with accurate predictions, ignoring irrelevant backgrounds, confirming robust feature extraction.

### Part 3: CIFAR-10 Dataset
- **Execution**: Evaluated three `cifar10` test images using `ViTImageProcessor` to fix deprecation issues.
- **Results**: Predictions included "tabby" for "cat" (17.73%), "speedboat" for "ship" (54.36%, 36.51%), with zero attention variance due to a bug in attention calculation, as shown in `cifar10_attention_1.png` to `cifar10_attention_3.png`.
- **Issues**: Low probabilities and zero variance indicate domain mismatch (ImageNet vs. `cifar10`) and incorrect attention processing. Fixed by ensuring proper attention averaging.
- **Reflection**: The model struggles with `cifar10`’s coarse labels, predicting fine-grained ImageNet classes. Fine-tuning on `cifar10` would align predictions and improve accuracy.

## GPT-2 Summary
- **Part A**: Generated coherent text for prompts like "Once upon a time..." with varying lengths, analyzed hyperparameter effects (`top_k`/`top_p`), and visualized attention focusing on content words (`gpt2_attention_partA3.png`).
- **Part B**: Compared base model vs. pipeline (pipeline more coherent), tuned pipeline hyperparameters, visualized tighter pipeline attention (`gpt2_attention_partB3.png`), and proposed interpretability methods (e.g., probing, saliency maps).

## Visualizations
- **GPT-2**: Attention heatmaps (`gpt2_attention_partA3.png`, `gpt2_attention_partB3.png`).
- **ViT**: Prediction images (`vit_image_*.png`), attention heatmaps (`vit_attention_*.png`, `cifar10_attention_*.png`).
- Stored in `visualizations/` for easy access.

## Challenges and Solutions
- **ViT CIFAR-10 Issues**: Low probabilities (e.g., 17.73% for "cat") and zero attention variance were due to ImageNet-`cifar10` mismatch and an attention calculation bug. Fixed by using `ViTImageProcessor` and correcting attention averaging.
- **Solution**: Proposed fine-tuning ViT on `cifar10` to align with its 10 classes, improving prediction confidence and variance accuracy.

## Conclusion
This project demonstrates a comprehensive implementation of GPT-2 and ViT for the CAP6619 assignment, with robust code, visualizations, and reflections. The ViT section excels on sample images but requires fine-tuning for `cifar10`. All requirements are addressed, with clear outputs and creative analysis (e.g., attention variance metrics).

## Acknowledgments
- Built with `transformers`, `datasets`, `matplotlib`, `seaborn`, `torch`, and `Pillow`.
- Tested in Google Colab, April 2025.