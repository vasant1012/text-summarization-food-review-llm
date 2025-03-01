# text-summarization-food-review-llm
Create end-to-end Deep learning pipeline using transfer learning from LLM

## Overview
This project focuses on text summarization using the **Amazon Food Review** dataset. The goal is to generate concise and meaningful summaries of customer reviews.

## Model and Framework
- Utilized the **BART Transformer (facebook/bart-large-cnn)** and **T5-base** for training.  
- Implemented using **PyTorch** for model architecture and training.  

## Dataset
The **Amazon Food Review** dataset consists of customer reviews, which serve as input for training the summarization model.

## Usage
1. Preprocess the dataset (tokenization, text cleaning, etc.).
2. Train the model using BART or T5-base.
3. Evaluate the model's summarization performance.

## Dependencies
- Python 3.8  
- PyTorch  
- Transformers (Hugging Face: "t5-base")  
- Pandas, NumPy  

## Future Enhancements
- Experiment with fine-tuning different transformer architectures.  
- Optimize inference speed for deployment.  
