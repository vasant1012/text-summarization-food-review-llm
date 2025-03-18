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
 - I recommend you to increase the training dataset size and build the model. The generalization capability of a deep learning model enhances with an increase in the training dataset size

 - Try implementing **Bi-Directional LSTM** which is capable of capturing the context from both the directions and results in a better context vector

 - Use the **beam search strategy** for decoding the test sequence instead of using the greedy approach (argmax)

 - Evaluate the performance of your model based on the **BLEU score**

 - Implement **pointer-generator networks** and **coverage mechanisms**