import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
import pandas as pd
from sklearn.model_selection import train_test_split
from train import TextSummarizationDataset
from transformers import Trainer, TrainingArguments,EarlyStoppingCallback
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# ====== Load Model and Tokenizer ======
def load_model(model_path, device):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model architecture
    model = T5ForConditionalGeneration.from_pretrained("t5-base")  # Change if needed

    state_dict = torch.load(model_path, map_location=device)  # Load to CPU first
    model.load_state_dict(state_dict, strict=False)  # Allow partial loading
    model.to(device)
    return model

def load_data(input_data, sample_size, test_size=0.2):
    traning_set = input_data.sample(sample_size)

    train_df, eval_df = train_test_split(traning_set, test_size=test_size, random_state=42)

    return train_df, eval_df


def load_trainer(train_df, eval_df, model):
    # Load tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base") 

    # Create datasets
    train_dataset = TextSummarizationDataset(train_df, tokenizer)
    eval_dataset = TextSummarizationDataset(eval_df, tokenizer)


    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',
        evaluation_strategy="epoch",  # Evaluate at the end of each epoch
        save_strategy="epoch",        # Save at the end of each epoch
        load_best_model_at_end=True,  # Load the best model at the end
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=50,
        remove_unused_columns=False,
        logging_dir='./logs',
        save_total_limit=3,
        fp16=True,
    )

    # Define early stopping callback
    early_stopping_callback = EarlyStoppingCallback(
        early_stopping_patience=3,  # Number of epochs with no improvement after which training will stop
        early_stopping_threshold=0.0   # Minimum improvement to consider for early stopping
    )

    # Initialize the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,  # Use the evaluation dataset
        tokenizer=tokenizer,
        callbacks=[early_stopping_callback]
    )
    return trainer


if __name__ == '__main__':
    # Load model and tokenizer
    model = load_model('path_to_checkpoint', device='cuda')
    df = pd.read_csv('preprocessed_df.csv')
    train_df, eval_df = load_data(df, 30000, test_size=0.2)
    trainer = load_trainer(train_df, eval_df)
    # Start training
    trainer.train()