from datasets import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import Trainer, TrainingArguments,EarlyStoppingCallback
from torch.utils.data import Dataset
from transformers import AutoTokenizer, T5ForConditionalGeneration

# Load T5 model
model = T5ForConditionalGeneration.from_pretrained("t5-base")

class TextSummarizationDataset(Dataset):
    def __init__(self, dataframe, tokenizer, text_max_length=30, summary_max_length=8):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.text_max_length = text_max_length
        self.summary_max_length = summary_max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        # Extract input text and summary
        text = self.dataframe.iloc[idx]["text"]
        summary = self.dataframe.iloc[idx]["summary"]

        # Tokenize input text
        input_encodings = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Tokenize summary
        target_encodings = self.tokenizer(
            summary,
            max_length=self.summary_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )

        # Remove batch dimension
        input_ids = input_encodings["input_ids"].squeeze(0)
        attention_mask = input_encodings["attention_mask"].squeeze(0)
        labels = target_encodings["input_ids"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }
    

def training_pipeline(training_set, tokenizer):
    train_df, eval_df = train_test_split(training_set, test_size=0.2, random_state=42)

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
        num_train_epochs=1,
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


# Start training
if __name__ == '__main__':
    var = TextSummarizationDataset(dataframe, tokenizer)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    df = pd.read_csv('preprocessed_df.csv')
    training_set = df.sample(30000)
    var.training_pipeline(training_set, tokenizer)