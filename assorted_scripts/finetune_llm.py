'''
A script to finuetune FLAN-T5 on the IMDB dataset for sentiment analysis task, 
so that the model can classify movie reviews as positive or negative.
'''
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

# load pretrained model and tokenizer
model_name = "google/flan-t5-base"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# load and preprocess dataset
dataset = load_dataset("imdb")

def preprocess_function(examples):

	inputs = ["sentiments: " + text for text in examples["text"]]
	targets = examples["label"]
	model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")
	labels = tokenizer(targets, max_length=2, truncation=True, padding="max_length")
	model_inputs["labels"] = labels["input_ids"]
	
	return model_inputs

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
	output_dir = "./results",
	num_train_epochs = 3,
	per_device_train_batch_size = 8,
	per_device_eval_batch_size = 8,
	warmup_steps = 500,
	weight_decay = 0.01,
	logging_dir = "./logs"
)

# Initialize trainer
trainer = Trainer(
	model = model,
	args = training_args,
	train_dataset = tokenized_datasets["train"],
	eval_dataset = tokenized_datasets["test"]
)

# Finetune the model
trainer.train()

# Save the finetuned model
model.save_pretrained("./flan-t5-sentiment")
tokenizer.save_pretrained("./flan-t5-sentiment")
