import os
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import sentencepiece as spm

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, disable_progress_bar
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("🧹 GPU memory cleared")
        
        
def load_full_dataset(dataset_path: str, dataset_splits: Dict[str, str]) -> DatasetDict:
    """
    Load Shipibo-Spanish parallel corpus with Sentiment labels from device
    
    Args:
        dataset_path: Path to the dataset file (e.g., "data/train.parquet")

    Returns:
        DatasetDict with train/validation/test splits
    """
    print(f"\n📥 Loading dataset: {dataset_path}")

    # Load the dataset
    dataset = load_dataset("parquet", data_files=dataset_splits, split=None)

    # Display statistics
    print("\n📊 Dataset Statistics:")
    for split in dataset.keys():
        print(f"   {split}: {len(dataset[split])} examples")
        if len(dataset[split]) > 0:
            # Show first example
            example = dataset[split][0]
            print(f"   Example Shipibo: {example.get('shp', example.get('shipibo', 'N/A'))[:50]}...")
            print(f"   Example Spanish: {example.get('spa', example.get('spanish', 'N/A'))[:50]}...")
    
    return dataset

def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted"),
        "confusion_matrix": confusion_matrix(labels, preds)
    }
    
def prepare_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    """
    Load and configure tokenizer for Shipibo→Spanish
    """
    print(f"\n🔧 Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"   Tokenizer loaded from: {tokenizer_path}")
    return tokenizer

def prepare_model(model_path: str, num_labels: int):
    """
    Get the pre-trained model locally.
    
    Args:
        model_path: Directory where the model is saved.
    """
    print(f"\n⬇️  Loading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.config.label2id = {'NEG': 0, 'NEU': 1, 'POS': 2}
    model.config.id2label = {0: 'NEG', 1: 'NEU', 2: 'POS'}
    print(f"   Model loaded from: {model_path}")
    return model

def download_model(model_path: str, output_dir: str):
    """
    Download and save the pre-trained model locally.
    
    Args:
        model_path: Directory where the model is saved.
        output_dir: Directory to save the downloaded model.
    """
    print(f"\n⬇️  Downloading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.save_pretrained(output_dir)
    print(f"   Model saved to: {output_dir}")
    
def download_tokenizer(tokenizer_path: str, output_dir: str):
    """
    Download and save the tokenizer locally.
    
    Args:
        tokenizer_name: Name of the tokenizer to download.
        tokenizer_path: Directory where the tokenizer is saved.
        output_dir: Directory to save the downloaded tokenizer.
    """
    print(f"\n⬇️  Downloading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)
    print(f"   Tokenizer saved to: {output_dir}")
 
 
def norm(x):
    return x.strip().lower()    
    
def dedupe_bilingual(dataset, langs=("spa", "shp")) -> Dataset:
    """
    Remove duplicate sentence pairs in a bilingual dataset.
    
    Args:
        dataset: Hugging Face Dataset containing bilingual sentence pairs.
        langs: Tuple of language column names (default: ("spa", "shp")).
    """
    seen = set()
    rows = []
    for row in dataset:
        pair = (norm(row[langs[0]]), norm(row[langs[1]]))
        if pair not in seen:
            seen.add(pair)
            rows.append(row)
    return Dataset.from_list(rows)

def overlap_percentage(a, b):
    """
    a, b are Python sets
    Returns: % of elements in a that also appear in b
    """
    if len(a) == 0:
        return 0
    return len(a & b) / len(a) * 100

def remove_overlaps_bilingual(dataset_from, dataset_against, langs=("spa", "shp")):
    """
    dataset_from, dataset_against: HuggingFace Datasets objects
    Returns: a new Dataset object with rows from dataset_from that do not
             appear in dataset_against (based on bilingual pairs)
    """
    pairs_against = {
        (norm(r[langs[0]]), norm(r[langs[1]])) for r in dataset_against
    }
    return dataset_from.filter(
        lambda x: (norm(x[langs[0]]), norm(x[langs[1]])) not in pairs_against
    )
    
def predict_dataset(pipeline, dataset, source_column: str, batch_size: int = 16) -> (List[str], List[str], List[float]):
    """
    Use a Hugging Face pipeline to predict labels for a dataset.
    
    Args:
        pipeline: Hugging Face pipeline for prediction.
        dataset: Hugging Face Dataset to predict on.
        source_column: Name of the column containing the input texts.
        batch_size: Number of samples per batch for prediction.
    Returns:
        Lists of texts, predicted_labels and scores.
    """
    texts = dataset[source_column]
    predicted_labels = []
    scores = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Predicting"):
        batch_texts = texts[i:i+batch_size]
        predictions = pipeline(batch_texts, truncation=True)
        for pred in predictions:
            predicted_labels.append(pred['label'])
            scores.append(pred['score'])
    
    return texts, predicted_labels, scores

def evaluate_dataset(pipeline, dataset, source_column: str = "shp", label_column: str = "sentiment_label", batch_size: int = 16) -> Dict[str, Any]:
    """
    Evaluate a Hugging Face pipeline on a labeled dataset.
    
    Args:
        pipeline: Hugging Face pipeline for prediction.
        dataset: Hugging Face Dataset to evaluate on.
        source_column: Name of the column containing the input texts.
        label_column: Name of the column containing the true labels.
        batch_size: Number of samples per batch for prediction.
    Returns:
        Dictionary with accuracy, balanced_accuracy, precision, recall, and f1 score.
    """
    texts = dataset[source_column]
    true_labels = dataset[label_column]
    predicted_labels = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch_texts = texts[i:i+batch_size]
        predictions = pipeline(batch_texts, truncation=True)
        for pred in predictions:
            predicted_labels.append(pred['label'])
    
    metrics = compute_metrics(predicted_labels, true_labels)
    return metrics

def train_model(base_model, base_tokenizer, train_dataset, val_dataset, test_dataset, training_args, output_dir):
    """Train the sentiment analysis model."""
    
    def preprocess_function(dataset: Dataset):
        """Tokenize and encode the Shipibo texts."""
        encodings = base_tokenizer(dataset["shp"], truncation=True, padding=False, max_length=256)
        # Map sentiment labels to IDs
        label_map = {'NEG': 0, 'NEU': 1, 'POS': 2}
        encodings['sentiment_label'] = [label_map[label] for label in dataset['sentiment_label']]
        return encodings

    # Metrics function
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        accuracy = accuracy_score(labels, predictions)
        precision = precision_score(labels, predictions, average='weighted', zero_division=0)
        recall = recall_score(labels, predictions, average='weighted', zero_division=0)
        f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
        balanced_acc = balanced_accuracy_score(labels, predictions)
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'balanced_accuracy': balanced_acc
        }
    
    def prepare_dataset(dataset: Dataset):
        """Prepare dataset for training/evaluation."""
        dataset = dataset.map(preprocess_function, batched=True)
        columns_to_remove = [ x for x in dataset.column_names if x not in ['input_ids', 'attention_mask', 'label']]
        dataset = dataset.remove_columns(columns_to_remove)
        return dataset

    
    # Tokenize datasets
    tokenized_train = prepare_dataset(train_dataset)
    tokenized_validation = prepare_dataset(val_dataset)
    tokenized_test = prepare_dataset(test_dataset)
    
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer=base_tokenizer)
    

    # Initialize Trainer
    trainer = Trainer(
        model=base_model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_validation,
        tokenizer=base_tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[early_stopping_callback],
    )

    # Train the model
    trainer.train()

    # Evaluate on test set
    test_results = trainer.evaluate(eval_dataset=tokenized_test)
    print("Test Results:", test_results)

    # Save the trained model and tokenizer
    trainer.save_model(output_dir)
    base_tokenizer.save_pretrained(output_dir)