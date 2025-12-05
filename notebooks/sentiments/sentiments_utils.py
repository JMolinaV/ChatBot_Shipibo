import os
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import sentencepiece as spm
import requests
import json
import time
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
        
        
def load_parallel_dataset_jsonl(train_path='train.jsonl', 
                          val_path='validation.jsonl', 
                          test_path='test.jsonl'):
    """
    Carga 3 archivos JSONL con formato:
    {"src": "texto español", "tgt": "texto shipibo"}
    
    Retorna: dict con 'train', 'validation', 'test'
    """
    
    def leer_jsonl(filepath):
        """Lee archivo JSONL y convierte src/tgt a spa/shp"""
        spa_texts = []
        shp_texts = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                spa_texts.append(item['spa'])
                shp_texts.append(item['shp'])
        
        return Dataset.from_dict({'spa': spa_texts, 'shp': shp_texts})
    
    print("\n" + "="*70, flush=True)
    print("📂 CARGANDO DATASETS JSONL", flush=True)
    print("="*70 + "\n", flush=True)
    
    # Cargar cada split
    train_dataset = leer_jsonl(train_path)
    val_dataset = leer_jsonl(val_path)
    test_dataset = leer_jsonl(test_path)
    
    print(f"✅ Train:      {len(train_dataset):,} ejemplos", flush=True)
    print(f"✅ Validation: {len(val_dataset):,} ejemplos", flush=True)
    print(f"✅ Test:       {len(test_dataset):,} ejemplos", flush=True)
    print()
    
    return {
        'train': train_dataset,
        'validation': val_dataset,
        'test': test_dataset
    }        
        
def load_parallel_dataset_text(path_es, path_shp):
    # Read each file
    with open(path_es, "r", encoding="utf-8") as f:
        es_lines = [line.strip() for line in f.readlines()]

    with open(path_shp, "r", encoding="utf-8") as f:
        shp_lines = [line.strip() for line in f.readlines()]

    # Safety check (even evil scientists check their assumptions!)
    if len(es_lines) != len(shp_lines):
        raise ValueError(f"Line count mismatch! es={len(es_lines)} shp={len(shp_lines)}")

    # Build dict for HF Dataset
    data = {
        "translation": [
            {"spa": es, "shp": shp}
            for es, shp in zip(es_lines, shp_lines)
        ]
    }

    return Dataset.from_list(data["translation"])        
        
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

def predict_dataset_with_scores(pipeline, dataset, source_column: str = "shp", parallel_source_column="spa", label_column: str|None = None, batch_size: int = 16):
    """
    Use a Hugging Face pipeline to predict labels and scores for a dataset.
    
    Args:
        pipeline: Hugging Face pipeline for prediction.
        dataset: Hugging Face Dataset to predict on.
        source_column: Name of the column containing the input texts.
        parallel_source_column: Name of the column containing the parallel texts.
        label_column: Name of the column containing the true labels (optional).
        batch_size: Number of samples per batch for prediction.
        
    Returns:
        Dataset with text, actual_label, predicted_label and score.
    """
    
    texts = dataset[source_column]
    true_labels = dataset[label_column] if label_column is not None else None
    parallel_texts = dataset[parallel_source_column]
    predicted_labels = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Evaluating"):
        batch_texts = texts[i:i+batch_size]
        predictions = pipeline(batch_texts, truncation=True)
        for pred in predictions:
            predicted_labels.append(pred['label'])
            
    results_dataset = {"text": texts,
                       "predicted_label": predicted_labels,}
    if true_labels is not None:
        results_dataset["actual_label"] = true_labels
        results_dataset["parallel_text"] = parallel_texts

    return Dataset.from_dict(results_dataset)

        

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

def train_model(base_model, base_tokenizer, train_dataset, val_dataset, test_dataset, training_args, output_dir, model_name):
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
    trainer.push_to_hub(model_name)
    

class LLMSentimentAnalyzer:
    def __init__(self, api_url: str, api_key: str, model: str = "gpt-3.5-turbo", max_retries: int = 3):
        """
        Initialize the sentiment analyzer.
        
        Args:
            api_url: Base URL for the LLM API (e.g., "https://api.openai.com/v1")
            api_key: Your API key
            model: Model name to use
        """
        self.api_url = api_url
        self.api_key = api_key
        self.model = model
        self.max_retries = max_retries
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
    
    def analyze_dataset(self, dataset: Dataset, text_column: str, batch_size: int = 10) -> Dataset:
        """
        Analyze sentiment for texts in a Hugging Face Dataset.
        
        Args:
            dataset: Hugging Face Dataset object
            text_column: Name of the column containing text to analyze
            batch_size: Number of texts to process per batch
            
        Returns:
            Dataset with added sentiment columns
        """
        texts = dataset[text_column]
        results = self.analyze_batch(texts, batch_size)
        
        
        # Add results as new columns to the dataset
        dataset = dataset.add_column("sentiment", [r["sentiment"] for r in results])
        dataset = dataset.add_column("sentiment_confidence", [r["confidence"] for r in results])
        dataset = dataset.add_column("sentiment_explanation", [r["explanation"] for r in results])
        
        
        # Print resuts summary
        print("\n" + "="*80)
        print("SENTIMENT ANALYSIS RESULTS (First 5)")
        print("="*80 + "\n")

        for i in range(min(5, len(dataset))):
            example = dataset[i]
            print(f"{i+1}. Text: {example[text_column][:80]}...")
            print(f"   Sentiment: {example['sentiment'].upper()} (Confidence: {example['sentiment_confidence']})")
            print(f"   Explanation: {example['sentiment_explanation']}")
            print()        
        
        
        # Show overall distribution of sentiments
        sentiment_counts = {"positive": 0, "negative": 0, "neutral": 0}
        for example in dataset:
            sentiment = example["sentiment"]
            if sentiment in sentiment_counts:
                sentiment_counts[sentiment] += 1
        print("\n" + "="*80)
        print("OVERALL SENTIMENT DISTRIBUTION")
        print("="*80 + "\n")
        for sentiment, count in sentiment_counts.items():
            print(f"{sentiment}: {count} ({(count/len(dataset))*100:.2f}%)")        
        
        # Show random sample of positive, negative, and neutral texts with explanations
        print("\n" + "="*80)
        print("SAMPLE TEXTS BY SENTIMENT")
        print("="*80 + "\n")
        samples_shown = {"positive": 0, "negative": 0, "neutral": 0}
        for example in dataset.shuffle(seed=42):
            sentiment = example['sentiment']
            if samples_shown[sentiment] < 5:
                print(f"Text: {example[text_column][:100]}...")
                print(f"Sentiment: {sentiment.upper()} (Confidence: {example['sentiment_confidence']})")
                print(f"Explanation: {example['sentiment_explanation']}\n")
                samples_shown[sentiment] += 1
            if all(count >= 5 for count in samples_shown.values()):
                break        
        
        return dataset
    
    def analyze_batch(self, texts: List[str], batch_size: int = 10) -> List[Dict]:
        """
        Analyze sentiment for a list of texts in batches.
        
        Args:
            texts: List of text strings to analyze
            batch_size: Number of texts to process per batch
            
        Returns:
            List of dictionaries with text and sentiment results
        """
        results = []
        
        for i in tqdm(range(0, len(texts), batch_size)):
            batch = texts[i:i + batch_size]
            # print(f"Processing batch {i//batch_size + 1} ({len(batch)} texts)...")
            
            batch_results = self._process_batch(batch)
            results.extend(batch_results)
            
            # Rate limiting - adjust as needed
            if i + batch_size < len(texts):
                time.sleep(1)
        
        return results
    
    def _process_batch(self, batch: List[str]) -> List[Dict]:
        """Process a single batch of texts in one API call with retry logic."""
        for attempt in range(self.max_retries):
            try:
                # Create numbered list of texts for the batch
                texts_formatted = "\n".join([f"{i+1}. {text}" for i, text in enumerate(batch)])
                
                prompt = f"""Analyze the sentiment of each of the following texts. Respond with a JSON array containing one object per text, in the same order.

Each object should have:
- sentiment: one of "positive", "negative", or "neutral"
- confidence: "high", "medium", or "low"
- explanation: brief reason for the classification

Texts:
{texts_formatted}

Respond only with a valid JSON array: [{{"sentiment": "...", "confidence": "...", "explanation": "..."}}, ...]"""

                payload = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": "You are a sentiment analysis assistant. Always respond with valid JSON arrays."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.3,
                    "max_tokens": 200 * len(batch)  # Scale tokens with batch size
                }
                
                response = requests.post(
                    f"{self.api_url}/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60
                )
                
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"]["content"]
                
                # Try to parse JSON array response
                try:
                    # Clean up common issues with JSON responses
                    content_cleaned = content.strip()
                    # Remove markdown code blocks if present
                    if content_cleaned.startswith("```"):
                        content_cleaned = content_cleaned.split("```")[1]
                        if content_cleaned.startswith("json"):
                            content_cleaned = content_cleaned[4:]
                        content_cleaned = content_cleaned.strip()
                    
                    sentiments = json.loads(content_cleaned)
                    
                    # Validate that it's a list
                    if not isinstance(sentiments, list):
                        raise ValueError("Response is not a JSON array")
                    
                    # Ensure we got the right number of results
                    if len(sentiments) != len(batch):
                        print(f"Warning: Expected {len(batch)} results, got {len(sentiments)}. Retrying...")
                        if attempt < self.max_retries - 1:
                            time.sleep(1)
                            continue
                        # Last attempt: pad with neutral if too few
                        while len(sentiments) < len(batch):
                            sentiments.append({
                                "sentiment": "neutral",
                                "confidence": "low",
                                "explanation": "Incomplete response after retries"
                            })
                    
                    # Combine texts with their sentiment results
                    results = []
                    for text, sentiment in zip(batch, sentiments[:len(batch)]):
                        results.append({
                            "text": text,
                            "sentiment": sentiment.get("sentiment", "neutral"),
                            "confidence": sentiment.get("confidence", "N/A"),
                            "explanation": sentiment.get("explanation", "")
                        })
                    
                    return results
                    
                except (json.JSONDecodeError, ValueError) as e:
                    print(f"JSON parse error on attempt {attempt + 1}/{self.max_retries}: {e}")
                    print(f"Response preview: {content[:200]}...")
                    
                    if attempt < self.max_retries - 1:
                        print(f"Retrying in 2 seconds...")
                        time.sleep(2)
                        continue
                    else:
                        print(f"Failed to parse JSON after {self.max_retries} attempts")
                        # Last resort: return neutral for all texts
                        return [{
                            "text": text,
                            "sentiment": "neutral",
                            "confidence": "low",
                            "explanation": f"JSON parse error after {self.max_retries} retries"
                        } for text in batch]
                    
            except requests.exceptions.RequestException as e:
                print(f"Request error on attempt {attempt + 1}/{self.max_retries}: {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"Request failed after {self.max_retries} attempts")
                    return [{
                        "text": text,
                        "sentiment": "error",
                        "confidence": "N/A",
                        "explanation": f"Request error after {self.max_retries} retries: {str(e)}"
                    } for text in batch]
            
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}/{self.max_retries}: {e}")
                
                if attempt < self.max_retries - 1:
                    print(f"Retrying in 2 seconds...")
                    time.sleep(2)
                    continue
                else:
                    print(f"Processing failed after {self.max_retries} attempts")
                    return [{
                        "text": text,
                        "sentiment": "error",
                        "confidence": "N/A",
                        "explanation": f"Error after {self.max_retries} retries: {str(e)}"
                    } for text in batch]
        
        # Should never reach here, but just in case
        return [{
            "text": text,
            "sentiment": "error",
            "confidence": "N/A",
            "explanation": "Unknown error"
        } for text in batch]