import torch
import time
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
    EarlyStoppingCallback,  # ⬅️ NUEVO
)
from datasets import Dataset, load_dataset
import json
import numpy as np
import csv
from evaluate import load

# =======================================================================
# TRADUCTOR INMEDIATO
# =======================================================================

class TraductorShipibo:

    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        print(f"📦 Cargando modelo: {model_name}", flush=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="spa_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print("✅ Modelo en GPU", flush=True)
        else:
            print("⚠️ Modelo en CPU", flush=True)

        self.model.eval()

        self.lang_codes = {
            'español': 'spa_Latn',
            'shipibo': 'quy_Latn',
            'inglés': 'eng_Latn',
            'quechua': 'quy_Latn',
        }

        print("✅ Todo listo!", flush=True)

    def translate(self, text, src_lang='español', tgt_lang='shipibo'):
        src_code = self.lang_codes[src_lang]
        tgt_code = self.lang_codes[tgt_lang]

        self.tokenizer.src_lang = src_code
        inputs = self.tokenizer(text, return_tensors="pt", max_length=128, truncation=True)

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        translated = self.model.generate(
            **inputs,
            forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(tgt_code),
            max_length=128,
            num_beams=5,
        )

        return self.tokenizer.decode(translated[0], skip_special_tokens=True)

# =======================================================================
# CARGAR DATASETS JSONL (train, validation, test)
# =======================================================================

def cargar_datasets_jsonl(train_path='train.jsonl', 
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
                spa_texts.append(item['src'])
                shp_texts.append(item['tgt'])
        
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

# =======================================================================
# CALLBACK: PROGRESO EN TIEMPO REAL
# =======================================================================

class ProgressCallback(TrainerCallback):
    """Callback para mostrar progreso cada 20 steps"""
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 20 == 0:
            print(f"⚡ Step {state.global_step}/{state.max_steps} | Epoch {state.epoch:.2f}", flush=True)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and 'loss' in logs:
            print(f"📉 Loss: {logs['loss']:.4f}", flush=True)

# =======================================================================
# CALLBACK: LOGGING COMPLETO POR ÉPOCA
# =======================================================================

class PrintEpochCallback(TrainerCallback):
    """
    Callback para imprimir información detallada al final de cada época
    """

    def __init__(self, tokenizer, model, sample_text="Hola, ¿cómo estás?", tgt_code="quy_Latn", batch_size=None):
        self.tokenizer = tokenizer
        self.model = model
        self.sample_text = sample_text
        self.tgt_code = tgt_code
        self.batch_size = batch_size
        self._epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        self._epoch_start_time = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"📖 Iniciando Época {int(state.epoch)}", flush=True)
        print(f"{'='*60}\n", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        elapsed = None
        if self._epoch_start_time is not None:
            elapsed = time.time() - self._epoch_start_time

        metrics = kwargs.get("metrics", {}) or {}
        loss = None
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                loss = entry.get('loss')
                break

        eval_loss = metrics.get('eval_loss', None)
        if eval_loss is None:
            for entry in reversed(state.log_history):
                if 'eval_loss' in entry:
                    eval_loss = entry.get('eval_loss')
                    break

        epoch_num = state.epoch if state.epoch is not None else (state.global_step or 0)

        gpu_mem_mb = None
        if torch.cuda.is_available():
            try:
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                try:
                    gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                except Exception:
                    gpu_mem_mb = None

        ejemplo = self.sample_text
        try:
            inputs = self.tokenizer(ejemplo, return_tensors="pt", truncation=True, max_length=128)
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            out = self.model.generate(
                **inputs,
                forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(self.tgt_code),
                max_length=128,
                num_beams=5,
            )
            pred = self.tokenizer.decode(out[0], skip_special_tokens=True)
        except Exception as e:
            pred = f"[error: {e}]"

        print("\n" + "="*60, flush=True)
        print(f"📌 Época completada: {int(epoch_num)}", flush=True)
        if loss is not None:
            print(f"🔹 loss: {loss:.6f}", flush=True)
        else:
            print("🔹 loss: N/A", flush=True)
        if eval_loss is not None:
            print(f"🔹 eval_loss: {eval_loss:.6f}", flush=True)
        else:
            print("🔹 eval_loss: N/A", flush=True)
        if elapsed is not None:
            print(f"⏱ Tiempo: {elapsed:.2f}s ({elapsed/60:.1f} min)", flush=True)
        if self.batch_size is not None:
            print(f"📦 Batch size: {self.batch_size}", flush=True)
        if gpu_mem_mb is not None:
            print(f"🧠 GPU memoria: {gpu_mem_mb:.1f} MB", flush=True)
        print(f"📝 Ejemplo ES: {ejemplo}", flush=True)
        print(f"🈶 Predicción SH: {pred}", flush=True)
        print("="*60 + "\n", flush=True)

# =======================================================================
# Entrenamiento del modelo
# =======================================================================

def entrenar_modelo(datasets_dict, 
                   output_dir='./modelo-shipibo-entrenado', 
                   num_epochs=50,  # ⬅️ CAMBIADO de 10 a 50
                   early_stopping_patience=5):  # ⬅️ NUEVO parámetro
    """
    Entrena el modelo usando train/validation/test ya separados
    
    Args:
        datasets_dict: dict con keys 'train', 'validation', 'test'
        num_epochs: número máximo de épocas (default 50)
        early_stopping_patience: cuántas épocas esperar sin mejora antes de detener (default 5)
    """

    print("\n" + "="*70, flush=True)
    print("🎓 ENTRENANDO MODELO SHIPIBO-KONIBO", flush=True)
    print(f"📊 Épocas máximas: {num_epochs}", flush=True)
    print(f"⏹️  Early Stopping: {early_stopping_patience} épocas sin mejora", flush=True)
    print("="*70 + "\n", flush=True)

    train_data = datasets_dict['train']
    val_data = datasets_dict['validation']
    test_data = datasets_dict['test']

    print(f"📊 Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}\n", flush=True)

    # Modelo base
    print("📦 Cargando NLLB base...", flush=True)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="spa_Latn", tgt_lang="quy_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"✅ Modelo cargado en {device.upper()}\n", flush=True)

    # FUNCIÓN DE PREPROCESAMIENTO
    def preprocess_function(examples):
        inputs = examples['spa']
        targets = examples['shp']

        tokenizer.src_lang = "spa_Latn"
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors=None
        )

        tokenizer.tgt_lang = "quy_Latn"

        labels_list = []
        for target in targets:
            tokenized = tokenizer(
                target,
                max_length=128,
                truncation=True,
                padding='max_length',
            )
            labels_list.append(tokenized['input_ids'])

        model_inputs['labels'] = labels_list

        return model_inputs

    # Preprocesar datos
    print("🔄 Preprocesando datos...", flush=True)
    train_tokenized = train_data.map(
        preprocess_function,
        batched=True,
        remove_columns=train_data.column_names,
        desc="Procesando train"
    )

    val_tokenized = val_data.map(
        preprocess_function,
        batched=True,
        remove_columns=val_data.column_names,
        desc="Procesando validation"
    )

    test_tokenized = test_data.map(
        preprocess_function,
        batched=True,
        remove_columns=test_data.column_names,
        desc="Procesando test"
    )

    print("✅ Datos preprocesados\n", flush=True)

    # Configurar entrenamiento
    print("⚙️ Configurando entrenamiento...", flush=True)
    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        num_train_epochs=num_epochs,  # ⬅️ Ahora usa el parámetro
        weight_decay=0.01,
        save_total_limit=5,  # ⬅️ CAMBIADO: Guardar 3 últimos checkpoints
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,
        load_best_model_at_end=True,  # ⬅️ CRÍTICO para early stopping
        metric_for_best_model="eval_loss",  # ⬅️ Métrica para evaluar
        greater_is_better=False,  # ⬅️ NUEVO: eval_loss menor es mejor
        report_to="none",
        disable_tqdm=True,
        logging_first_step=True,
        dataloader_num_workers=0,
    )

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    epoch_callback = PrintEpochCallback(
        tokenizer=tokenizer,
        model=model,
        sample_text="Hola, ¿cómo estás?",
        tgt_code="quy_Latn",
        batch_size=training_args.per_device_train_batch_size
    )
    
    progress_callback = ProgressCallback()
    
    # ⬅️ NUEVO: Early Stopping Callback
    early_stopping = EarlyStoppingCallback(
        early_stopping_patience=early_stopping_patience,
        early_stopping_threshold=0.0  # Cualquier mejora cuenta
    )

    # ✅ Agregar early_stopping a los callbacks
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        callbacks=[epoch_callback, progress_callback, early_stopping],  # ⬅️ AGREGADO
    )

    print("✅ Trainer configurado\n", flush=True)

    print("🚀 Iniciando entrenamiento...", flush=True)
    print(f"⏱ Estimación máxima: {num_epochs * 2}-{num_epochs * 4} minutos", flush=True)
    print(f"🛑 Se detendrá automáticamente si no mejora en {early_stopping_patience} épocas\n", flush=True)
    print("="*70, flush=True)
    print("⏳ ESPERANDO PRIMER STEP (puede tardar 5-10 min)...", flush=True)
    print("="*70 + "\n", flush=True)

    try:
        train_result = trainer.train()

        # Información sobre early stopping
        if hasattr(trainer.state, 'best_metric'):
            print(f"\n🏆 Mejor eval_loss: {trainer.state.best_metric:.6f}", flush=True)
            print(f"📍 Alcanzado en época: {train_result.metrics.get('epoch', 'N/A')}", flush=True)

        print("\n💾 Guardando modelo...", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n🎉 ¡ÉXITO! Modelo guardado en: {output_dir}", flush=True)
        print(f"📌 Úsalo con: TraductorShipibo(model_name='{output_dir}')", flush=True)

        # Guardar métricas de entrenamiento
        metrics_file = os.path.join(output_dir, "training_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=4)
        print(f"📊 Métricas guardadas en: {metrics_file}", flush=True)

        return trainer, test_data

    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}", flush=True)
        raise

# =======================================================================
# MODELO ENTRENADO
# =======================================================================

def usar_modelo_entrenado(model_path='./modelo-shipibo-entrenado'):

    if not os.path.exists(model_path):
        print(f"❌ No existe: {model_path}", flush=True)
        print("⚠️ Primero entrena con: entrenar_modelo(datasets)", flush=True)
        return None

    print(f"\n📦 Cargando modelo entrenado...", flush=True)
    traductor = TraductorShipibo(model_name=model_path)

    print("\n" + "="*70, flush=True)
    print("🧪 PROBANDO MODELO ENTRENADO", flush=True)
    print("="*70 + "\n", flush=True)

    frases_test = [
        "Hola",
        "Buenos días",
        "¿Cómo estás?",
        "Gracias",
        "Me gusta el río",
    ]

    print("🗣️ Español → Shipibo:\n", flush=True)
    for frase in frases_test:
        traduccion = traductor.translate(frase, 'español', 'shipibo')
        print(f"ES: {frase}", flush=True)
        print(f"SH: {traduccion}\n", flush=True)

    return traductor

# =======================================================================
# EVALUACIÓN BLEU (export CSV/JSON)
# =======================================================================

def evaluar_bleu(modelo_path, test_data, num_ejemplos=None,
                 save_csv="resultados_bleu.csv",
                 save_json="resultados_bleu.json"):
    """
    Evalúa el modelo con BLEU en el conjunto de TEST (no usado en entrenamiento)
    """
    print("\n" + "="*70, flush=True)
    print("📊 EVALUACIÓN FINAL CON BLEU (TEST SET)", flush=True)
    print("="*70 + "\n", flush=True)

    traductor = TraductorShipibo(model_name=modelo_path)
    bleu = load("sacrebleu")

    if num_ejemplos:
        test_data = test_data.select(range(min(num_ejemplos, len(test_data))))

    traducciones = []
    referencias = []
    filas_exportar = []

    print(f"🔍 Evaluando {len(test_data)} ejemplos de TEST...\n", flush=True)

    for i, ejemplo in enumerate(test_data):

        traduccion = traductor.translate(
            ejemplo['spa'],
            src_lang='español',
            tgt_lang='shipibo'
        ).strip()

        referencia = ejemplo['shp'].strip()

        traducciones.append(traduccion)
        referencias.append([referencia])

        filas_exportar.append({
            "index": i,
            "spa": ejemplo["spa"],
            "shp_reference": referencia,
            "shp_predicted": traduccion
        })

        if i < 5:
            print(f"📝 Ejemplo {i+1}:", flush=True)
            print(f"   Español:    {ejemplo['spa']}", flush=True)
            print(f"   Shipibo GT: {referencia}", flush=True)
            print(f"   Generado:   {traduccion}\n", flush=True)
            
        if (i + 1) % 50 == 0:
            print(f"⚡ Progreso: {i+1}/{len(test_data)}", flush=True)

    resultado = bleu.compute(predictions=traducciones, references=referencias)
    bleu_score = resultado['score']

    print("\n" + "="*70, flush=True)
    print(f"🎯 BLEU Score (TEST): {bleu_score:.2f}", flush=True)
    print("="*70, flush=True)

    if save_csv is not None:
        with open(save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "spa", "shp_reference", "shp_predicted"])
            writer.writeheader()
            writer.writerows(filas_exportar)
        print(f"📁 CSV guardado: {save_csv}", flush=True)

    if save_json is not None:
        data_json = {
            "bleu_score": bleu_score,
            "num_ejemplos": len(test_data),
            "resultados": filas_exportar
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)
        print(f"📁 JSON guardado: {save_json}", flush=True)

    return bleu_score, traducciones, referencias

# =======================================================================
# EJEMPLO COMPLETO DE USO
# =======================================================================

if __name__ == "__main__":

    print("="*70, flush=True)
    print("🌍 TRADUCTOR ESPAÑOL-SHIPIBO-KONIBO", flush=True)
    print("="*70 + "\n", flush=True)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    print("PASO1:")

    # PASO 1: Cargar datasets desde JSONL
    datasets = cargar_datasets_jsonl(
        train_path='train.jsonl',
        val_path='validation.jsonl',
        test_path='test.jsonl'
    )

    print("PASO2:")

    # PASO 2: Entrenar modelo con early stopping
    trainer, test_data = entrenar_modelo(
        datasets, 
        output_dir='./modelo-shipibo-entrenado',
        num_epochs=50,  # ⬅️ Hasta 50 épocas
        early_stopping_patience=5  # ⬅️ Se detiene si no mejora en 5 épocas
    )

    print("PASO3:")

    # PASO 3: Probar modelo entrenado
    traductor = usar_modelo_entrenado('./modelo-shipibo-entrenado')
    
    if traductor:
        print("\n🧪 Prueba rápida:", flush=True)
        print(traductor.translate('Quiero ir a Lima', 'español', 'shipibo'), flush=True)

    print("PASO4:")
    # PASO 4: Evaluación BLEU en TEST
    evaluar_bleu(
        "./modelo-shipibo-entrenado",
        test_data,
        num_ejemplos=None,
        save_csv="evaluacion_bleu_test.csv",
        save_json="evaluacion_bleu_test.json"
    )