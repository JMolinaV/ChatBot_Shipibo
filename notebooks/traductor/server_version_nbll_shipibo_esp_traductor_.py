import torch
import time
import os
import sys

# Forzar unbuffered output
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# Desactivar wandb
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "offline"
os.environ["WANDB_SILENT"] = "true"

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    TrainerCallback,
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
        print(f"🔄 Cargando modelo: {model_name}", flush=True)

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

        print("✅ Todo Listo!", flush=True)

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
# CARGAMOS DATASET
# =======================================================================

def cargar_dataset(source, tipo='json'):
    """Carga dataset desde diferentes fuentes"""

    if tipo == 'json':
        print(f"📂 Cargando: {source}", flush=True)
        with open(source, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if isinstance(data, list):
            spa = [item['spa'] for item in data]
            shp = [item['shp'] for item in data]
        else:
            spa = data['spa']
            shp = data['shp']

        dataset = Dataset.from_dict({'spa': spa, 'shp': shp})

    elif tipo == 'huggingface':
        print(f"📂 Cargando: {source}", flush=True)
        dataset = load_dataset(source)
        if 'train' in dataset:
            dataset = dataset['train']

    print(f"✅ {len(dataset)} pares cargados", flush=True)
    return dataset

# =======================================================================
# CALLBACK: PROGRESO EN TIEMPO REAL
# =======================================================================

class ProgressCallback(TrainerCallback):
    """
    Callback para mostrar progreso detallado durante entrenamiento
    """
    
    def __init__(self):
        self.step_times = []
        self.last_step_time = None
        
    def on_train_begin(self, args, state, control, **kwargs):
        print("\n" + "🚀 " + "="*68, flush=True)
        print("🚀 ENTRENAMIENTO INICIADO", flush=True)
        print("🚀 " + "="*68, flush=True)
        self.last_step_time = time.time()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Calcular tiempo por step
        current_time = time.time()
        if self.last_step_time is not None:
            step_time = current_time - self.last_step_time
            self.step_times.append(step_time)
        self.last_step_time = current_time
        
        # Mostrar progreso cada 20 pasos
        if state.global_step % 20 == 0:
            avg_time = np.mean(self.step_times[-20:]) if self.step_times else 0
            
            # Calcular ETA
            total_steps = state.max_steps
            remaining_steps = total_steps - state.global_step
            eta_seconds = remaining_steps * avg_time
            eta_minutes = eta_seconds / 60
            
            print(f"\n⚡ Step {state.global_step}/{total_steps} | "
                  f"Epoch {state.epoch:.2f} | "
                  f"Tiempo/step: {avg_time:.2f}s | "
                  f"ETA: {eta_minutes:.1f} min", flush=True)
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        # Mostrar loss cuando esté disponible
        if logs and 'loss' in logs:
            print(f"📉 Loss: {logs['loss']:.4f}", flush=True)

# =======================================================================
# CALLBACK: LOGGING COMPLETO POR ÉPOCA
# =======================================================================

class PrintEpochCallback(TrainerCallback):
    """
    Callback para imprimir información detallada al final de cada época:
    - epoch, loss, eval_loss
    - tiempo por época
    - memoria GPU utilizada (si aplica)
    - traducción de ejemplo
    - batch size informado
    """

    def __init__(self, tokenizer, model, sample_text="Hola, ¿cómo estás?", tgt_code="quy_Latn", batch_size=None):
        self.tokenizer = tokenizer
        self.model = model
        self.sample_text = sample_text
        self.tgt_code = tgt_code
        self.batch_size = batch_size
        self._epoch_start_time = None

    def on_epoch_begin(self, args, state, control, **kwargs):
        # Guardar tiempo de inicio de la época
        self._epoch_start_time = time.time()
        print(f"\n{'='*60}", flush=True)
        print(f"📖 Iniciando Época {state.epoch:.0f}", flush=True)
        print(f"{'='*60}", flush=True)

    def on_epoch_end(self, args, state, control, **kwargs):
        # Tiempo de la época
        elapsed = None
        if self._epoch_start_time is not None:
            elapsed = time.time() - self._epoch_start_time

        # Intentar obtener métricas desde kwargs o state
        metrics = kwargs.get("metrics", {}) or {}
        # Extraer loss desde state.log_history (última aparición de 'loss')
        loss = None
        for entry in reversed(state.log_history):
            if 'loss' in entry:
                loss = entry.get('loss')
                break

        # Eval loss: preferir metrics luego revisar state.log_history
        eval_loss = metrics.get('eval_loss', None)
        if eval_loss is None:
            for entry in reversed(state.log_history):
                if 'eval_loss' in entry:
                    eval_loss = entry.get('eval_loss')
                    break

        # Epoch number (a veces state.epoch es float)
        epoch_num = state.epoch if state.epoch is not None else (state.global_step or 0)

        # GPU memoria (MB)
        gpu_mem_mb = None
        if torch.cuda.is_available():
            try:
                # max_memory_allocated puede no estar disponible en algunas versiones; usar memory_allocated como fallback
                gpu_mem_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
            except Exception:
                try:
                    gpu_mem_mb = torch.cuda.memory_allocated() / (1024 ** 2)
                except Exception:
                    gpu_mem_mb = None

        # Traducción de ejemplo
        ejemplo = self.sample_text
        try:
            # preparar inputs y mover al device del modelo
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
            pred = f"[error al generar ejemplo: {e}]"

        # Imprimir información completa
        print("\n" + "="*60, flush=True)
        print(f"📌 Época completada: {epoch_num}", flush=True)
        if loss is not None:
            print(f"🔹 loss: {loss:.6f}", flush=True)
        else:
            print("🔹 loss: N/A", flush=True)
        if eval_loss is not None:
            print(f"🔹 eval_loss: {eval_loss:.6f}", flush=True)
        else:
            print("🔹 eval_loss: N/A", flush=True)
        if elapsed is not None:
            print(f"⏱ Tiempo en la época: {elapsed:.2f} segundos ({elapsed/60:.1f} min)", flush=True)
        if self.batch_size is not None:
            print(f"📦 Batch size por dispositivo: {self.batch_size}", flush=True)
        if gpu_mem_mb is not None:
            print(f"🧠 GPU memoria (max alloc): {gpu_mem_mb:.1f} MB", flush=True)
        print(f"📝 Ejemplo (ES): {ejemplo}", flush=True)
        print(f"🈶 Predicción (SH): {pred}", flush=True)
        print("="*60 + "\n", flush=True)

# =======================================================================
# Entrenamiento del modelo
# =======================================================================

def entrenar_modelo(dataset, output_dir='./modelo-shipibo-entrenado', num_epochs=10):

    print("\n" + "="*70, flush=True)
    print("🎓 ENTRENANDO MODELO SHIPIBO-KONIBO", flush=True)
    print("="*70 + "\n", flush=True)

    # Dataset en train y test
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = split['train']
    test_data = split['test']

    print(f"✅ Train: {len(train_data)} | Test: {len(test_data)}\n", flush=True)

    # Modelo base
    print("🔄 Cargando NLLB base...", flush=True)
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="spa_Latn", tgt_lang="quy_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Mover modelo a GPU si está disponible
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print(f"✅ Modelo base cargado en {device.upper()}\n", flush=True)

    # FUNCIÓN DE PREPROCESAMIENTO
    def preprocess_function(examples):
        """Preprocesa sin usar as_target_tokenizer (deprecated)"""

        inputs = examples['spa']
        targets = examples['shp']

        # Tokenizar inputs con idioma fuente
        tokenizer.src_lang = "spa_Latn"
        model_inputs = tokenizer(
            inputs,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_tensors=None  # Importante para batched processing
        )

        # Tokenizar targets manualmente con idioma destino
        tokenizer.tgt_lang = "quy_Latn"

        # Método correcto sin deprecation warning
        labels_list = []
        for target in targets:
            # Tokenizar cada target individualmente
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
        num_train_epochs=num_epochs,
        weight_decay=0.01,
        save_total_limit=2,
        predict_with_generate=True,
        fp16=torch.cuda.is_available(),
        logging_steps=10,  # ⬅️ Cambiado de 100 a 10
        load_best_model_at_end=True,
        report_to="none",
        disable_tqdm=False,  # ⬅️ Habilitar tqdm
        logging_first_step=True,  # ⬅️ Log desde el primer step
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Crear instancias de los callbacks
    epoch_callback = PrintEpochCallback(
        tokenizer=tokenizer,
        model=model,
        sample_text="Hola, ¿cómo estás?",
        tgt_code="quy_Latn",
        batch_size=training_args.per_device_train_batch_size
    )
    
    progress_callback = ProgressCallback()

    # Crear trainer (añadiendo ambos callbacks)
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=[epoch_callback, progress_callback],  # ⬅️ Ambos callbacks
    )

    print("✅ Trainer configurado\n", flush=True)

    # Entrenar
    print("🚀 Iniciando entrenamiento...", flush=True)
    print(f"⏱️ Esto tomará aproximadamente {num_epochs * 2}-{num_epochs * 4} minutos\n", flush=True)

    try:
        trainer.train()

        # Guardar modelo
        print("\n💾 Guardando modelo...", flush=True)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n✅ ¡ÉXITO! Modelo guardado en: {output_dir}", flush=True)
        print(f"💡 Úsalo con: TraductorShipibo(model_name='{output_dir}')", flush=True)

        return trainer

    except Exception as e:
        print(f"\n❌ Error durante entrenamiento: {e}", flush=True)
        raise

# =======================================================================
# MODELO ENTRENADO
# =======================================================================

def usar_modelo_entrenado(model_path='./modelo-shipibo-entrenado'):
    """Carga y prueba el modelo entrenado"""

    if not os.path.exists(model_path):
        print(f"❌ No existe: {model_path}", flush=True)
        print("💡 Primero entrena con: entrenar_modelo(dataset)", flush=True)
        return None

    print(f"\n🔄 Cargando modelo entrenado...", flush=True)
    traductor = TraductorShipibo(model_name=model_path)

    print("\n" + "="*70, flush=True)
    print("🧪 PROBANDO MODELO ENTRENADO", flush=True)
    print("="*70 + "\n", flush=True)

    # Pruebas
    frases_test = [
        "Hola",
        "Buenos días",
        "¿Cómo estás?",
        "Gracias",
        "Me gusta el río",
    ]

    print("📝 Español → Shipibo:\n", flush=True)
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
    Evalúa el modelo con BLEU y guarda los resultados en CSV y JSON.
    """
    print("\n" + "="*70, flush=True)
    print("📊 EVALUANDO MODELO CON BLEU", flush=True)
    print("="*70 + "\n", flush=True)

    # 1. Cargar modelo entrenado
    traductor = TraductorShipibo(model_name=modelo_path)

    # 2. Cargar métrica BLEU
    bleu = load("sacrebleu")

    # 3. Limitar ejemplos si se solicita
    if num_ejemplos:
        test_data = test_data.select(range(min(num_ejemplos, len(test_data))))

    traducciones = []
    referencias = []

    filas_exportar = []

    print(f"🔄 Evaluando {len(test_data)} ejemplos...", flush=True)

    # 4. Evaluación
    for i, ejemplo in enumerate(test_data):

        # Traducción generada
        traduccion = traductor.translate(
            ejemplo['spa'],
            src_lang='español',
            tgt_lang='shipibo'
        ).strip()

        # Referencia real
        referencia = ejemplo['shp'].strip()

        traducciones.append(traduccion)
        referencias.append([referencia])

        # Agregar a tabla para exportar
        filas_exportar.append({
            "index": i,
            "spa": ejemplo["spa"],
            "shp_reference": referencia,
            "shp_predicted": traduccion
        })

        # Mostrar primeros ejemplos
        if i < 5:
            print(f"\n📝 Ejemplo {i+1}:", flush=True)
            print(f"   Español:    {ejemplo['spa']}", flush=True)
            print(f"   Shipibo GT: {referencia}", flush=True)
            print(f"   Generado:   {traduccion}", flush=True)
            
        # Mostrar progreso cada 50 ejemplos
        if (i + 1) % 50 == 0:
            print(f"⚡ Progreso: {i+1}/{len(test_data)}", flush=True)

    # 5. Calcular BLEU
    resultado = bleu.compute(predictions=traducciones, references=referencias)
    bleu_score = resultado['score']

    print("\n" + "="*70, flush=True)
    print(f"🎯 BLEU Score: {bleu_score:.2f}", flush=True)
    print("="*70, flush=True)

    # -------- GUARDAR EN CSV --------
    if save_csv is not None:
        with open(save_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["index", "spa", "shp_reference", "shp_predicted"])
            writer.writeheader()
            writer.writerows(filas_exportar)
        print(f"📁 Resultados guardados en CSV: {save_csv}", flush=True)

    # -------- GUARDAR EN JSON --------
    if save_json is not None:
        data_json = {
            "bleu_score": bleu_score,
            "num_ejemplos": len(test_data),
            "resultados": filas_exportar
        }
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)
        print(f"📁 Resultados guardados en JSON: {save_json}", flush=True)

    return bleu_score, traducciones, referencias

# =======================================================================
# EJEMPLO COMPLETO DE USO
# =======================================================================

if __name__ == "__main__":

    print("="*70, flush=True)
    print("🌎 TRADUCTOR ESPAÑOL-SHIPIBO-KONIBO", flush=True)
    print("="*70 + "\n", flush=True)

    # PASO 1: Probar traductor base (sin entrenar)
    print("PASO 1: Probando traductor base (inmediato)\n", flush=True)
    traductor_base = TraductorShipibo()

    print("\n📝 Ejemplos con modelo base:\n", flush=True)
    ejemplos = ["Hola", "Buenos días", "Gracias"]
    for ej in ejemplos:
        print(f"ES: {ej}", flush=True)
        print(f"SH: {traductor_base.translate(ej)}\n", flush=True)

    print("\n" + "="*70, flush=True)

    # ==================================================================
    # INICIAMOS PROCESAMIENTO
    # ==================================================================

    dataset = cargar_dataset('train_merged.json', 'json')
    trainer = entrenar_modelo(dataset, num_epochs=10)
    traductor = usar_modelo_entrenado('./modelo-shipibo-entrenado')
    print(traductor.translate('Quiero ir a Lima', 'español', 'shipibo'), flush=True)

    # Evaluación BLEU
    split = dataset.train_test_split(test_size=0.2)
    test_data = split["test"]

    evaluar_bleu(
        "modelo-shipibo-entrenado",
        test_data,
        num_ejemplos=200,   # opcional
        save_csv="evaluacion_bleu.csv",
        save_json="evaluacion_bleu.json"
    )