

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset, load_dataset
import json
import os
from datasets import load_metric
import numpy as np

# ==============================================================================
# TRADUCTOR INMEDIATO
# ==============================================================================

class TraductorShipibo:

    def __init__(self, model_name="facebook/nllb-200-distilled-600M"):
        print(f" Cargando modelo: {model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="spa_Latn")
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

        if torch.cuda.is_available():
            self.model = self.model.cuda()
            print(" Modelo en GPU")
        else:
            print(" Modelo en CPU")

        self.model.eval()

        self.lang_codes = {
            'español': 'spa_Latn',
            'shipibo': 'quy_Latn',
            'inglés': 'eng_Latn',
            'quechua': 'quy_Latn',
        }

        print(" todo Listo!")

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

# ==============================================================================
# CARGAMOS DATASET
# ==============================================================================

def cargar_dataset(source, tipo='json'):
    """Carga dataset desde diferentes fuentes"""

    if tipo == 'json':
        print(f" Cargando: {source}")
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
        print(f" Cargando: {source}")
        dataset = load_dataset(source)
        if 'train' in dataset:
            dataset = dataset['train']

    print(f" {len(dataset)} pares cargados")
    return dataset

# ==============================================================================
# Entrenamiento del modelo
# ==============================================================================

def entrenar_modelo(dataset, output_dir='./modelo-shipibo-entrenado', num_epochs=10):

    print("\n" + "="*70)
    print("🎓 ENTRENANDO MODELO SHIPIBO-KONIBO")
    print("="*70 + "\n")

    #  dataset en train y test
    split = dataset.train_test_split(test_size=0.2, seed=42)
    train_data = split['train']
    test_data = split['test']

    print(f" Train: {len(train_data)} | Test: {len(test_data)}\n")

    # modelo base
    print(" Cargando NLLB base...")
    model_name = "facebook/nllb-200-distilled-600M"
    tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang="spa_Latn", tgt_lang="quy_Latn")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    print(" Modelo base cargado\n")

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
    print(" Preprocesando datos...")
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

    print(" Datos preprocesados\n")

    # Configurar entrenamiento
    print("  Configurando entrenamiento...")
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
        logging_steps=100,
        load_best_model_at_end=True,
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Crear trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=test_tokenized,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print(" Trainer configurado\n")

    # Entrenar
    print(" Iniciando entrenamiento...")
    print(f" Esto tomará aproximadamente {num_epochs * 2}-{num_epochs * 4} minutos\n")

    try:
        trainer.train()

        # Guardar modelo
        print("\n Guardando modelo...")
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)

        print(f"\n ¡ÉXITO! Modelo guardado en: {output_dir}")
        print(f" Úsalo con: TraductorShipibo(model_name='{output_dir}')")

        return trainer

    except Exception as e:
        print(f"\n Error durante entrenamiento: {e}")
        raise

# ==============================================================================
# MODELO ENTRENADO
# ==============================================================================

def usar_modelo_entrenado(model_path='./modelo-shipibo-entrenado'):
    """Carga y prueba el modelo entrenado"""

    if not os.path.exists(model_path):
        print(f" No existe: {model_path}")
        print(" Primero entrena con: entrenar_modelo(dataset)")
        return None

    print(f"\n Cargando modelo entrenado...")
    traductor = TraductorShipibo(model_name=model_path)

    print("\n" + "="*70)
    print(" PROBANDO MODELO ENTRENADO")
    print("="*70 + "\n")

    # Pruebas
    frases_test = [
        "Hola",
        "Buenos días",
        "¿Cómo estás?",
        "Gracias",
        "Me gusta el río",
    ]

    print(" Español → Shipibo:\n")
    for frase in frases_test:
        traduccion = traductor.translate(frase, 'español', 'shipibo')
        print(f"ES: {frase}")
        print(f"SH: {traduccion}\n")

    return traductor

def evaluar_bleu(modelo_path, test_data, num_ejemplos=None):
    """
    Evalúa el modelo con BLEU score
    """
    print("\n" + "="*70)
    print("📊 EVALUANDO MODELO CON BLEU")
    print("="*70 + "\n")

    # Cargar modelo entrenado
    traductor = TraductorShipibo(model_name=modelo_path)

    # Cargar métrica BLEU
    bleu = load_metric("sacrebleu")

    # Limitar ejemplos si es necesario
    if num_ejemplos:
        test_data = test_data.select(range(min(num_ejemplos, len(test_data))))

    traducciones = []
    referencias = []

    print(f"Evaluando {len(test_data)} ejemplos...")

    for i, ejemplo in enumerate(test_data):
        # Traducir
        traduccion = traductor.translate(
            ejemplo['spa'],
            src_lang='español',
            tgt_lang='shipibo'
        )
        referencia = ejemplo['shp']

        traducciones.append(traduccion)
        referencias.append([referencia])  # Lista de listas para BLEU

        # Mostrar algunos ejemplos
        if i < 5:
            print(f"\n📝 Ejemplo {i+1}:")
            print(f"   Español:    {ejemplo['spa']}")
            print(f"   Shipibo:    {referencia}")
            print(f"   Generado:   {traduccion}")

    # Calcular BLEU
    resultado = bleu.compute(predictions=traducciones, references=referencias)
    bleu_score = resultado['score']

    print("\n" + "="*70)
    print(f"🎯 BLEU Score: {bleu_score:.2f}")
    print("="*70)

    # Interpretación
    if bleu_score >= 50:
        print("✅ Excelente traducción")
    elif bleu_score >= 30:
        print("✅ Buena traducción")
    elif bleu_score >= 15:
        print("⚠️  Traducción aceptable, hay margen de mejora")
    else:
        print("❌ Traducción pobre, necesita más datos/épocas")

    return bleu_score, traducciones, referencias


# ==============================================================================
# EJEMPLO COMPLETO DE USO
# ==============================================================================

if __name__ == "__main__":

    print("="*70)
    print(" TRADUCTOR ESPAÑOL-SHIPIBO-KONIBO")
    print("="*70 + "\n")

    # PASO 1: Probar traductor base (sin entrenar)
    print("PASO 1: Probando traductor base (inmediato)\n")
    traductor_base = TraductorShipibo()

    print("\n Ejemplos con modelo base:\n")
    ejemplos = ["Hola", "Buenos días", "Gracias"]
    for ej in ejemplos:
        print(f"ES: {ej}")
        print(f"SH: {traductor_base.translate(ej)}\n")

    print("\n" + "="*70)

    # ==============================================================================
    #INICIAMOOOS PROCESAMIENTOO
    # ==============================================================================

    dataset = cargar_dataset('train_merged.json', 'json')
    entrenar_modelo(dataset, num_epochs=10)
    traductor = usar_modelo_entrenado('./modelo-shipibo-entrenado')
    print(traductor.translate('Quiero ir a Lima', 'español', 'shipibo'))

# Comprimir la carpeta del modelo
#!zip -r modelo-shipibo.zip modelo-shipibo-entrenado

# Descargar el archivo ZIP
#from google.colab import files
#files.download('modelo-shipibo.zip')


    # Usar:
    split = dataset.train_test_split(test_size=0.2, seed=42)
    test_data = split['test']
    bleu_score, preds, refs = evaluar_bleu('modelo-shipibo-entrenado', test_data)