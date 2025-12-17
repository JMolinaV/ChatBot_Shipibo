# -*- coding: utf-8 -*-
"""
INFERENCIA ESPAÑOL -> SHIPIBO (es2shp)
Entrada: test.jsonl  (campos: src, tgt)
Salida: CSV con inferencia
"""

import json
import csv
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------------------------
# ARGUMENTOS
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--input_jsonl", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Usando dispositivo: {device}")

    # -------------------------------------------------
    # CARGA DEL MODELO
    # -------------------------------------------------
    print("▶ Cargando modelo:", args.model_folder)
    tokenizer = AutoTokenizer.from_pretrained(args.model_folder)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_folder)
    model.to(device)
    model.eval()

    # -------------------------------------------------
    # LEER JSONL
    # -------------------------------------------------
    data = []
    with open(args.input_jsonl, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                data.append(ex)

    print(f"▶ {len(data)} ejemplos cargados desde JSONL")

    # -------------------------------------------------
    # INFERENCIA
    # -------------------------------------------------
    rows = []

    for i, ex in enumerate(data):
        src = ex["src"].strip()
        tgt = ex.get("tgt", "").strip()

        inputs = tokenizer(
            src,
            return_tensors="pt",
            truncation=True,
            max_length=256
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=args.max_length,
                num_beams=4,                  # evita colapso
                no_repeat_ngram_size=3,
                early_stopping=True
            )

        pred = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        rows.append({
            "source": src,
            "target": tgt,
            "inference": pred
        })

        if i < 5:
            print(f"\n🔹 EJEMPLO {i+1}")
            print("ES :", src)
            print("GT :", tgt)
            print("SH :", pred)

    # -------------------------------------------------
    # GUARDAR CSV
    # -------------------------------------------------
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["source", "target", "inference"]
        )
        writer.writeheader()
        writer.writerows(rows)

    print("\n✅ INFERENCIA COMPLETADA")
    print("📄 Archivo generado:", out_path)


if __name__ == "__main__":
    main()
