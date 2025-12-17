# -*- coding: utf-8 -*-
"""
INFERENCIA ESPAÑOL -> SHIPIBO (es2shp)
Evita colapso de decoding (kikin kikin kikin)
"""

import os
import csv
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------------------------
# ARGUMENTOS
# -------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--direction", type=str, default="es2shp")
    parser.add_argument("--max_length", type=int, default=128)
    return parser.parse_args()


# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    args = parse_args()

    assert args.direction == "es2shp", "Este script es solo para es2shp"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"▶ Usando dispositivo: {device}")

    # -------------------------------------------------
    # CARGA CORRECTA DEL MODELO Y TOKENIZER
    # -------------------------------------------------
    print("▶ Cargando modelo desde:", args.model_folder)

    tokenizer = AutoTokenizer.from_pretrained(args.model_folder)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_folder)
    model.to(device)
    model.eval()

    print("▶ Modelo cargado correctamente")
    print("▶ vocab_size:", model.config.vocab_size)

    # -------------------------------------------------
    # LECTURA CSV
    # -------------------------------------------------
    rows = []
    with open(args.input_csv, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"▶ {len(rows)} ejemplos cargados")

    # -------------------------------------------------
    # INFERENCIA
    # -------------------------------------------------
    output_rows = []

    for i, row in enumerate(rows):
        src = row["source"].strip()

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
                num_beams=4,                  # 👈 CLAVE
                early_stopping=True,
                no_repeat_ngram_size=3,        # 👈 CLAVE
            )

        pred = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        out_row = dict(row)
        out_row["inference"] = pred
        output_rows.append(out_row)

        if i < 5:
            print(f"\n🔹 EJEMPLO {i+1}")
            print("ES :", src)
            print("SH :", pred)

    # -------------------------------------------------
    # GUARDAR CSV
    # -------------------------------------------------
    fieldnames = list(output_rows[0].keys())

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)

    with open(args.output_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    print("\n✅ INFERENCIA COMPLETADA")
    print("📄 Archivo generado:", args.output_csv)


if __name__ == "__main__":
    main()
