#!/usr/bin/env python3
import argparse
import pandas as pd
import random
from pathlib import Path


def token_len(text: str) -> int:
    return len(text.strip().split())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_full_csv", type=str, default="sample_200_full.csv")
    parser.add_argument("--output_eval_csv", type=str, default="sample_200_eval.csv")

    parser.add_argument("--bleu_alt_max", type=float, default=80.0)
    parser.add_argument("--bleu_min", type=float, default=30.0)
    parser.add_argument("--bleu_mid_min", type=float, default=40.0)
    parser.add_argument("--bleu_mid_max", type=float, default=85.0)
    parser.add_argument("--bleu_high_min", type=float, default=90.0)
    parser.add_argument("--bleu_high_max", type=float, default=100.0)

    parser.add_argument("--min_src_tokens", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    df = pd.read_csv(args.input_csv)

    # -----------------------------
    # Filtros globales
    # -----------------------------
    df = df[df["bleu_alt"] < args.bleu_alt_max].copy()
    df["src_len"] = df["source"].apply(token_len)
    df = df[df["src_len"] >= args.min_src_tokens]
    df = df[df["bleu"] >= args.bleu_min]

    # Pools por rango BLEU
    df_high = df[
        (df["bleu"] >= args.bleu_high_min) &
        (df["bleu"] <= args.bleu_high_max)
    ].copy()

    df_mid = df[
        (df["bleu"] >= args.bleu_mid_min) &
        (df["bleu"] <= args.bleu_mid_max)
    ].copy()

    if len(df_high) < 20:
        raise ValueError("No hay suficientes ejemplos BLEU 90–100")

    if len(df_mid) < 140:
        raise ValueError("No hay suficientes ejemplos BLEU intermedios")

    # -----------------------------
    # Selección estratificada
    # -----------------------------
    num_groups = 10  # 200 / 10
    selected_rows = []

    df_high = df_high.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_mid = df_mid.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    high_ptr = 0
    mid_ptr = 0

    for g in range(num_groups):
        group_rows = []

        # 1 caso BLEU alto obligatorio
        group_rows.append(df_high.iloc[high_ptr])
        high_ptr += 1

        # 9 casos BLEU intermedio
        for _ in range(9):
            group_rows.append(df_mid.iloc[mid_ptr])
            mid_ptr += 1

        selected_rows.extend(group_rows)

    df_sel = pd.DataFrame(selected_rows).reset_index(drop=True)

    # -----------------------------
    # Construcción de grupos y CSVs
    # -----------------------------
    full_rows = []
    eval_rows = []

    for g in range(num_groups):
        group_df = df_sel.iloc[g * 10:(g + 1) * 10].copy()
        group_id = (g + 1) * 10

        group_df["grupo"] = group_id
        group_df["id_grupo"] = range(1, 11)

        # Elegir 2 posiciones para target
        target_positions = set(random.sample(range(10), 2))

        for _, row in group_df.iterrows():
            pos = row["id_grupo"] - 1

            if pos in target_positions:
                shown_text = row["target"]
                shown_type = "target"
            else:
                shown_text = row["inference"]
                shown_type = "inference"

            full_row = row.to_dict()
            full_row["shown_type"] = shown_type
            full_row["shown_text"] = shown_text
            full_rows.append(full_row)

            eval_rows.append({
                "grupo": group_id,
                "id": row["id_grupo"],
                "source": row["source"],
                "text": shown_text,
            })

    df_full = pd.DataFrame(full_rows)
    df_eval = pd.DataFrame(eval_rows)

    df_full.to_csv(args.output_full_csv, index=False, encoding="utf-8")
    df_eval.to_csv(args.output_eval_csv, index=False, encoding="utf-8")

    print(f"CSV completo generado: {args.output_full_csv}")
    print(f"CSV de evaluación generado: {args.output_eval_csv}")


if __name__ == "__main__":
    main()
