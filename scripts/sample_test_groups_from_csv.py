#!/usr/bin/env python3
import argparse
import pandas as pd
import random


def token_len(text: str) -> int:
    return len(text.strip().split())


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_csv", type=str, required=True)
    parser.add_argument("--output_full_csv", type=str, default="sample_full.csv")
    parser.add_argument("--output_eval_csv", type=str, default="sample_eval.csv")

    parser.add_argument("--bleu_alt_max", type=float, default=80.0)
    parser.add_argument("--bleu_min", type=float, default=10.0)

    parser.add_argument("--bleu_mid_min", type=float, default=10.0)
    parser.add_argument("--bleu_mid_max", type=float, default=25.0)

    parser.add_argument("--bleu_high_min", type=float, default=25.0)
    parser.add_argument("--bleu_high_max", type=float, default=100.0)

    parser.add_argument("--min_src_tokens", type=int, default=4)
    parser.add_argument("--num_groups", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    random.seed(args.seed)

    # -----------------------------
    # Cargar datos
    # -----------------------------
    df = pd.read_csv(args.input_csv)

    # -----------------------------
    # Filtros globales
    # -----------------------------
    df = df[df["bleu_alt"] < args.bleu_alt_max].copy()
    df["src_len"] = df["source"].apply(token_len)
    df = df[df["src_len"] >= args.min_src_tokens]
    df = df[df["bleu"] >= args.bleu_min]

    # -----------------------------
    # Pools por rango BLEU
    # -----------------------------
    df_high = df[
        (df["bleu"] >= args.bleu_high_min) &
        (df["bleu"] <= args.bleu_high_max)
    ].copy()

    df_mid = df[
        (df["bleu"] >= args.bleu_mid_min) &
        (df["bleu"] < args.bleu_high_min)
    ].copy()

    if len(df_high) < args.num_groups:
        raise ValueError(
            f"No hay suficientes ejemplos BLEU alto: {len(df_high)} < {args.num_groups}"
        )

    mid_per_group = len(df_mid) // args.num_groups
    if mid_per_group < 1:
        raise ValueError(
            f"No hay suficientes ejemplos BLEU intermedios para {args.num_groups} grupos"
        )

    # -----------------------------
    # Muestreo estratificado adaptativo
    # -----------------------------
    df_high = df_high.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    df_mid = df_mid.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    selected_rows = []
    high_ptr = 0
    mid_ptr = 0

    for g in range(args.num_groups):
        group_rows = []

        # 1 BLEU alto
        group_rows.append(df_high.iloc[high_ptr])
        high_ptr += 1

        # BLEU intermedios adaptativos
        for _ in range(mid_per_group):
            group_rows.append(df_mid.iloc[mid_ptr])
            mid_ptr += 1

        selected_rows.extend(group_rows)

    df_sel = pd.DataFrame(selected_rows).reset_index(drop=True)

    # -----------------------------
    # Construcción de grupos y CSVs
    # -----------------------------
    full_rows = []
    eval_rows = []

    idx = 0
    for g in range(args.num_groups):
        group_size = 1 + mid_per_group
        group_df = df_sel.iloc[idx: idx + group_size].copy()
        idx += group_size

        group_id = g + 1
        group_df["grupo"] = group_id
        group_df["id_grupo"] = range(1, group_size + 1)

        # Elegir 2 posiciones para target
        target_positions = set(
            random.sample(range(group_size), min(2, group_size))
        )

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

    print("✅ Muestreo completado correctamente")
    print(f"   Grupos           : {args.num_groups}")
    print(f"   BLEU alto/grupo  : 1")
    print(f"   BLEU mid/grupo   : {mid_per_group}")
    print(f"   Total ejemplos   : {len(df_full)}")
    print(f"📄 CSV completo     : {args.output_full_csv}")
    print(f"📄 CSV evaluación   : {args.output_eval_csv}")


if __name__ == "__main__":
    main()
