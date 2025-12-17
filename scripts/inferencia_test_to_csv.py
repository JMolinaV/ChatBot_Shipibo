#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from sacrebleu import sentence_bleu
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            ex = json.loads(line)
            yield ex["src"].strip(), ex["tgt"].strip()


def load_test_split(data_dir: Path):
    test_path = data_dir / "test.jsonl"
    if not test_path.exists():
        raise FileNotFoundError(f"No se encontró test.jsonl en {data_dir}")
    return list(read_jsonl(test_path))


def make_id(src: str, tgt: str) -> str:
    h = hashlib.sha1()
    h.update((src + tgt).encode("utf-8"))
    return h.hexdigest()


def get_langs_from_direction(direction: str):
    if direction == "shp2es":
        return "quy_Latn", "spa_Latn"
    elif direction == "es2shp":
        return "spa_Latn", "quy_Latn"
    else:
        raise ValueError("direction debe ser shp2es o es2shp")


def load_model_and_tokenizer(model_folder: Path, src_lang: str, tgt_lang: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_folder,
        src_lang=src_lang,
        tgt_lang=tgt_lang,
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(model_folder)

    if hasattr(tokenizer, "lang_code_to_id") and tgt_lang in tokenizer.lang_code_to_id:
        forced_bos = tokenizer.lang_code_to_id[tgt_lang]
    else:
        forced_bos = tokenizer.convert_tokens_to_ids(tgt_lang)

    model.config.forced_bos_token_id = forced_bos
    model.config.decoder_start_token_id = forced_bos

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    return tokenizer, model, device


def translate_batch(texts, tokenizer, model, device, max_len, num_beams):
    enc = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len,
    )
    enc = {k: v.to(device) for k, v in enc.items()}

    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_len,
            num_beams=num_beams,
        )

    return tokenizer.batch_decode(out, skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--direction", type=str, required=True,
                        help="shp2es o es2shp")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--output_csv", type=str, default="test_inference.csv")

    args = parser.parse_args()

    src_lang, tgt_lang = get_langs_from_direction(args.direction)

    data = load_test_split(Path(args.data_dir))

    tokenizer, model, device = load_model_and_tokenizer(
        Path(args.model_folder),
        src_lang,
        tgt_lang,
    )

    rows = []

    for i in range(0, len(data), args.batch_size):
        batch = data[i : i + args.batch_size]
        src_batch = [x[0] for x in batch]
        tgt_batch = [x[1] for x in batch]

        preds = translate_batch(
            src_batch,
            tokenizer,
            model,
            device,
            args.max_len,
            args.num_beams,
        )

        for src, tgt, pred in zip(src_batch, tgt_batch, preds):
            uid = make_id(src, tgt)

            bleu_pred = sentence_bleu(
                pred,
                [tgt],
                tokenize="13a",
            ).score

            bleu_alt = sentence_bleu(
                src,
                [tgt],
                tokenize="13a",
            ).score

            rows.append({
                "id": uid,
                "source": src,
                "target": tgt,
                "inference": pred,
                "bleu": round(bleu_pred, 2),
                "bleu_alt": round(bleu_alt, 2),
            })

    out_path = Path(args.output_csv)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "source",
                "target",
                "inference",
                "bleu",
                "bleu_alt",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    print(f"CSV generado con {len(rows)} registros en {out_path}")


if __name__ == "__main__":
    main()