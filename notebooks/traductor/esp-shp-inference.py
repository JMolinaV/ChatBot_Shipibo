import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def cargar_traductor(ruta_modelo):
    """
    Carga el tokenizer y el modelo desde una carpeta local.
    """
    print(f"Cargando modelo desde: {ruta_modelo}")
    tokenizer = AutoTokenizer.from_pretrained(ruta_modelo)
    model = AutoModelForSeq2SeqLM.from_pretrained(ruta_modelo)
    model.eval()
    return tokenizer, model


def traducir(texto, tokenizer, model, max_len=128):
    """
    Traduce una oración usando el modelo cargado.
    """
    # Tokenizar
    inputs = tokenizer(
        texto,
        return_tensors="pt",
        truncation=True,
        padding=True
    )

    # Generar traducción
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_length=max_len,
            num_beams=5,
            early_stopping=True
        )

    # Decodificar
    return tokenizer.decode(output_ids[0], skip_special_tokens=True)


if __name__ == "__main__":
    # Ruta al modelo entrenado
    RUTA = "./modelo-trad-esp-ship-entrenado"

    tokenizer, model = cargar_traductor(RUTA)

    print("=== Traductor listo ===")
    print("Escribe una frase en Shipibo o Español:")
    
    while True:
        texto = input("\n> ")
        if texto.lower() in ["salir", "exit"]:
            break

        traduccion = traducir(texto, tokenizer, model)
        print("Traducción:", traduccion)
