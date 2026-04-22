import argparse
import os
import torch
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import sacrebleu


def load_model(model_name: str, device: str):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def translate_batch(lines: list[str], tokenizer, model, device: str) -> list[str]:
    inputs = tokenizer(lines, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
    with torch.no_grad():
        translated = model.generate(**inputs)
    return [tokenizer.decode(t, skip_special_tokens=True) for t in translated]


def main():
    parser = argparse.ArgumentParser(description="Bengali-to-English translation using Helsinki-NLP/opus-mt-bn-en")
    parser.add_argument("--input", required=True, help="Path to Bengali input text file")
    parser.add_argument("--reference", required=True, help="Path to English reference text file")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size for translation")
    parser.add_argument("--output", default="output.txt", help="Path to save translated output")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    model_name = "Helsinki-NLP/opus-mt-bn-en"
    print(f"Loading model: {model_name} ...")
    tokenizer, model = load_model(model_name, device)

    with open(args.input, "r", encoding="utf-8") as f:
        input_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    print(f"Translating {len(input_lines)} lines ...")
    translations = []
    for i in tqdm(range(0, len(input_lines), args.batch_size), desc="Translating"):
        batch = input_lines[i : i + args.batch_size]
        translations.extend(translate_batch(batch, tokenizer, model, device))

    output_path = os.path.join(os.path.dirname(os.path.abspath(args.output)), "output.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        for line in translations:
            f.write(line + "\n")
    print(f"\nTranslations saved to: {output_path}")

    print(f"\nFirst translated line:\n  {translations[0]}")

    with open(args.reference, "r", encoding="utf-8") as f:
        reference_lines = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    # sacrebleu expects a list of hypothesis strings and a list-of-lists of reference strings
    result = sacrebleu.corpus_bleu(translations, [reference_lines])
    print(f"\nBLEU Score: {result.score:.2f}")
    print(f"  (signature: {result.format()})")


if __name__ == "__main__":
    main()
