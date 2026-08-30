#!/usr/bin/env python3
"""
English → Telugu Agriculture Glossary Translation
Using IndicTrans2 (AI4Bharat)

Pipeline:
PDF → Text Extraction (pdfplumber) → Preprocessing
    → IndicTrans2 Translation → Telugu CSV

Usage:
    python translation.py --input glossary.pdf --output telugu_glossary.csv

For gated model access, first:
1. Accept access to ai4bharat/indictrans2-en-indic-dist-200M on Hugging Face.
2. Authenticate using either:
       huggingface-cli login
   or:
       export HF_TOKEN=your_token

Optional:
    python translation.py --input glossary.pdf --output telugu_glossary.csv --batch-size 8
"""

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd
import pdfplumber
import torch
from huggingface_hub import login
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor


MODEL_NAME = "ai4bharat/indictrans2-en-indic-dist-200M"
SRC_LANG = "eng_Latn"
TGT_LANG = "tel_Telu"


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Translate an English glossary PDF into Telugu using IndicTrans2."
    )

    parser.add_argument(
        "--input",
        "-i",
        required=True,
        help="Path to the input English glossary PDF.",
    )

    parser.add_argument(
        "--output",
        "-o",
        default="telugu_glossary.csv",
        help="Path for the output CSV file. Default: telugu_glossary.csv",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Number of entries translated at once. Default: 8",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum token generation length. Default: 256",
    )

    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU execution even if CUDA is available.",
    )

    return parser.parse_args()


def authenticate_huggingface():
    """
    Authenticate with Hugging Face if HF_TOKEN is available.

    The model is gated, so the user must first accept access on the
    Hugging Face model page.
    """
    token = os.getenv("HF_TOKEN")

    if token:
        print("Authenticating with Hugging Face using HF_TOKEN...")
        login(token=token, add_to_git_credential=False)
    else:
        print(
            "\nNo HF_TOKEN environment variable found.\n"
            "The IndicTrans2 model is gated. Make sure you have:\n"
            "1. Accepted access to the model on Hugging Face.\n"
            "2. Logged in using 'huggingface-cli login' or set HF_TOKEN.\n"
        )


def load_model(device):
    """Load IndicTrans2 tokenizer, model, and processor."""

    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
    )

    print("Loading IndicTrans2 model...")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    ).to(device)

    model.eval()

    processor = IndicProcessor(inference=True)

    print(f"Model loaded successfully on: {device}")
    return tokenizer, model, processor


def extract_pdf_text(pdf_path):
    """Extract text from every page of a PDF."""

    pages = []

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_number, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""

                if text.strip():
                    pages.append(text)
                else:
                    print(f"Warning: No extractable text found on page {page_number}.")

    except Exception as error:
        raise RuntimeError(f"Failed to read PDF '{pdf_path}': {error}") from error

    return "\n".join(pages)


def clean_entries(text):
    """
    Convert extracted PDF text into glossary entries.

    Current logic:
    - Removes empty lines.
    - Normalizes repeated whitespace.
    - Removes lines containing only page numbers.

    Depending on the source PDF format, this function may need to be
    customized further.
    """

    entries = []

    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()

        if not line:
            continue

        if re.fullmatch(r"\d+", line):
            continue

        entries.append(line)

    return entries


def translate_entries(
    sentences,
    tokenizer,
    model,
    processor,
    device,
    batch_size=8,
    max_length=256,
):
    """Translate English sentences to Telugu in batches."""

    translated = []

    total = len(sentences)

    for start in range(0, total, batch_size):
        batch = sentences[start : start + batch_size]

        processed_batch = processor.preprocess_batch(
            batch,
            src_lang=SRC_LANG,
            tgt_lang=TGT_LANG,
        )

        inputs = tokenizer(
            processed_batch,
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            generated_tokens = model.generate(
                **inputs,
                use_cache=True,
                min_length=0,
                max_length=max_length,
                num_beams=4,
                num_return_sequences=1,
            )

        decoded_batch = tokenizer.batch_decode(
            generated_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        translated_batch = processor.postprocess_batch(
            decoded_batch,
            lang=TGT_LANG,
        )

        translated.extend(translated_batch)

        completed = min(start + batch_size, total)
        print(f"Translated {completed}/{total} entries")

    return translated


def main():
    args = parse_arguments()

    input_path = Path(args.input)
    output_path = Path(args.output)

    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print("Error: The input file must be a PDF.")
        sys.exit(1)

    if args.batch_size <= 0:
        print("Error: --batch-size must be greater than 0.")
        sys.exit(1)

    if args.max_length <= 0:
        print("Error: --max-length must be greater than 0.")
        sys.exit(1)

    device = "cpu" if args.cpu else (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("=" * 60)
    print("English → Telugu Agriculture Glossary Translation")
    print("=" * 60)
    print(f"Input PDF : {input_path}")
    print(f"Output CSV: {output_path}")
    print(f"Device    : {device}")
    print("=" * 60)

    authenticate_huggingface()

    try:
        tokenizer, model, processor = load_model(device)
    except Exception as error:
        print("\nFailed to load IndicTrans2.")
        print(f"Details: {error}")
        print(
            "\nIf you received a gated repository error:\n"
            "1. Visit the IndicTrans2 model page on Hugging Face.\n"
            "2. Accept the access conditions.\n"
            "3. Run 'huggingface-cli login' or set HF_TOKEN.\n"
        )
        sys.exit(1)

    print("\nExtracting text from PDF...")
    raw_text = extract_pdf_text(input_path)

    if not raw_text.strip():
        print("Error: No text could be extracted from the PDF.")
        print(
            "The PDF may be image-based/scanned and require OCR before translation."
        )
        sys.exit(1)

    entries = clean_entries(raw_text)

    if not entries:
        print("Error: No valid glossary entries were found after preprocessing.")
        sys.exit(1)

    print(f"Found {len(entries)} glossary entries.")

    print("\nTranslating entries...")
    telugu_entries = translate_entries(
        entries,
        tokenizer,
        model,
        processor,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    if len(entries) != len(telugu_entries):
        print(
            "Error: Number of translations does not match the number of input entries."
        )
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataframe = pd.DataFrame(
        {
            "English": entries,
            "Telugu": telugu_entries,
        }
    )

    dataframe.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print("Translation completed successfully!")
    print(f"Translated entries: {len(dataframe)}")
    print(f"Saved output to   : {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
