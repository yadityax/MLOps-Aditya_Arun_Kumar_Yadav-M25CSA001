# Q1: Bengali-to-English NLP Translation Evaluation

Model: [Helsinki-NLP/opus-mt-bn-en](https://huggingface.co/Helsinki-NLP/opus-mt-bn-en)

## Files
| File | Description |
|---|---|
| `translate.py` | Translates Bengali input to English, saves `output.txt`, reports BLEU score |
| `Dockerfile` | Container with all dependencies |
| `requirements.txt` | Python dependencies |
| `input.txt` | Bengali source sentences (downloaded from exam link) |
| `reference.txt` | English reference translations (downloaded from exam link) |
| `output.txt` | Model-generated English translations (produced by script) |

## Setup — without Docker

```bash
pip install -r requirements.txt
python translate.py --input input.txt --reference reference.txt
```

## Setup — with Docker

**Build:**
```bash
docker build -t q1-translate .
```

**Run** (mount current directory so input/output files are accessible):
```bash
docker run --rm -v $(pwd):/app q1-translate
```

> `input.txt` and `reference.txt` must be in the same directory before running.

## Results

**First translated line:**
____________________________________________________________________

**BLEU Score:** ____
