# Jargon 

Privacy policies are essential documents that outline how an organization collects, uses, and protects user data. Analyzing these policies manually can be time-consuming and challenging due to their length and complex language. Therefore, this project leverages natural language processing techniques, specifically NER, to automate the analysis process.

This project aims to analyze privacy policies using Named Entity Recognition (NER). The goal is to automatically extract relevant information from privacy policies, such as business-related topics, legal aspects, regulations, usability factors, educational aspects, technology, and multidisciplinary aspects to name a few.

## Quickstart:

1) Install deps (Poetry)

```bash
cd /Users/arjuns/Downloads/tnc
poetry install
```

2) Set OpenAI key

```bash
export OPENAI_API_KEY=... 
```

3) Annotate 5k policies

```bash
poetry run python -m tnc annotate \
  --sqlite-path /Users/arjuns/Downloads/tnc/release_db.sqlite \
  --entities-path /Users/arjuns/Downloads/tnc/entities.json \
  --limit 5000 \
  --out /Users/arjuns/Downloads/tnc/artifacts/annos \
  --model gpt-4o-mini
```

4) Prepare BIO dataset

```bash
poetry run python -m tnc prepare \
  --annotations /Users/arjuns/Downloads/tnc/artifacts/annos/annotations.jsonl \
  --entities-path /Users/arjuns/Downloads/tnc/entities.json \
  --train-out /Users/arjuns/Downloads/tnc/artifacts/data
```

5) Train

```bash
poetry run python -m tnc train \
  --prepared /Users/arjuns/Downloads/tnc/artifacts/data/train.json \
  --output /Users/arjuns/Downloads/tnc/artifacts/model \
  --base-model bert-base-cased \
  --epochs 3 \
  --batch-size 8
```

6) Predict

```bash
poetry run python -m tnc predict \
  --text "We use cookies and third-party analytics to track usage." \
  --model-dir /Users/arjuns/Downloads/tnc/artifacts/model
```

## License
Jargon is licensed under the MIT License. See the LICENSE file for more details.

## References

[A Multidisciplinary Definition of Privacy Labels: The Story of Princess Privacy and the Seven Helpers? - Johanna Johansena,∗ , Tore Pedersenb , Simone Fischer-Hübnerc , Christian Johansend , Gerardo Schneidere , Arnold Roosendaalf , Harald Zwingelbergg , Anders Jakob Sivesinda , Josef Nollh](https://arxiv.org/pdf/2012.01813.pdf)

[Privacy Policies over Time: Curation and Analysis of a Million-Document Dataset Ryan Amos, Gunes Acar, Eli Lucherini, Mihir Kshirsagar, Arvind Narayanan, Jonathan Mayer](https://arxiv.org/abs/2008.09159)

Special thanks to [Ryan B. Amos](mailto:rbamos@cs.princeton.edu) for providing access to their dataset :)
