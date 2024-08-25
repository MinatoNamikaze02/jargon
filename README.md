# JARGON

## Project Overview

Privacy policies are essential documents that outline how an organization collects, uses, and protects user data. Analyzing these policies manually can be time-consuming and challenging due to their length and complex language. Therefore, this project leverages natural language processing techniques, specifically NER, to automate the analysis process.

This project aims to analyze privacy policies using Named Entity Recognition (NER). The goal is to automatically extract relevant information from privacy policies, such as business-related topics, legal aspects, regulations, usability factors, educational aspects, technology, and multidisciplinary aspects to name a few.

This contextual understanding helps disambiguate potential ambiguities that may arise from analyzing complex textual data.

This is but a primitive approach to the solution

The key features of Jargon include:
- Named Entity Recognition (NER) for contextually understanding privacy policies.
- Highlighting of sections of possible importance (does not guarantee extraction of all relevant sections).

## Dependencies

- spaCy (version 3.6.0)
- Flask (version 2.0.1)

## Getting Started

To get started with Jargon, follow these steps:

```
//clone
>> git clone https://github.com/MinatoNamikaze02/jargon.git

// move to client dir under jargon
>> cd .../jargon/client

// requirements
>> pip install -r requirements.txt

// run
>> python server.py
```

Open up ```http://127.0.0.1:5050/```

## How to Use

Just go to this [website](https://jargon-privacy-policy-analyzer.onrender.com) (The server runs on a free tier service. Expect slow response times :|

<img width="1710" alt="Screenshot 2024-08-25 at 19 50 10" src="https://github.com/user-attachments/assets/d7bca1d5-3b52-4e9e-b96a-f8a3c5470f35">

## License

Jargon is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## References

[A Multidisciplinary Definition of Privacy Labels: The Story of Princess Privacy and the Seven Helpers? - Johanna Johansena,∗ , Tore Pedersenb , Simone Fischer-Hübnerc , Christian Johansend , Gerardo Schneidere , Arnold Roosendaalf , Harald Zwingelbergg , Anders Jakob Sivesinda , Josef Nollh](https://arxiv.org/pdf/2012.01813.pdf)

[Privacy Policies over Time: Curation and Analysis of a Million-Document Dataset Ryan Amos, Gunes Acar, Eli Lucherini, Mihir Kshirsagar, Arvind Narayanan, Jonathan Mayer](https://arxiv.org/abs/2008.09159)

Special thanks to [Ryan B. Amos](mailto:rbamos@cs.princeton.edu) for providing access to their dataset :)

---
