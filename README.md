# codex_ai_test

This repository contains simple examples for Codex.

## Pyramid script

Run `pyramid.py` with a height value to print a star pyramid:

```bash
python pyramid.py 5
```

## SmartThings Test Case Generator

`test_case_generator.py` creates test cases from a description of app changes. It
uses a Hugging Face model (default: `beomi/KoAlpaca-Polyglot-12.8B`) via LangChain.
The prompts and output are in Korean. Existing test cases can be supplied in a directory of `.txt` files to provide additional context.

Example usage:

```bash
python test_case_generator.py "Change description" \
    --test-case-dir path/to/cases \
    --output cases.csv
```
