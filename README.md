# codex_ai_test

This repository contains simple examples for Codex.

## Pyramid script

Run `pyramid.py` with a height value to print a star pyramid:

```bash
python pyramid.py 5
```

## SmartThings Test Case Generator


`test_case_generator.py` creates test cases from a description of app changes.
It uses a Korean Hugging Face model via LangChain. Existing test cases can be
supplied in a directory of `.txt` files to provide additional context. If no
directory is provided, the script falls back to a few built in examples.

Example usage:

```bash
python test_case_generator.py "Change description" \
    --test-case-dir path/to/cases \
    --output cases.csv
```
