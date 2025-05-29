# codex_ai_test

This repository contains simple examples for Codex.

## Pyramid script

Run `pyramid.py` with a height value to print a star pyramid:

```bash
python pyramid.py 5
```

## SmartThings Test Case Generator

`test_case_generator.py` creates test cases from a description of app changes. It
uses a Hugging Face model via LangChain. Existing test cases can be supplied in
a directory of `.txt` files to provide additional context. If no directory is
given, a small set of built-in cases is used instead so the script works out of
the box.

Example usage:

```bash
python test_case_generator.py "Change description" \
    --test-case-dir path/to/cases \
    --output cases.csv
```
