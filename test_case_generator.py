"""LLM-based SmartThings test case generator.

This script provides utilities to generate test cases from textual
and optional image inputs using a Hugging Face model. Existing test
case files can be included in the context using a simple retrieval
step. Results can be exported to CSV.

This implementation is intentionally lightweight and uses LangChain
for compatibility so that the underlying model can be swapped easily.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import csv


# Some simple default test cases used when the user does not supply their own
# files. They provide basic examples for the retrieval step.
DEFAULT_CASES = [
    "Login with valid credentials should take the user to the home screen",
    "Logging out should return the user to the login screen",
    "Adding a new device should display the device in the list",
    "Removing a device should no longer show it in the list",
]

import csv


@dataclass
class TestCase:
    """Data class representing a single test case."""

    id: int
    description: str
    steps: str
    expected: str


class TestCaseGenerator:
    """Generate test cases using a Hugging Face model."""

    def __init__(
        self,
        model_name: str = "google/flan-t5-base",
        test_case_dir: Optional[str] = None,
        load_model: bool = True,
    ) -> None:
        self.llm = None
        self.tokenizer = None
        self.vectorstore = None
        self.case_texts: List[str] = []

        if load_model:
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
            from langchain.llms import HuggingFacePipeline

            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            pipe = pipeline("text2text-generation", model=model, tokenizer=self.tokenizer)
            self.llm = HuggingFacePipeline(pipeline=pipe)

        # Always attempt to load existing or default test cases. When model loading
        # is skipped, retrieval will fall back to simple text matching so that the
        # script remains functional without extra dependencies.
        self._load_cases(directory=test_case_dir, build_vectorstore=load_model)

    def _load_cases(self, directory: Optional[str], build_vectorstore: bool) -> None:
        """Load existing or default test cases.

        When ``build_vectorstore`` is True and LangChain is available, a FAISS
        vector store will be created for similarity search. Otherwise the texts
        are stored in ``self.case_texts`` for simple retrieval.
        """

        documents: List[str] = []
        if directory:
            paths = list(Path(directory).rglob("*.txt"))
            for path in paths:
                try:
                    documents.append(path.read_text())
                except OSError:
                    continue

        if not documents:
            documents = DEFAULT_CASES

        self.case_texts = documents

        if not build_vectorstore:
            return

        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import FAISS
        except Exception:
            return

        embeddings = HuggingFaceEmbeddings()
        self.vectorstore = FAISS.from_texts(documents, embeddings)

    def _retrieve_context(self, query: str, k: int = 2) -> str:
        """Retrieve similar test cases as additional context."""
        if self.vectorstore:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n".join(d.page_content for d in docs)

        if self.case_texts:
            return "\n".join(self.case_texts[:k])

        return ""

    def _build_prompt(self, change_desc: str, context: str) -> str:
        """Construct the prompt for the language model."""
        template = (
            "You are a QA assistant generating test cases for the SmartThings app.\n"
            "Here is the description of a new change:\n{change}\n"
            "Here are related existing test cases:\n{context}\n"
            "Provide a numbered list of test cases in the format:\n"
            "ID. Description | Steps | Expected Result"
        )
        try:
            from langchain.prompts import PromptTemplate
        except Exception:
            return template.format(change=change_desc, context=context)

        prompt = PromptTemplate.from_template(template)
        return prompt.format(change=change_desc, context=context)

    def generate(self, change_desc: str, image_desc: str | None = None, num_cases: int = 5) -> List[TestCase]:
        """Generate test cases for the given change description."""
        combined_desc = change_desc
        if image_desc:
            combined_desc += f"\nImage description: {image_desc}"

        context = self._retrieve_context(combined_desc)
        prompt = self._build_prompt(combined_desc, context)
        if not self.llm:
            raise RuntimeError("Language model is not loaded")

        output = self.llm(prompt)

        cases: List[TestCase] = []
        for line in output.splitlines():
            if not line.strip() or "|" not in line:
                continue
            parts = [part.strip() for part in line.split("|")]
            if len(parts) != 3:
                continue
            id_part, desc = parts[0].split(".", 1)
            case = TestCase(
                id=int(id_part.strip()),
                description=desc.strip(),
                steps=parts[1],
                expected=parts[2],
            )
            cases.append(case)
            if len(cases) >= num_cases:
                break
        return cases

    def export_csv(self, cases: Iterable[TestCase], path: str) -> None:
        """Export test cases to a CSV file."""
        with open(path, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["ID", "Description", "Steps", "Expected"])
            for c in cases:
                writer.writerow([c.id, c.description, c.steps, c.expected])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate SmartThings test cases.")
    parser.add_argument("description", help="Text describing the change")
    parser.add_argument("--image", help="Optional image description", default=None)
    parser.add_argument("--model", help="Hugging Face model name", default="google/flan-t5-base")
    parser.add_argument("--test-case-dir", help="Directory of existing test case txt files")
    parser.add_argument("--output", help="Output CSV file", default="test_cases.csv")
    parser.add_argument("--no-model", action="store_true", help="Skip loading the language model")
    args = parser.parse_args()

    generator = TestCaseGenerator(
        model_name=args.model,
        test_case_dir=args.test_case_dir,
        load_model=not args.no_model,
    )
    cases = generator.generate(args.description, image_desc=args.image)
    generator.export_csv(cases, args.output)
    print(f"Generated {len(cases)} test cases -> {args.output}")
