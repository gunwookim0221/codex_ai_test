"""LLM-based SmartThings test case generator.


This script provides utilities to generate test cases from textual and
optional image inputs using a Hugging Face model. Existing test case
files can be included in the context using a simple retrieval step and
results can be exported to CSV. When no test case directory is
supplied, a small set of built in examples is used so the script works
without any external files.

The implementation uses LangChain for compatibility so that the
underlying model can be swapped easily.

"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import csv



DEFAULT_CASES = [
    "로그인 기능 | 앱 실행 후 로그인 정보를 입력한다 | 로그인 성공 메시지가 표시된다",
    "로그아웃 기능 | 설정 화면에서 로그아웃 버튼을 누른다 | 로그아웃되고 로그인 화면이 나타난다",
    "프로필 수정 | 프로필 화면에서 이름을 변경한다 | 변경된 이름이 저장된다",
]


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

        model_name: str = "beomi/KoAlpaca-Polyglot-12.8B",

        test_case_dir: Optional[str] = None,
        load_model: bool = True,
    ) -> None:
        self.llm = None
        self.tokenizer = None
        if load_model:

            try:
                from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
                from langchain.llms import HuggingFacePipeline

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                pipe = pipeline(
                    "text2text-generation",
                    model=model,
                    tokenizer=self.tokenizer,
                )
                self.llm = HuggingFacePipeline(pipeline=pipe)
            except Exception:
                # Dependencies may be missing in minimal environments
                self.llm = None

        self.vectorstore = None
        if test_case_dir:
            self._load_vectorstore(directory=test_case_dir)
        else:
            self._load_vectorstore(cases=DEFAULT_CASES)

    def _load_vectorstore(
        self,
        directory: Optional[str] = None,
        cases: Optional[List[str]] = None,
    ) -> None:
        """Load test cases into a FAISS vector store."""
        documents = []
        if directory:
            paths = list(Path(directory).rglob("*.txt"))
            for path in paths:
                documents.append(path.read_text())
        if cases:
            documents.extend(cases)
        if not documents:
            return
        try:
            from langchain.embeddings import HuggingFaceEmbeddings
            from langchain.vectorstores import FAISS

            embeddings = HuggingFaceEmbeddings()
            self.vectorstore = FAISS.from_texts(documents, embeddings)
        except Exception:
            self.vectorstore = None


    def _retrieve_context(self, query: str, k: int = 2) -> str:
        """Retrieve similar test cases as additional context."""
        if not self.vectorstore:
            return ""
        docs = self.vectorstore.similarity_search(query, k=k)
        return "\n".join(d.page_content for d in docs)

    def _build_prompt(self, change_desc: str, context: str) -> str:
        """Construct the prompt for the language model."""
        from langchain.prompts import PromptTemplate

        template = (

            "당신은 SmartThings 앱의 QA 어시스턴트입니다.\n"
            "다음은 새로운 변경 사항에 대한 설명입니다:\n{change}\n"
            "관련된 기존 테스트 케이스는 다음과 같습니다:\n{context}\n"
            "아래 형식으로 번호가 매겨진 테스트 케이스를 제시하세요:\n"
            "번호. 설명 | 절차 | 기대 결과"

        )
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

    parser.add_argument(
        "--model",
        help="Hugging Face model name",
        default="beomi/KoAlpaca-Polyglot-12.8B",
    )

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
