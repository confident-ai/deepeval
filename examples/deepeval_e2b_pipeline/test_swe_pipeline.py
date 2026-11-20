import pytest
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from e2b_code_interpreter import Sandbox

from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

# ================= SETUP =================

load_dotenv()

with open("tasks.json") as f:
    TASKS = json.load(f)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ================= HELPERS =================

def clean(code: str) -> str:
    code = code.strip()
    if code.startswith("```"):
        code = code.split("```")[1]
    return code.strip()

def normalize(x: str) -> str:
    return str(x).replace("\n", "").strip()

# ================= LLM CODE =================

def generate_code(task: str) -> str:
    prompt = f"""
Write ONLY Python code.
No markdown.
Must print final answer.
Do NOT hardcode result.

Task:
{task}
"""

    r = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )

    return clean(r.choices[0].message.content)

# ================= UNIT TEST =================

def generate_test(task: str, expected: str) -> str:
    prompt = f"""
Write Python unit test.
Use assert.
Print result.

Task:
{task}

Expected:
{expected}
"""

    r = client.chat.completions.create(
        model="gpt-5",
        messages=[{"role": "user", "content": prompt}]
    )

    return clean(r.choices[0].message.content)

# ================= SANDBOX =================

def run_sandbox(code: str) -> str:
    with Sandbox.create() as s:
        r = s.run_code(code)

        if r.logs and r.logs.stdout:
            if isinstance(r.logs.stdout, list):
                return r.logs.stdout[0].strip()
            return str(r.logs.stdout).strip()

        if r.text:
            if isinstance(r.text, list):
                return r.text[0].strip()
            return str(r.text).strip()

        return ""

# ================= DEEPEVAL =================

swe_metric = GEval(
    name="SWE-Bench",
    criteria="""
0.2 valid python
0.3 no runtime errors
0.5 output matches expected
""",
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT
    ],
    threshold=0.5,
    model="gpt-4o-mini"
)

# ================= TEST =================

@pytest.mark.parametrize("data", TASKS)
def test_swe_pipeline(data):

    task = data["task"]
    expected = normalize(data["expected"])

    print("\nTASK:", task)

    # 1. LLM CODE
    code = generate_code(task)
    print("\nCODE:\n", code)

    # 2. RUN CODE
    output = normalize(run_sandbox(code))
    print("OUTPUT:", output)

    # 3. HARD ASSERT
    assert output == expected

    # 4. LLM UNIT TEST
    test_code = generate_test(task, expected)
    print("\nUNIT TEST:\n", test_code)

    test_output = normalize(run_sandbox(test_code))
    print("TEST RESULT:", test_output)

    # 5. DEEPEVAL
    test_case = LLMTestCase(
        input=task,
        actual_output=output,
        expected_output=expected
    )

    assert_test(test_case, [swe_metric])

# ================= RUN =================

if __name__ == "__main__":
    pytest.main([__file__])
