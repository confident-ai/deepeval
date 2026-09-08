# DeepEval + E2B Secure LLM Code Evaluation Pipeline

This example demonstrates a secure, reproducible SWE-bench style evaluation pipeline using:

- LLM (OpenAI) → Generate Python code from tasks
- E2B Sandbox → Safely execute generated code
- LLM-generated unit tests → Verify correctness
- DeepEval (GEval) → Structured evaluation + scoring

It is designed for:

- Evaluating LLM code generation
- Running secure sandboxed executions
- Comparing prompts & models
- Building SWE-bench style experiments
---
## Architecture
```bash
Task → LLM (code) → E2B sandbox → output
            ↓
        LLM (unit tests)
            ↓
        E2B sandbox
            ↓
        DeepEval scoring
```
---
## Example Structure
```bash
deepeval_e2b_pipeline/
│
├── test_swe_pipeline.py   # main pipeline
├── tasks.json             # evaluation tasks
├── requirements.txt
├── .env                   # API keys
└── README.md
```
---
## Setup and Installation
### 1. Clone Repository
```bash
git clone https://github.com/yourusername/deepeval_e2b_pipeline.git
cd deepeval_e2b_pipeline
```
### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```
### 3. Install Dependencies
```python
pip install -r requirements.txt
```
### 4. Configure API Keys

Create .env file:
```bash
touch .env
```
Add inside:
```bash
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx
E2B_API_KEY=e2b_xxxxxxxxxxxxxxxx
```
### 5. Run Evaluation Pipeline
```python
deepeval test run test_swe_pipeline.py
```
### 6. Expected Output

You will see:

- LLM generated code
- Sandbox execution result
- Generated unit tests
- DeepEval metric scores

Example:
```bash
TASK: Count 'r's in 'strawberry'
OUTPUT: 3
TEST RESULT: 3
SWE-Bench [GEval]: PASSED
```