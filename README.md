# PromptGuard-Eval (v0.1)

A lightweight, open-source evaluation harness for testing conversational AI systems and open-weight LLMs against **prompt injections**, **jailbreak framing**, **misinformation traps**, and **unsafe compliance requests**.

## Why This Exists
While frontier AI labs maintain internal red-teaming suites, independent developers and smaller teams deploying open-weight models or LLM wrappers often lack accessible, reproducible safety harnesses. **PromptGuard-Eval** pairs deterministic string-leak checks with an LLM-as-a-Judge rubric to produce auditable safety pass rates across failure categories.

## Quickstart

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
