# PromptForge — OpenEnv RL Environment

**PromptForge** is a reinforcement learning environment for the **Meta PyTorch × Hugging Face OpenEnv Hackathon**. It trains RL agents to autonomously detect and eliminate **Prompt Debt** from production LLM system prompts.

Built on the official [`openenv-core`](https://pypi.org/project/openenv-core/) library from Meta PyTorch.

---

## What is Prompt Debt?

Prompt Debt is the accumulation of technical debt inside LLM system prompts:

- **Dead Instructions** — rules that no longer apply
- **Duplicate Few-Shot Examples** — redundant examples that waste tokens
- **Mandate-Prohibition Conflicts** — contradictory "always do X / never do Y" pairs
- **Hidden Tool-Schema Dependencies** — deprecated parameter names that cause hallucinations

PromptForge formalises this as an AST-based RL task with deterministic grading.

---

## Architecture

```
├── __init__.py                    ← Package exports (Action, Observation, Client)
├── models.py                      ← PromptForgeAction + PromptForgeObservation
├── client.py                      ← PromptForgeEnvClient
└── server/
    ├── app.py                     ← create_app(PromptForgeEnvironment, ...)
    ├── promptforge_environment.py ← openenv-core Environment subclass
    ├── ast_parser.py              ← Recursive prompt AST (DOCUMENT→SECTION→RULE)
    ├── tasks.py                   ← 3 PromptDebt scenarios + ground truth fixtures
    ├── graders.py                 ← Deterministic JSON Grader (OpenAI/Groq API)
    ├── reward.py                  ← (token_reduction × quality) − penalties
    └── gradio_ui.py               ← Gradio UI mounted on the FastAPI app

Root:
├── app.py                        ← Compatibility wrapper for the PromptForge server entrypoint
├── inference.py                  ← Baseline agent (OpenAI-compatible client)
├── Dockerfile                    ← Root build for HF Spaces deployment
├── openenv.yaml                  ← OpenEnv submission manifest
└── test_smoke.py                 ← Lightweight smoke checks (log contract + openenv validate)
```

---

## Three Task Scenarios

| Task                      | Difficulty | Prompt Debt Type                                                                      |
| ------------------------- | ---------- | ------------------------------------------------------------------------------------- |
| `task_few_shot_debt`      | Easy       | Duplicate / placeholder few-shot examples                                             |
| `task_mandate_conflict`   | Medium     | "always claim X" vs "never claim X" conflicts                                         |
| `task_schema_archaeology` | Hard       | Deprecated param `ticket_priority` vs. correct `ticket_priority_level` (lexical trap) |

---

## Action Space

All actions use the **flat `PromptForgeAction` model** (inherits from openenv-core `Action`):

| `action_type`   | Required fields               | Effect                                               |
| --------------- | ----------------------------- | ---------------------------------------------------- |
| `START_EPISODE` | `task_difficulty`             | Reset episode with chosen difficulty                 |
| `PRUNE_BRANCH`  | `node_id`                     | Permanently delete AST node + subtree                |
| `MOVE_NODE`     | `node_id`, `target_parent_id` | Relocate node to new parent                          |
| `MERGE_NODES`   | `node_id`, `node_id_2`        | Combine two sibling nodes                            |
| `PROBE`         | `node_id`                     | Non-destructive coherence check (AST restored after) |
| `SUBMIT`        | —                             | Terminate episode, trigger grader                    |

---

## Reward Formula

The reward function uses **Potential-Based Reward Shaping (PBRS)** to provide
dense, incremental feedback at every step (Ng et al., 1999).

```
Intermediate steps (PRUNE_BRANCH / MOVE_NODE / MERGE_NODES):
    reward = Φ(s') − Φ(s)
    where Φ(s) = (original_tokens − current_tokens) / original_tokens

PROBE:       reward = −0.02  (per-use information cost)
Destructive: reward = −0.50  (perplexity guard fires if coherence degrades)
SUBMIT:      reward = token_reduction_ratio × quality_score

Episode score = MinMaxNorm(Σ rewards)   # cumulative return, clipped to [0, 1]
```

---

## Quick Start

The current submission is validated with `openenv validate` and the root `inference.py` emits the required `[START]`, `[STEP]`, and `[END]` lines, including the final `score` field expected by the sample inference contract.

### Submission Checklist

- `inference.py` is in the project root.
- `inference.py` uses `OpenAI` client with evaluator-injected `API_BASE_URL` and `API_KEY`.
- OpenEnv manifest exists: `openenv.yaml`.
- Docker build context exists: `Dockerfile`.
- `python -m openenv.cli validate` passes.

### Run Locally

```bash
pip install -r requirements.txt
# create .env manually if needed and set at least: API_BASE_URL, API_KEY

# Start server (port 7860)
uvicorn server.app:app --port 7860

# In another terminal — run baseline agent
python inference.py
```

### Test with the Environment Directly

```python
from models import PromptForgeAction
from server.promptforge_environment import PromptForgeEnvironment

env = PromptForgeEnvironment()
obs = env.reset()

# Start a hard episode
obs = env.step(PromptForgeAction(action_type="START_EPISODE", task_difficulty="hard"))
print(f"{obs.original_token_count} tokens | {len(obs.ast_summary)} AST nodes")

# Inspect nodes and prune debt
for node in obs.ast_summary:
    print(node["node_id"][:8], node["node_type"], node["content_preview"][:60])

obs = env.step(PromptForgeAction(action_type="PRUNE_BRANCH", node_id=obs.ast_summary[2]["node_id"]))
obs = env.step(PromptForgeAction(action_type="SUBMIT"))
print(f"done={obs.done}, reward={obs.reward:.3f}")
```

### Groq Configuration (fast, free testing)

```bash
# .env
GRADER_API_BASE=https://api.groq.com/openai/v1
GRADER_MODEL_NAME=llama-3.3-70b-versatile
GRADER_API_KEY=gsk_... # Groq API key for grader

API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile
API_KEY=gsk_...        # Inference key used by OpenAI client
```

### Run Smoke Tests

```bash
python test_smoke.py
```

### Deploy to Hugging Face Spaces

```bash
# Using openenv CLI (recommended)
openenv push --repo-id raunaqmittal2004/promptforge

# OR upload manually
python upload_space.py
```

### Validate Before Submission

```bash
python test_smoke.py
python -m openenv.cli validate
bash scripts/validate-submission.sh https://your-space.hf.space .
```

### Baseline Performance Scores

Measured baseline benchmark using `inference.py` (local server on `127.0.0.1:7860`, inference model `Qwen/Qwen2.5-72B-Instruct`, grader `llama-3.3-70b-versatile`):

| Task   | Steps | Cumulative Score | Terminal Quality Bonus | Success Rate |
| ------ | ----: | ---------------: | ---------------------: | -----------: |
| Easy   |     6 |             0.79 |                   0.33 |         1.00 |
| Medium |     3 |             0.67 |                   0.10 |         1.00 |
| Hard   |    10 |             0.81 |                   0.38 |         1.00 |

_Scores reflect the **Potential-Based Reward Shaping (PBRS)** and **Cumulative Return Normalization**, where intermediate efficient pruning correctly yields continuous positive gradients to prevent Advantage Collapse under GRPO._

Recent quality improvement: the hard-task grader now accepts both `HIGH` and `CRITICAL` ticket priorities for severe payment outages, which better reflects real incident triage behavior.

_Zero-shot summarization models struggle significantly with the `Hard` schema task due to rigid API parameter constraints, necessitating multi-step `PROBE` state exploration._

---

## Environment Variables

| Variable                          | Default                            | Description                                |
| --------------------------------- | ---------------------------------- | ------------------------------------------ |
| `GRADER_API_BASE`                 | `https://api.openai.com/v1`        | Grader API endpoint                        |
| `GRADER_MODEL_NAME`               | `gpt-4o-mini`                      | Grader model                               |
| `GRADER_API_KEY`                  | _(falls back to `HF_TOKEN`)_       | Grader API key                             |
| `API_BASE_URL`                    | `https://router.huggingface.co/v1` | Inference agent endpoint                   |
| `MODEL_NAME`                      | `Qwen/Qwen2.5-72B-Instruct`        | Inference agent model                      |
| `API_KEY`                         | _(required)_                       | Inference API key (proxy-injected in eval) |
| `ENV_BASE_URL`                    | `http://localhost:7860`            | PromptForge server URL                     |
| `PERPLEXITY_THRESHOLD_MULTIPLIER` | `1.5`                              | PPL guard threshold                        |
| `PERPLEXITY_PENALTY_SCALAR`       | `-0.5`                             | PPL guard penalty                          |


---
title: PromptForge
emoji: 🔨
colorFrom: purple
colorTo: blue
sdk: docker
pinned: true
app_port: 7860
tags:
  - openenv
  - reinforcement-learning
  - llmops
  - prompt-engineering
---
