"""v2 post-hoc verification subsystem.

The verifier looks at (retrieved evidence, model answer) and judges whether the
answer is grounded. It does NOT regenerate the answer (that would create an
LLM->LLM->LLM loop in which no one can tell who made the mistake).
"""
