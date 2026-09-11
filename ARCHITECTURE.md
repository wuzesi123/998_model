# Architecture

## Layer 1 — Universal visual process understanding

Video -> causal frame features -> short / medium / global temporal representation -> observable evidence.

The base runner uses deterministic OpenCV evidence so an untrained neural network is never presented as chemistry. A trained TCPT checkpoint can be loaded later.

## Layer 2 — Chemical semantic interpretation

`ReactionContext` contains participants, roles, concentrations/amounts, conditions and procedure. `ORDRAG` retrieves relevant reaction records / local chemistry knowledge. `SemanticReasoner` combines visual evidence + known context + retrieved knowledge while keeping the evidence sources separate.

## Causal RAG policy

By default ORD ingestion indexes inputs, setup, conditions, notes, observations and provenance. It excludes `outcomes` so product/yield/final-result fields do not leak the answer into a real-time inference task.

## Core experiment ablation

1. Visual-only heuristic / visual model.
2. TCPT temporal visual model.
3. TCPT + ReactionContext.
4. TCPT + ReactionContext + ORD RAG.

Evaluation should separately score observable visual tasks and semantic chemical interpretation.
