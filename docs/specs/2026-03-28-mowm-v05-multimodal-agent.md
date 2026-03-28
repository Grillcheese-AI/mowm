# MoWM v0.5.0 Vision — Multimodal Embodied Agent

**Date:** 2026-03-28
**Status:** Vision (after v0.4.0 reasoning decoder)

## The Stack

```
v0.1.0  Axiom library + routing + pipeline         (DONE, 121 tests)
v0.2.0  Executable formulas + causal + logic        (DONE, 381 tests)
v0.3.0  Q&A pipeline (direct + augmented)           (DONE, 542 tests)
v0.4.0  Reasoning decoder (be the LLM)             (IN PROGRESS)
v0.5.0  Multimodal embodied agent                   (THIS SPEC)
```

## v0.5.0 Components

### 1. Sensory Cortex (DenseNet CNN → VSA)
- DenseNet extracts spatial features from images/charts/maps
- Projection matrix (trained via contrastive learning) maps visual
  features into the same VSA space as text axioms
- `VisionProjector.embed_image_to_graph_space(image) -> (k, l) VSA vec`
- Same cosine similarity search as text → axioms light up from images
- Reference: cubemind/perception/grilly_densenet.py already has DenseNet

### 2. Causal News Extractor (Text → SemanticEdge)
- spaCy dependency parsing + verb lexicon → extract Subject-Verb-Object
- Map verbs to EdgeType: "crush" → DECREASES, "boost" → INCREASES
- No LLM needed — deterministic SVO extraction
- Output: SemanticEdge(source, target, edge_type, weight=0.8)
- Ephemeral edges (tagged metadata["ephemeral"]=True)

### 3. Live News Injector (RSS → Ephemeral Graph)
- Pulls from 17 RSS feeds (already spec'd in v0.3.0 features)
- Vectorizes headlines via NV-Embed codebook projection
- Injects temporary SemanticEdges into KnowledgeContext
- News events don't pollute the verified AxiomLibrary
- Time-decayed: ephemeral edges lose weight over hours

### 4. Vocal Tract (Reasoning → TTS)
- Composer output → TTS engine (local VITS/Tacotron2 or API)
- Prosody modulated by reasoning intent:
  - EXPLAIN: slow, instructive
  - COMPARE: dynamic, analytical
  - BREAKING_NEWS: urgent, faster
  - CAUSAL: measured, step-by-step
- SemanticPayload carries tone + intent for prosody selection

### 5. Full Multimodal Pipeline

```
User asks (voice or text)
    |
    v
[STT if voice] -> text question
    |
    v
[DenseNet if image attached] -> visual VSA vector
    |
    v
[News Injector] -> ephemeral edges from live feeds
    |
    v
[QAPipeline + ReasoningDecoder]
  - KnowledgeSearch (text + visual + news context)
  - ChainBuilder (derivation from axioms)
  - CausalTracer (cancellation + ephemeral news edges)
  - Composer (structured text)
    |
    v
[VocalTract] -> spoken response with prosody
```

### Example: Hurricane Question

"What is happening with the hurricane and why is it so destructive?"

1. **News Injector**: pulls NOAA reports, "Category 4, 150mph winds"
2. **DenseNet**: live radar image → Fluid Dynamics, Coriolis Effect
3. **KnowledgeSearch**: KE = 0.5*m*v^2 (v is squared!)
4. **CausalTracer**: wind_speed → kinetic_energy (+), v^2 means exponential
5. **Composer**: "Wind speeds of 150mph. Because KE scales with v^2,
   doubling wind speed quadruples destructive force."
6. **VocalTract**: BREAKING_NEWS prosody, urgent cadence

### 6. Combiner-Axial Attention (Token Pre-processor)
- O(L*sqrt(L)) sub-quadratic attention from cubemind/reasoning/combiner.py
- Processes raw token streams BEFORE VSA block-code encoding
- Flow: Raw tokens → Combiner-Axial → SequenceEncoder → block-code VSA
- Enables 10,000+ token context (WSJ articles, full documents, genomes)
- Local direct attention + global block summaries with joint normalization
- Already implemented in cubemind — just needs integration into mowm pipeline

## Dependencies (not yet in mowm)

- spaCy (`en_core_web_sm`) for causal news extraction
- DenseNet from cubemind/perception or torchvision
- TTS engine (VITS, Tacotron2, or API)
- RSS feed reader (feedparser)
- STT for voice input (whisper or API)

## Key Insight

The DenseNet doesn't replace text reasoning — it adds a **visual input
channel** that maps to the same VSA space. Images become axiom queries.
News becomes ephemeral graph edges. Voice becomes prosody-modulated
output. The core reasoning engine (v0.4.0) stays the same.
