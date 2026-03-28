# Rule Auto-Discovery Architecture

Two discovery mechanisms, context-selected:

## Method 1: DiscoveryLoop (Mathematical Rules)
- **When:** Prediction residual is high + context maps to a domain with math operators
- **How:** Context vector → nearest axioms → operator grammar → GP-SR evolves expression trees
- **Output:** New Axiom registered in library (e.g., "F = m * a")
- **Module:** mowm/discovery.py (GeneticProgrammingSR + DiscoveryLoop)

## Method 2: MultiViewHMM (Behavioral Rules)  
- **When:** Context is a temporal/sequential pattern (not a single equation)
- **How:** Sequence of VSA states → unbind deltas → Baum-Welch EM learns transition matrix
- **Output:** Learned HMM transition rule (the "grammar" of the behavior)
- **Module:** cubemind/reasoning/hmm_rule.py (MultiViewHMM)
- Three views: Absolute (raw states), Delta (transformations), Row Bundle (macro-patterns)

## Context-Based Selection
The reasoning decoder should auto-select:
1. If input is (numeric_inputs, numeric_outputs) → DiscoveryLoop (SR)
2. If input is (sequence_of_states) → MultiViewHMM (EM)
3. The CognitiveSNN's hormonal state can modulate: Dopamine floods when either method succeeds, strengthening the routing pathway via STDP.

## Integration with Q&A Pipeline
- "Find the equation relating mass and force" → DiscoveryLoop
- "What pattern do you see in this sequence?" → MultiViewHMM
- Both return new Axioms that get registered in the living library
