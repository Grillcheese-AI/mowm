# The VM's world calls (2026-11-10)

Requested by Nick ("we also need to code the missing function like to create a world and the other ones"). His
decision: **MoWM owns the worlds**; the CubeLang VM calls MoWM through a host-call channel (cubelang
`CUBELANG_FIXES.md`, P5-1 and P5-2). This spec is MoWM's side: one handler per call, built on code that exists.

**In plain words:** a CubeLang program can ask MoWM to create, split, route to or question a world, and MoWM
answers with the numbers and the calibrations behind each decision, so every world decision can be replayed.
- **Over a conventional model:** a mixture of experts decides inside the network, and nobody can replay why.

## Handlers

| Call | Built on | Returns |
|---|---|---|
| `create(name)` | `MoWMPossibility.add_world` / `MoWMRouter.add_world` | the world's id |
| `split(parent)` | `World.spawn_subworld` (refuses at z_max) | the child's id at z + 1, or the code `z_max` |
| `route(text)` | `CubbyBridge.attempt` (route if the best centroid clears the floor and the margin clears `tau_margin`, else spawn) | outcome, best world and score, runner-up, margin, the calibration ids of the floor and `tau_margin` |
| `best_member(world, text)` | `World.best_member` | the member and its score |
| `centroid_score(world, text)` | the router's centroid scoring (inside `MoWMRouter.query_worlds`) | the score |
| `depth(world)`, `parent(world)`, `list()` | the router's world table | integers |
| `learn(world, fact)` | `MoWMPossibility.learn` | whether it was added |
| `retire(world)` | the router's world table | retired; never deleted |
| `specialize(before, after)` | the transition rule (spawn below tau, else consolidate; cubemind's `WorldManager.process_transition`) | the world id, and whether it spawned |

## Rules
- **One encoder:** every score comes from the encoder the thresholds were calibrated on. A call states the encoder's
  id with its result.
- **Thresholds with their calibration:** a result that used a threshold names its calibration (screen, data, date,
  encoder, deployment). A threshold without a calibration is reported as a code default, never as a screen (CubbyLLM
  0064, step 4).
- **The world state's hash:** every result carries a hash of the world tree after the call, so a run report pins
  the state it ran against.
- **Errors are codes:** no such world, z_max, budget, encoder mismatch. Never a default value.
- **Nothing is deleted.** If `prune` deletes worlds today, that is changed to retiring them, with a test.
- **Ids:** state which counter issues each id (the router's from 5000, `spawn_subworld`'s from 1000) and how both
  persist without colliding.

## Tests
- Each handler on fixtures, including `split` at z_max (refused) and `route` at the exact threshold.
- Replay: the same calls in the same order on the same starting state give the same results and the same state
  hash.
- No call deletes a world.
