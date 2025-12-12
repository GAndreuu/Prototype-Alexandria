# Alexandria System Integration Map
## Complete Module Connectivity

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                         ALEXANDRIA UNIFIED CORE                              ║
║                      (alexandria_unified.py - 700 LOC)                       ║
╚═══════════════════════════════════════╦══════════════════════════════════════╝
                                        │
        ┌───────────────────────────────┼───────────────────────────────┐
        │                               │                               │
        ▼                               ▼                               ▼
┌───────────────────┐  ┌───────────────────────────┐  ┌───────────────────┐
│ PHASE 1: FOUNDATION │  │    PHASE 2: REASONING     │  │ PHASE 3: AGENTS  │
└─────────┬─────────┘  └─────────────┬─────────────┘  └────────┬─────────┘
          │                          │                          │
          ▼                          ▼                          ▼
┌──────────────────────┐  ┌──────────────────────────┐  ┌────────────────────────┐
│ nemesis_bridge_      │  │ abduction_compositional_ │  │ agents_compositional_  │
│ integration.py       │  │ integration.py           │  │ integration.py         │
│ (590 LOC)            │  │ (730 LOC)                │  │ (680 LOC)              │
│                      │  │                          │  │                        │
│ • GeometricEFE       │  │ • GeometricGap           │  │ • GeometricActionAgent │
│ • GeometricAction    │  │ • GeodesicHypothesis     │  │ • GeometricBridgeAgent │
│ • select_action_geo  │  │ • detect_gaps_geo        │  │ • GeometricCriticAgent │
│ • update_beliefs_geo │  │ • generate_geo_hyp       │  │ • GeometricOracle      │
└──────────┬───────────┘  │ • validate_hypothesis    │  │ • full_pipeline        │
           │              └────────────┬─────────────┘  └───────────┬────────────┘
           │                           │                            │
           │              ┌────────────┴─────────────┐              │
           │              ▼                          │              │
           │  ┌──────────────────────────┐           │              │
           │  │ learning_field_          │           │              │
           │  │ integration.py           │           │              │
           │  │ (850 LOC)                │           │              │
           │  │                          │           │              │
           │  │ • GeometricPredictiveCod │           │              │
           │  │ • GeometricActiveInf     │           │              │
           │  │ • GeometricMetaHebbian   │           │              │
           │  │ • cognitive_cycle        │           │              │
           │  └────────────┬─────────────┘           │              │
           │               │                         │              │
           └───────────────┼─────────────────────────┼──────────────┘
                           │                         │
                           ▼                         ▼
              ┌─────────────────────────────────────────────────────┐
              │              PHASE 4: AUTONOMOUS LOOP               │
              │         loop_compositional_integration.py           │
              │                     (750 LOC)                       │
              │                                                     │
              │  • LoopState (PERCEIVE → REASON → ACT → LEARN)     │
              │  • autonomous_cycle()                               │
              │  • geodesic_hypothesis_generation                   │
              │  • metric_deformation_learning                      │
              │  • feedback_propagation                             │
              └───────────────────────┬─────────────────────────────┘
                                      │
                                      ▼
              ┌─────────────────────────────────────────────────────┐
              │              GEOMETRIC FOUNDATION                   │
              └─────────────────────────────────────────────────────┘
                           │                         │
                           ▼                         ▼
              ┌──────────────────────┐  ┌──────────────────────────┐
              │ vqvae_manifold_      │  │ geodesic_bridge_         │
              │ bridge.py            │  │ integration.py           │
              │ (877 LOC)            │  │ (669 LOC)                │
              │                      │  │                          │
              │ • VQVAEManifoldBridge│  │ • GeodesicBridgeInteg    │
              │ • embed()            │  │ • compute_geodesic()     │
              │ • from_vqvae_codes() │  │ • shortest_path()        │
              │ • compute_metric     │  │ • propagate_activation() │
              │ • get_nearest_anchor │  │ • find_attractor_basin() │
              └──────────────────────┘  └──────────────────────────┘
                           │                         │
                           └────────────┬────────────┘
                                        ▼
              ┌─────────────────────────────────────────────────────┐
              │             compositional_reasoning.py              │
              │                     (1090 LOC)                      │
              │                                                     │
              │  • CompositionalReasoner                            │
              │  • reason() - geodesic with residual accumulation   │
              │  • analogy() - a:b :: c:d via geodesic transform    │
              │  • Residual modes: HEBBIAN, ATTENTION, FIELD, HYBRID│
              └─────────────────────────────────────────────────────┘
```

## Integration Matrix

| From \ To | Bridge | Learning | Abduction | Agents | Loop | Compositional |
|-----------|--------|----------|-----------|--------|------|---------------|
| **Bridge** | - | ✓ metric | ✓ gaps | ✓ action | ✓ F | ✓ geodesic |
| **Learning** | ✓ PC | - | ✓ errors | ✓ beliefs | ✓ update | ✓ error |
| **Abduction** | ✓ detect | ✓ hyp | - | ✓ hyp | ✓ gaps | ✓ validate |
| **Agents** | ✓ embed | ✓ obs | ✓ critic | - | ✓ action | ✓ synthesize |
| **Loop** | ✓ state | ✓ learn | ✓ reason | ✓ act | - | ✓ reason |
| **Compositional** | ✓ path | ✓ trace | ✓ path | ✓ path | ✓ path | - |

## Data Flow

```
                    ┌─────────────────────────────────────────┐
                    │           EXTERNAL INPUT                │
                    │     (embedding 384D, query, goal)       │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │         VQVAEManifoldBridge             │
                    │   384D → 512D → ManifoldPoint           │
                    │   (with attractor gravity)              │
                    └───────────────────┬─────────────────────┘
                                        │
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
                    ▼                   ▼                   ▼
            ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
            │   PERCEIVE    │   │    REASON     │   │     ACT       │
            │               │   │               │   │               │
            │ PC encode     │   │ detect gaps   │   │ select action │
            │ compute F     │   │ gen hypotheses│   │ via EFE geom  │
            │ update belief │   │ validate path │   │ execute       │
            └───────┬───────┘   └───────┬───────┘   └───────┬───────┘
                    │                   │                   │
                    └───────────────────┼───────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │              LEARN                      │
                    │                                         │
                    │   • Update metric (deform manifold)     │
                    │   • Propagate feedback via geodesic     │
                    │   • Meta-Hebbian rule evolution         │
                    │   • Consolidate validated hypotheses    │
                    └───────────────────┬─────────────────────┘
                                        │
                                        ▼
                    ┌─────────────────────────────────────────┐
                    │           EXTERNAL OUTPUT               │
                    │   (action, synthesis, updated state)    │
                    └─────────────────────────────────────────┘
```

## Module Statistics

| Module | LOC | Classes | Key Methods |
|--------|-----|---------|-------------|
| alexandria_unified.py | 700 | 3 | cognitive_cycle, autonomous_run |
| vqvae_manifold_bridge.py | 877 | 6 | embed, from_vqvae_codes, compute_metric |
| compositional_reasoning.py | 1090 | 8 | reason, analogy, reason_chain |
| loop_compositional.py | 750 | 4 | autonomous_cycle, step, phases |
| learning_field.py | 850 | 5 | process_observation, plan_action |
| abduction_compositional.py | 730 | 4 | detect_gaps, generate_hypotheses |
| agents_compositional.py | 680 | 6 | full_pipeline, synthesize |
| nemesis_bridge.py | 590 | 4 | select_action_geometric |
| geodesic_bridge.py | 669 | 4 | compute_geodesic, shortest_path |
| **TOTAL** | **~6900** | **44** | - |

## Usage Example

```python
from alexandria_unified import AlexandriaCore

# Initialize from VQ-VAE model
core = AlexandriaCore.from_vqvae(vqvae_model)

# Health check
health = core.health_check()
# {'bridge': True, 'nemesis': True, 'learning': True, ...}

# Single cognitive cycle
result = core.cognitive_cycle(observation, goal)
# result.free_energy, result.gaps_detected, result.action_selected

# Autonomous loop
run_result = core.autonomous_run(observation, goal, max_iterations=100)
# run_result.trajectory, run_result.final_free_energy

# Direct access to subsystems
core.nemesis.select_action_geometric(...)
core.learning.process_observation(...)
core.abduction.detect_gaps_geometric(...)
core.agents.synthesize_geometric(...)
core.loop.step(...)
```

## Cosmic Garden Mapping

| Alexandria | Cosmic Garden | Function |
|------------|---------------|----------|
| Observation | Input q | External perturbation |
| ManifoldPoint | State S | Current position |
| Attractor | κ (checkpoint) | Stable concept |
| Free Energy F | Tension | System stress |
| Geodesic | T (transition) | Navigation |
| Residual | ∇F | Composition force |
| Metric g_ij | Field geometry | Curvature |
| Learning | BURN/REPLANT | Metric evolution |

## File Locations

All modules are in `/mnt/user-data/outputs/`:
- `alexandria_unified.py` - Master integration
- `vqvae_manifold_bridge.py` - Geometric foundation
- `compositional_reasoning.py` - Geodesic reasoning
- `geodesic_bridge_integration.py` - Path computation
- `nemesis_bridge_integration.py` - Active Inference + geometry
- `learning_field_integration.py` - PC + AI + Meta-Hebbian
- `abduction_compositional_integration.py` - Hypothesis generation
- `agents_compositional_integration.py` - Agent coordination
- `loop_compositional_integration.py` - Autonomous cycle

## Installation

```bash
# Copy to Alexandria project
cp /mnt/user-data/outputs/*.py /path/to/alexandria/core/integrations/

# In your code
from core.integrations.alexandria_unified import AlexandriaCore
```

## Next Steps

1. **Calibration**: Run calibration scripts to tune parameters
2. **Testing**: Test with real VQ-VAE and embeddings
3. **Mycelial**: Connect mycelial graph to Hebbian residuals
4. **Documentation**: Generate full API docs
5. **Performance**: Profile and optimize hot paths
