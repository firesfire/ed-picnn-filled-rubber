# Material models

| Directory | Implementation |
| --- | --- |
| `2_TeacherModel/` | Equilibrium-anchored scalar memory (T0), seven-branch Ogden models with shared (T1) or fast/slow (T2) flow parameters, joint identification and teacher trajectory generation. |
| `6_final_Model/` | ED-PICNN energy and dissipation potentials, stress integration, staged calibration and frozen-model evaluation. |
| `4_Exper/` | Stress-controlled creep and preloaded DMA at the measurement coordinates. |

From this directory, run `python 6_final_Model/code/evaluate_frozen.py --help` for relaxation, unpreloaded frequency sweeps or the 1 Hz amplitude sweep. Run `python 4_Exper/code/evaluate.py --help` for creep and preloaded sweeps. Both use the sibling `experimental_data/` directory by default.

Frozen T2-Ogden predictions require its fitted parameter JSON. ED-PICNN predictions also require a trained checkpoint. Supply these separately. Recalibration uses the loading histories and inputs for the corresponding training stages.
