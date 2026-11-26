# Coupling nonequilibrium energy and viscous flow in a finite-strain model for filled rubber

This repository accompanies the study by Zaicheng Zhang, Ming Luo, Shuo Gu and Xiaoang Liu. The study examines whether one set of material parameters can describe finite-strain stress relaxation and small-amplitude dynamic moduli in carbon-black-filled natural rubber, and how those parameters transfer to a rubber bushing.

| Directory | Contents |
| --- | --- |
| [experimental_data](experimental_data/) | Multiaxial tension, stress relaxation, creep, dynamic mechanical analysis (DMA) and bushing tests. |
| [material_models](material_models/README.md) | The T0, T1–Ogden and T2–Ogden models, T2 teacher-trajectory generation, and the ED-PICNN constitutive, calibration and material-point evaluation code. |
| [finite_element](finite_element/README.md) | Abaqus/Standard user material (UMAT), four preload and creep analyses, and forty dynamic-stiffness inputs covering ten frequencies per preload. |

## Getting started

The CSV files have English column headings and can be read directly. For example, [the unpreloaded frequency sweep](experimental_data/unpreloaded_dma/frequency_sweep.csv) contains frequency in Hz and storage and loss moduli in MPa.

In the [relaxation tables](experimental_data/stress_relaxation/), `loading_ramp` denotes stresses calculated with the Zhang–Li model along a prescribed strain ramp of 0.2 s⁻¹, sampled every 0.05 s and matched to the measured stress at hold onset. The hold-stage curves use the experimental data.

The [raw creep table](experimental_data/material_creep/material_creep_raw_observations.csv) contains all 217 readings used to evaluate total compliance. The [uniaxial](experimental_data/material_creep/uniaxial_creep.csv) and [simple-shear](experimental_data/material_creep/simple_shear_creep.csv) curves average readings taken at the same time within a condition, giving 215 distinct condition–time pairs for the baseline and compliance increments.

For the Python material models, use Python 3.11 and install the dependencies from the repository root.

```sh
python -m pip install -r requirements.txt
python material_models/6_final_Model/code/evaluate_frozen.py --help
```

The release contains experimental tables, source code, configurations and finite-element inputs. To evaluate the published frozen ED-PICNN, supply its trained checkpoint and the T2–Ogden material parameter file separately. Full recalibration also requires the loading histories and stage inputs described in the [material-model guide](material_models/README.md). Running the bushing analyses requires Abaqus/Standard and a compatible Fortran compiler.
