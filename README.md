# Coupling nonequilibrium energy and viscous flow in a finite-strain model for filled rubber

This repository accompanies the study by Zaicheng Zhang, Ming Luo, Shuo Gu and Xiaoang Liu. The study examines whether one set of material parameters can describe finite-strain stress relaxation and small-amplitude dynamic moduli in carbon-black-filled natural rubber, and how those parameters transfer to a rubber bushing.

| Directory | Contents |
| --- | --- |
| [experimental_data](experimental_data/) | Eleven CSV files covering multiaxial tension, stress relaxation, creep, dynamic mechanical analysis (DMA) and bushing tests. The tables give the experimental coordinates and quantities used in the paper's figures. |
| [material_models](material_models/README.md) | The T0, T1–Ogden and T2–Ogden models, T2 teacher-trajectory generation, and the ED-PICNN constitutive, calibration and material-point evaluation code. |
| [finite_element](finite_element/README.md) | Abaqus/Standard user material (UMAT), four preload and creep analyses, and forty dynamic-stiffness inputs covering ten frequencies per preload. |

## Getting started

The CSV files have English column headings and can be read directly. For example, [the unpreloaded frequency sweep](experimental_data/unpreloaded_dma/frequency_sweep.csv) contains frequency in Hz and storage and loss moduli in MPa.

For the Python material models, use Python 3.11 and install the dependencies from the repository root.

```sh
python -m pip install -r requirements.txt
python material_models/6_final_Model/code/evaluate_frozen.py --help
```

The release contains experimental tables, source code, configurations and finite-element inputs. To evaluate the published frozen ED-PICNN, supply its trained checkpoint and the T2–Ogden material parameter file separately. Full recalibration also requires the loading histories and stage inputs described in the [material-model guide](material_models/README.md). Running the bushing analyses requires Abaqus/Standard and a compatible Fortran compiler.
