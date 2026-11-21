# Rubber bushing finite-element analysis

Run the examples below from this directory with Abaqus/Standard and a compatible Fortran compiler. Supply the trained ED-PICNN checkpoint and T2 parameter file to assemble the material subroutine. If you already have the exported parameter arrays, use `python build_umat.py frozen_weights.npz material.f90` for the first command.

```text
python build_umat.py edpicnn_stage8_final.pt material.f90 --teacher T2_chapter2_joint_evaluation_parameters.json
abaqus job=full_creepf_2000 input=inp/full_creepf_2000.inp user=material.f90 cpus=10 interactive
abaqus job=dyn2000a0p25_edpicnn_5hz oldjob=full_creepf_2000 input=inp/dyn2000a0p25_edpicnn_5hz.inp user=material.f90 cpus=10 interactive
abaqus python extract_response.py full_creepf_2000.odb creep.csv
abaqus python extract_response.py dyn2000a0p25_edpicnn_5hz.odb stiffness.csv
```

Each frequency restarts independently from the end of the corresponding preload hold. The example uses 2000 N and 5 Hz. If you rerun the parent job, check that its final hold increment and mean displacement match the settings in the restart deck.

The dynamic decks apply five cycles with a peak displacement amplitude of 0.25 mm. The extraction script uses the force and displacement harmonics of the complete fifth cycle to calculate storage and loss stiffness.
