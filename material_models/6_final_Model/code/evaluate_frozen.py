"""Run the frozen ED-PICNN on relaxation, reference frequency, or 1 Hz amplitude sweeps."""
import argparse
from pathlib import Path

import pandas as pd
import torch

from dual_potential.solvers import MaterialPointSolver
from frozen_material import load_frozen
from prepare_material_observations import prepare
from run_stage3_final_validation import curve_rows
from stage3_real import DMACondition, _downsample_curve
from stage5b_energy import dma_rollout


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=['relaxation', 'frequency', 'amplitude'])
    parser.add_argument('--data', type=Path, default=Path(__file__).resolve().parents[3] / 'experimental_data')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--teacher-parameters', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    material = load_frozen(args.checkpoint, args.teacher_parameters)
    torch.set_num_threads(1)
    if args.task == 'relaxation':
        rows = prepare(args.data)['relaxation']
        conditions = [_downsample_curve(row, material.equilibrium_energy_parameters, 6, 18, material.dtype,
                                        material.device) for row in rows]
        output = pd.DataFrame(curve_rows(material, conditions, 'ED-PICNN', 'full_tensor_local_implicit')[0])
    elif args.task == 'frequency':
        output = pd.read_csv(args.data / 'unpreloaded_dma/frequency_sweep.csv')
        hz = torch.tensor(output.frequency_Hz.to_numpy(), dtype=material.dtype)
        storage, loss = MaterialPointSolver(material).local_linear_shear_moduli(hz)
        output['predicted_storage_MPa'] = storage.detach().numpy()
        output['predicted_loss_MPa'] = loss.detach().numpy()
    else:
        output = pd.read_csv(args.data / 'unpreloaded_dma/amplitude_sweep_1Hz.csv')
        cases = [DMACondition(1., pct / 100., float('nan'), float('nan'), 'prediction')
                 for pct in output.strain_amplitude_percent]
        response = dma_rollout(material, cases, cycles=5, points_per_cycle=48, differentiable=False)
        output['predicted_storage_MPa'] = response['storage'].detach().numpy()
        output['predicted_loss_MPa'] = response['loss'].detach().numpy()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)


if __name__ == '__main__':
    main()
