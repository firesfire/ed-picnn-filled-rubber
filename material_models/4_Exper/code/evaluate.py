"""Evaluate material creep and preloaded DMA at the released experimental coordinates."""
import argparse
from contextlib import nullcontext
from pathlib import Path
import sys

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT / '6_final_Model/code'))
from frozen_material import load_frozen
from observations import batch_periodic_dma, predict_creep
from stage8_common import build_t2
from t2_paths import t2_interface


def load_observations(data):
    creep = []
    for mode, filename in [('uniaxial', 'uniaxial_creep.csv'), ('shear', 'simple_shear_creep.csv')]:
        frame = pd.read_csv(data / 'material_creep' / filename).rename(columns={
            'nominal_stress_MPa': 'target_stress_MPa', 'compliance_per_MPa': 'experimental_compliance_per_MPa'})
        frame['mode'] = mode
        frame['experimental_strain'] = frame.target_stress_MPa * frame.experimental_compliance_per_MPa
        creep.append(frame)
    frames = []
    for field, filename in [('frequency', 'frequency_sweeps.csv'), ('amplitude', 'amplitude_sweeps_1Hz.csv')]:
        frame = pd.read_csv(data / 'preloaded_dma' / filename).rename(columns={'preload_shear_strain': 'shear_preload'})
        if field == 'frequency':
            frame['strain_amplitude_fraction'] = 0.001
        else:
            frame['frequency_Hz'] = 1.0
            frame['strain_amplitude_fraction'] = frame.strain_amplitude_percent / 100.0
        frame['sweep'] = field
        frames.append(frame)
    return pd.concat(creep, ignore_index=True), pd.concat(frames, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['ED-PICNN', 'T2-Ogden'], default='ED-PICNN')
    parser.add_argument('--data', type=Path, default=PROJECT.parent / 'experimental_data')
    parser.add_argument('--teacher-parameters', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.model == 'ED-PICNN':
        if args.checkpoint is None:
            parser.error('--checkpoint is required for the frozen ED-PICNN')
        material = load_frozen(args.checkpoint, args.teacher_parameters)
        context = nullcontext()
    else:
        material, _ = build_t2(args.teacher_parameters)
        potential = material.dissipation_potential
        potential.register_buffer('sigma_ref', potential.equilibrium_shear.clone())
        context = t2_interface()
    creep, dma = load_observations(args.data)
    args.output.mkdir(parents=True, exist_ok=True)
    coords = ['shear_preload', 'frequency_Hz', 'strain_amplitude_fraction']
    with context:
        predict_creep(material, creep, {'creep': dict(loading_time_s=1.0, loading_points=21,
                     stress_tolerance_MPa=1.e-5, maximum_outer_iterations=40)}, args.output)
        predictions = batch_periodic_dma(material, dma[coords].drop_duplicates(),
            dict(points_per_cycle=48, maximum_cycles=5, minimum_cycles=3, steady_tolerance=2.e-3))
    predictions['model'] = args.model
    predictions = predictions[coords + ['model', 'storage_modulus_MPa', 'loss_modulus_MPa',
                                        'cycle_change', 'steady']].rename(columns={
            'storage_modulus_MPa': 'predicted_storage_MPa', 'loss_modulus_MPa': 'predicted_loss_MPa'})
    dma.merge(predictions, on=coords, how='left').to_csv(args.output / 'preloaded_dma.csv', index=False)


if __name__ == '__main__':
    main()
