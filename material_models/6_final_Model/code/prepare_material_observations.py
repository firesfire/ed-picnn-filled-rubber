"""Prepare the eight relaxation histories and reference frequency sweep for identification."""
import argparse
import csv
import json
from pathlib import Path


def prepare(data):
    histories = []
    for mode, name, target, strain, stress in [
        ('uniaxial', 'uniaxial_relaxation', 'target_engineering_strain', 'engineering_strain', 'P11_MPa'),
        ('shear', 'simple_shear_relaxation', 'target_shear_strain', 'shear_strain', 'P12_MPa'),
    ]:
        groups = {}
        with (data / 'stress_relaxation' / (name + '.csv')).open(encoding='utf-8-sig', newline='') as handle:
            for row in csv.DictReader(handle):
                if row['figure'] == '8':
                    groups.setdefault(float(row[target]), []).append(row)
        for level, rows in sorted(groups.items()):
            label = f'{100*level:g}%' if mode == 'uniaxial' else f'{level:.2f}'
            rows.sort(key=lambda row: float(row['time_from_loading_start_s']))
            histories.append(dict(key=f'{mode}:{label}', mode=mode, label=label,
                split='validation' if (mode, level) in [('uniaxial', .5), ('shear', .6)] else 'training',
                time_s=[float(row['time_from_loading_start_s']) for row in rows],
                strain=[float(row[strain]) for row in rows], nominal_stress_MPa=[float(row[stress]) for row in rows]))
    frequencies = []
    with (data / 'unpreloaded_dma/frequency_sweep.csv').open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            hz = float(row['frequency_Hz'])
            frequencies.append(dict(frequency_hz=hz, amplitude_fraction=.001,
                storage_MPa=float(row['storage_modulus_MPa']), loss_MPa=float(row['loss_modulus_MPa']),
                split='validation' if hz in (5., 7., 10., 15.) else 'training'))
    return dict(frequency_dma_amplitude_fraction=.001, amplitude_labels_loaded=False,
                relaxation=histories, frequency_dma=frequencies)


if __name__ == '__main__':
    model = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=model.parent.parent / 'experimental_data')
    parser.add_argument('--output', type=Path, default=model / 'audit/stage3/stage3_real_training_data.json')
    args = parser.parse_args()
    result = prepare(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False), encoding='utf-8')
