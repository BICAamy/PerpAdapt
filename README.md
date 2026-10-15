# PerpAdapt

Official implementation of **PerpAdapt: Radiation-free Perpendicular Pivot
Calibration for Sensor-Attached C-arms with Memory Sample Consensus**.

This revision contains corrected geometric implementations of:

- PPC: one C-arm-axis circle plus one x-axis circle;
- Lee PC(4): three C-arm-axis circles plus one x-axis circle;
- Lee PC(6): three C-arm-axis circles plus three x-axis circles.

The corrected code fits a 3-D circle as a plane plus a 2-D circle, represents
each rotation axis as an offset line, and does not assume that an axis passes
through the world-coordinate origin.

## Reproduce the corrected comparison

```bash
python -m pip install -r requirements.txt
python simulation.py --output outputs/clean_seed20261015.npz --instances 1000 --samples-per-circle 100 --seed 20261015 --noise-std 0
python evaluate_corrected.py --dataset outputs/clean_seed20261015.npz
```

For the complete clean/noisy commands, data schema, metric definitions, and
reference results, see [docs/reproducibility.md](docs/reproducibility.md).

Run the regression tests with:

```bash
python -m unittest discover -s tests -v
```
