# Pass 5 — Multi-seed verdict (mean, SD, sign rule vs plain CoLaR)

| arm | scenario | backbone | n | mean AA | SD | per-seed AA | Δ vs plain | sign-agree 3/3 |
|---|---|---|---|---|---|---|---|---|
| colar | cil_cord | layoutlmv3 | 4 | 20.06 | 1.10 | s42=19.1, s123=20.5, s42=21.7, s7=19.0 | +0.0, +0.0, +2.6, +0.0 | no |
| colar | cil_imagenet_r | vit | 4 | 55.55 | 1.13 | s123=55.2, s42=53.8, s7=56.7, s42=56.4 | +0.0, +0.0, +0.0, +2.6 | no |
| colar | dil | layoutlmv3 | 7 | 75.57 | 8.08 | s42=87.1, s123=69.6, s123=69.6, s42=71.9, s42=86.4, s7=65.1, s7=79.2 | +0.0, +0.0, -0.0, -15.2, -0.7, +0.0, +14.1 | no |
| colar | mixed | layoutlmv3 | 4 | 58.76 | 0.78 | s42=58.8, s123=59.2, s42=57.5, s7=59.6 | +0.0, +0.0, -1.3, +0.0 | no |
| colar_adaptive | dil | layoutlmv3 | 3 | 75.17 | 3.21 | s123=75.3, s42=79.0, s7=71.2 | +5.6, -8.1, +6.1 | no |
| colar_e10 | dil | layoutlmv3 | 3 | 87.75 | 0.39 | s123=87.4, s42=87.6, s7=88.3 | +17.7, +0.5, +23.2 | yes |
| colar_r_oracle | cil_cord | layoutlmv3 | 3 | 26.30 | 2.93 | s123=29.6, s42=26.8, s7=22.5 | +9.1, +7.7, +3.5 | yes |
| colar_r_oracle | mixed | layoutlmv3 | 3 | 64.36 | 1.55 | s123=63.0, s42=63.5, s7=66.5 | +3.8, +4.8, +7.0 | yes |
| colar_r_pseudo | cil_cord | layoutlmv3 | 3 | 19.98 | 3.37 | s123=20.6, s42=23.7, s7=15.6 | +0.2, +4.7, -3.4 | no |
| colar_r_pseudo | mixed | layoutlmv3 | 3 | 59.51 | 0.04 | s123=59.5, s42=59.5, s7=59.6 | +0.3, +0.7, +0.0 | no |
| colar_r_sliced | cil_cord | layoutlmv3 | 3 | 25.38 | 1.95 | s123=25.5, s42=27.7, s7=22.9 | +5.1, +8.6, +3.9 | yes |
| colar_r_sliced | mixed | layoutlmv3 | 3 | 61.30 | 1.56 | s123=59.1, s42=62.0, s7=62.7 | -0.1, +3.2, +3.2 | no |
| colar_r_sliced_pseudo | cil_cord | layoutlmv3 | 3 | 20.68 | 0.81 | s123=21.2, s42=19.5, s7=21.3 | +0.8, +0.4, +2.3 | yes |
| colar_r_sliced_pseudo | mixed | layoutlmv3 | 3 | 60.09 | 0.85 | s123=60.9, s42=60.5, s7=58.9 | +1.7, +1.7, -0.6 | no |
| docB1 | dil | layoutlmv3 | 1 | 74.56 | 0.00 | s42=74.6 | -12.5 | n/a |
| docB2b | dil | layoutlmv3 | 1 | 86.22 | 0.00 | s42=86.2 | -0.9 | n/a |
| docD1 | dil | layoutlmv3 | 1 | 79.57 | 0.00 | s42=79.6 | -7.5 | n/a |
| docD2 | dil | layoutlmv3 | 1 | 61.20 | 0.00 | s42=61.2 | -25.9 | n/a |
| docRfar | dil | layoutlmv3 | 4 | 84.84 | 4.10 | s123=86.3, s42=88.2, s7=87.0, s42=77.8 | +16.7, +1.1, +22.0, -9.3 | no |
| docRnear | dil | layoutlmv3 | 2 | 75.27 | 11.26 | s42=86.5, s42=64.0 | -0.6, -23.1 | n/a |
| docRtask | dil | layoutlmv3 | 2 | 80.34 | 6.99 | s42=87.3, s42=73.4 | +0.2, -13.7 | n/a |
| docU | dil | layoutlmv3 | 3 | 83.15 | 5.09 | s123=86.9, s7=86.6, s42=75.9 | +17.2, +21.5, -11.2 | no |
| lexslot | cil_cord | bert | 3 | 18.86 | 0.20 | s123=18.9, s42=19.1, s7=18.6 |  |  |
| lexslot | cil_cord | layoutlmv3 | 1 | 19.18 | 0.00 | s42=19.2 | +0.1 | n/a |
| lexslot | dil | bert | 3 | 37.96 | 0.37 | s123=37.6, s42=38.5, s7=37.9 |  |  |
| lexslot | dil | layoutlmv3 | 7 | 80.39 | 16.31 | s123=86.3, s42=40.5, s42=87.1, s42=88.4, s42=88.4, s42=84.8, s7=87.2 | +16.7, -46.6, -0.0, +1.3, +1.3, -2.3, +22.2 | no |
| lexslot | mixed | bert | 3 | 26.70 | 1.49 | s123=24.7, s42=27.0, s7=28.3 |  |  |
| lexslot | mixed | layoutlmv3 | 1 | 33.13 | 0.00 | s42=33.1 | -25.7 | n/a |
