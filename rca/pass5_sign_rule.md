| arm | scen | bb | d/task | r | n | mean AA | SD | per-seed | Δ vs plain(same recipe) | 3/3 sign |
|---|---|---|---|---|---|---|---|---|---|---|
| colar | cil_cord | layoutlmv3 | 50 | 128 | 3 | 20.38 | 1.09 | s123=20.5, s42=21.7, s7=19.0 |  |  |
| colar | cil_imagenet_r | vit | 500 | 128 | 3 | 55.26 | 1.17 | s123=55.2, s42=53.8, s7=56.7 |  |  |
| colar | dil | layoutlmv3 | 10 | 64 | 3 | 73.59 | 4.08 | s123=69.6, s42=71.9, s7=79.2 |  |  |
| colar | dil | layoutlmv3 | 5 | 64 | 2 | 67.36 | 2.28 | s123=69.6, s7=65.1 |  |  |
| colar | dil | layoutlmv3 | 50 | 128 | 1 | 86.38 | 0.00 | s42=86.4 |  |  |
| colar | mixed | layoutlmv3 | 50 | 128 | 3 | 58.75 | 0.90 | s123=59.2, s42=57.5, s7=59.6 |  |  |
| colar_adaptive | dil | layoutlmv3 | 5 | 64 | 3 | 75.17 | 3.21 | s123=75.3, s42=79.0, s7=71.2 | s123:+5.6, s7:+6.1 | n/a (n=2) |
| colar_e10 | dil | layoutlmv3 | 50 | 128 | 3 | 87.75 | 0.39 | s123=87.4, s42=87.6, s7=88.3 | s42:+1.2 | n/a (n=1) |
| colar_r_oracle | cil_cord | layoutlmv3 | 50 | 128 | 3 | 26.30 | 2.93 | s123=29.6, s42=26.8, s7=22.5 | s123:+9.1, s42:+5.1, s7:+3.5 | yes |
| colar_r_oracle | mixed | layoutlmv3 | 50 | 128 | 3 | 64.36 | 1.55 | s123=63.0, s42=63.5, s7=66.5 | s123:+3.8, s42:+6.1, s7:+7.0 | yes |
| colar_r_pseudo | cil_cord | layoutlmv3 | 50 | 128 | 3 | 19.98 | 3.37 | s123=20.6, s42=23.7, s7=15.6 | s123:+0.2, s42:+2.1, s7:-3.4 | no |
| colar_r_pseudo | mixed | layoutlmv3 | 50 | 128 | 3 | 59.51 | 0.04 | s123=59.5, s42=59.5, s7=59.6 | s123:+0.3, s42:+2.0, s7:+0.0 | no |
| colar_r_sliced | cil_cord | layoutlmv3 | 50 | 128 | 3 | 25.38 | 1.95 | s123=25.5, s42=27.7, s7=22.9 | s123:+5.1, s42:+6.0, s7:+3.9 | yes |
| colar_r_sliced | mixed | layoutlmv3 | 50 | 128 | 3 | 61.30 | 1.56 | s123=59.1, s42=62.0, s7=62.7 | s123:-0.1, s42:+4.5, s7:+3.2 | no |
| colar_r_sliced_pseudo | cil_cord | layoutlmv3 | 50 | 128 | 3 | 20.68 | 0.81 | s123=21.2, s42=19.5, s7=21.3 | s123:+0.8, s42:-2.1, s7:+2.3 | no |
| colar_r_sliced_pseudo | mixed | layoutlmv3 | 50 | 128 | 3 | 60.09 | 0.85 | s123=60.9, s42=60.5, s7=58.9 | s123:+1.7, s42:+3.0, s7:-0.6 | no |
| docB1 | dil | layoutlmv3 | 5 | 128 | 1 | 74.56 | 0.00 | s42=74.6 |  | no plain match |
| docB2b | dil | layoutlmv3 | 50 | 128 | 1 | 86.22 | 0.00 | s42=86.2 | s42:-0.2 | n/a (n=1) |
| docD1 | dil | layoutlmv3 | 50 | 128 | 1 | 79.57 | 0.00 | s42=79.6 | s42:-6.8 | n/a (n=1) |
| docD2 | dil | layoutlmv3 | 50 | 128 | 1 | 61.20 | 0.00 | s42=61.2 | s42:-25.2 | n/a (n=1) |
| docRfar | dil | layoutlmv3 | 5 | 128 | 1 | 77.84 | 0.00 | s42=77.8 |  | no plain match |
| docRfar | dil | layoutlmv3 | 50 | 128 | 3 | 87.18 | 0.78 | s123=86.3, s42=88.2, s7=87.0 | s42:+1.8 | n/a (n=1) |
| docRnear | dil | layoutlmv3 | 5 | 128 | 1 | 64.02 | 0.00 | s42=64.0 |  | no plain match |
| docRnear | dil | layoutlmv3 | 50 | 128 | 1 | 86.53 | 0.00 | s42=86.5 | s42:+0.1 | n/a (n=1) |
| docRtask | dil | layoutlmv3 | 5 | 128 | 1 | 73.36 | 0.00 | s42=73.4 |  | no plain match |
| docRtask | dil | layoutlmv3 | 50 | 128 | 1 | 87.33 | 0.00 | s42=87.3 | s42:+0.9 | n/a (n=1) |
| docU | dil | layoutlmv3 | 5 | 128 | 1 | 75.94 | 0.00 | s42=75.9 |  | no plain match |
| docU | dil | layoutlmv3 | 50 | 128 | 2 | 86.75 | 0.13 | s123=86.9, s7=86.6 |  | no plain match |
| lexslot | cil_cord | layoutlmv3 | - | - | 3 | 18.86 | 0.20 | s123=18.9, s42=19.1, s7=18.6 |  | no plain match |
| lexslot | dil | layoutlmv3 | - | - | 9 | 70.68 | 23.16 | s123=37.6, s123=86.3, s42=38.5, s42=87.1, s42=88.4, s42=88.4, s42=84.8, s7=37.9, s7=87.2 |  | no plain match |
| lexslot | mixed | layoutlmv3 | - | - | 3 | 26.70 | 1.49 | s123=24.7, s42=27.0, s7=28.3 |  | no plain match |
