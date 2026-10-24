# Real-world measurements

Recorded 2026-10-20. These are for the sim environments (`envs/*.json`) and system identification
(`video/`, `xnew_rrt/fit_to_master.py`). The vine pressure and the box mass may change.

| What | Value as given | Value used in the sim | Notes |
|---|---|---|---|
| Box | 9.8 mm x 9.8 mm, 503 g | 98 x 98 mm, 0.503 kg | Almost certainly cm. A 9.8 mm box cannot weigh 503 g. |
| Brick | 20 mm x 9.5 mm, 2800 g | 200 x 95 mm, 2.8 kg | Also cm. The video tracking measured its top face at about 203 x 92 mm (`video/trials/make_trials.py`, which assumed 2.3 kg). |
| Vine target pressure | about 1 psi | (6.9 kPa) | Subject to change. |
| Vine radius | 76.2 / pi mm, about 24.2552 mm | 24.26 mm | That is a 152.4 mm (6 in) circumference. Video tracking gave about 25 mm. |

## Consequences for the video fit

The fit used a 2.3 kg brick. Only mass x friction mattered there, so at 2.8 kg the fitted box
friction scales down by 2.3 / 2.8:

- 0.097 becomes 0.080 (20 N growth force);
- 0.115 becomes 0.094 (40 N growth force).

The sim environments use 0.08 for both the boxes and the brick, which sit on the same pegboard.
