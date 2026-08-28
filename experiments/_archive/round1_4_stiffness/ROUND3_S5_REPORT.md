# Round 3 (S5) Quantitative Results

This report compiles the raw quantitative numbers from the S5 validation suite. 
All results are cited to their corresponding runs in the `results/` directory.

## S5.1: Non-Stationary Drift (Rotating MNIST)
**Source:** `results/S5_1_rot_*`

- **λ=0.0**: 
  - Accuracy (Next-Task): 0.9164 ± 0.0012
  - Retention (Prev-Tasks): 0.4765 ± 0.0047
- **λ=1.0**: 
  - Accuracy (Next-Task): 0.9193 ± 0.0013
  - Retention (Prev-Tasks): 0.4245 ± 0.0070
- **λ=10.0**: 
  - Accuracy (Next-Task): 0.9158 ± 0.0004
  - Retention (Prev-Tasks): 0.4055 ± 0.0071

## S5.2: Vision Domain (Split CIFAR-100 on ConvNets)
**Source:** `results/S5_2_*_seed*`

- **bp** (Backprop Baseline): 
  - Accuracy (Next-Task): 0.0772 ± 0.0122
  - Retention (Prev-Tasks): 0.0119 ± 0.0009
- **ln_l2** (LayerNorm + L2): 
  - Accuracy (Next-Task): 0.0602 ± 0.0131
  - Retention (Prev-Tasks): 0.0118 ± 0.0009
- **rs** (Readiness Sink): 
  - Accuracy (Next-Task): 0.0610 ± 0.0080
  - Retention (Prev-Tasks): 0.0121 ± 0.0014
- **cbp** (Continual Backprop): 
  - *Failed execution.* Encountered `AttributeError: 'Conv2d' object has no attribute 'out_features'` because the CBP implementation in `src/methods/cbp.py` expects MLPs (nn.Linear) and is incompatible with ConvNets. (See `results/S5_2_cbp_seed1.log`)

*(Note: Random guessing for a 10-class split is 0.10. The 1.7M parameter ConvNet baseline accuracy on Split CIFAR-100 at 10 epochs performs worse than random guessing on the current task and has a retention of 1%, demonstrating catastrophic failure to learn/remember the splits under these constraints).*

## S5.3: Activation Functions (LeakyReLU vs ReLU)
**Source:** `results/S5_3_*`

- **LeakyReLU, λ=0.0**: 
  - Accuracy (Next-Task): 0.7689 ± 0.0023
  - Retention (Prev-Tasks): 0.3363 ± 0.0068
- **LeakyReLU, λ=1.0**: 
  - Accuracy (Next-Task): 0.7789 ± 0.0010
  - Retention (Prev-Tasks): 0.4163 ± 0.0076

- **ReLU, λ=0.0**: 
  - Accuracy (Next-Task): 0.7682 ± 0.0025
  - Retention (Prev-Tasks): 0.3369 ± 0.0068
- **ReLU, λ=1.0**: 
  - Accuracy (Next-Task): 0.7782 ± 0.0008
  - Retention (Prev-Tasks): 0.4177 ± 0.0076
