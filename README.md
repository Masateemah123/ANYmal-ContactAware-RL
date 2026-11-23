# Cross-Morphology Contact-Aware Quadruped Locomotion in Isaac Lab

> **Status:** Codebase and simulation benchmarks from Master's thesis (*Cross-Morphology Proprioceptive Contact Estimation and Contact-Aware Locomotion*). Manuscript in preparation.

![System Architecture Overview](figures/architecture_overview.jpg)

---

## 1. Closed-Loop Perception & Control Pipeline

Rather than relying on physical foot-contact switches or drift-prone base IMU measurements, this framework embeds a frozen, IMU-free recurrent contact estimator ([**ContactNet**](https://github.com/Masateemah123/ContactNet)) directly inside the **NVIDIA Isaac Lab** reinforcement learning loop:

1. **Frozen ContactNet (`PCNN-GRU`):**
   - Consumes a rolling $150$-step ($0.15\text{ s}$) history window $X_t \in \mathbb{R}^{150 \times 48}$ of IMU-free leg-local proprioception: joint angles $q \in \mathbb{R}^{12}$, joint velocities $\dot{q} \in \mathbb{R}^{12}$, base-relative foot positions $p_f \in \mathbb{R}^{12}$, and base-frame foot velocities $v_f \in \mathbb{R}^{12}$.
   - Outputs $16$-class logits $z \in \mathbb{R}^{16}$ and posterior configuration probabilities $P(S = s_i \mid X_t) \in \mathbb{R}^{16}$.
2. **Marginalization & Predictive Entropy:**
   - **Contact Belief Vector (Observation):** Marginalizes the $16$-state posterior into per-foot soft contact probabilities $b_f(t) = P(C_f = 1 \mid X_t) \in [0, 1]^4$ (`[LF, RF, LH, RH]`), appended to the PPO policy observation vector $o(t)$.
   - **Predictive Entropy (Reward Regularization):** Computes Shannon entropy $\mathcal{H}(t) = -\sum_{i=1}^{16} p_i \log p_i \in [0, \log 16]$ as an epistemic/aleatoric uncertainty signal used **in the reward function only** (uncertainty penalty) to discourage gait states where contact observability degrades.
3. **PPO Locomotion Policy & Cross-Morphology Actuation:**
   - Concatenates $b(t) \in \mathbb{R}^4$ with base linear/angular velocities, projected gravity, joint states $(q, \dot{q})$, previous actions, and velocity commands to output $12$-DoF joint target actions $a(t) \in \mathbb{R}^{12}$ across **Unitree Go1**, **ANYmal C**, and **ANYmal D**.

---

## 2. Repository Structure

```text
ANYmal-ContactAware-RL/
├── README.md
├── figures/
│   └── architecture_overview.jpg          # Closed-loop perception + PPO control diagram
└── anymal_c_contact_estimator/
    ├── __init__.py                        # Isaac Lab Gym task registration
    ├── velocity_env_cfg.py                # Base ManagerBasedRLEnvCfg with proprioceptive & contact-belief obs
    ├── contact_estimator_wrapper.py       # Frozen ContactNet inference, marginalization & Shannon entropy
    ├── play.py                            # Policy evaluation & rollout script
    ├── config/
    │   ├── scene.py                       # Multi-terrain simulation scene configuration
    │   ├── go_1/                          # Unitree Go1 flat & rough terrain PPO configs
    │   ├── anymal_c/                      # ANYmal C flat & rough terrain PPO configs
    │   └── anymal_d/                      # ANYmal D flat & rough terrain PPO configs
    └── mdp/
        ├── observations.py                # 48D proprioceptive builder + estimated_contacts observation term
        ├── contact_estimator_event.py     # Startup init, 150-step rolling buffer update & env reset hooks
        ├── rewards.py                     # Velocity tracking, gait rewards & entropy uncertainty penalty
        ├── terminations.py                # Base/link termination conditions
        ├── curriculums.py                 # Terrain difficulty progression
        ├── metrics.py                     # Contact estimation & locomotion tracking metrics
        └── symmetry/anymal.py             # Morphological symmetry augmentation
```

---

## 3. Running Policy Evaluation (`play.py`)

Point `CONTACTNET_SRC` and `CONTACTNET_CKPT` to your local [**ContactNet**](https://github.com/Masateemah123/ContactNet) directory and run rollout evaluation in Isaac Lab:

```bash
export CONTACTNET_SRC=/path/to/ContactNet/src
export CONTACTNET_CKPT=/path/to/ContactNet/checkpoints/contactnet_48d_epoch19.ckpt

python anymal_c_contact_estimator/play.py \
    --task Isaac-Velocity-Rough-Anymal-D-ContactEst-v0 --num_envs 32
```

---

## 4. Related Repository

- **[Masateemah123/ContactNet](https://github.com/Masateemah123/ContactNet):** 48D IMU-free Dilated 1D-CNN + GRU contact estimator trained on Unitree Go1 and validated zero-shot across outdoor terrains and Unitree Go2.
