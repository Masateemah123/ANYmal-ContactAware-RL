# Cross-Morphology Contact-Aware Quadruped Locomotion in Isaac Lab

Proprioceptive reinforcement learning locomotion framework developed in **NVIDIA Isaac Lab** that integrates a frozen, IMU-free recurrent contact estimator ([**ContactNet**](https://github.com/Masateemah123/ContactNet)) into the PPO observation pipeline for cross-platform quadruped locomotion (**Unitree Go1**, **ANYmal C**, and **ANYmal D**).

---

## 1. System Architecture: Frozen Belief-State Perception + PPO Control

Rather than relying on physical foot-contact switches or drift-prone base IMU measurements, the policy runs **ContactNet** online inside the Isaac Lab environment loop:

1. **48D Leg-Local Proprioceptive Frame (`mdp/observations.py`):**
   Constructs a 48-dimensional proprioceptive vector per step comprising joint positions ($q \in \mathbb{R}^{12}$), joint velocities ($\dot{q} \in \mathbb{R}^{12}$), base-relative foot positions ($p_f^b \in \mathbb{R}^{12}$), and base-frame foot linear velocities ($v_f^b \in \mathbb{R}^{12}$).
2. **150-Step Rolling Buffer & Bayesian Contact Belief (`mdp/contact_estimator_event.py` & `contact_estimator_wrapper.py`):**
   Maintains a per-environment rolling history $(N_{\text{envs}}, 150, 48)$ and evaluates the frozen 1D-CNN + GRU estimator to obtain a 16-state discrete posterior belief $b_t \in \Delta^{15}$ over all $2^4$ foot contact configurations, projected via $b_t M$ into continuous per-leg contact probabilities $\hat{\mathbf{c}}_t \in [0, 1]^4$ (`[LF, RF, LH, RH]`) and Shannon belief entropy $H(b_t)$.
3. **Multi-Robot Policy Configurations (`config/go_1`, `config/anymal_c`, `config/anymal_d`):**
   Exposes `estimated_contacts` (`(N, 4)`) to the PPO actor across flat and rough terrain curricula, enabling contact-aware locomotion across distinct quadruped morphologies without physical contact sensors.

---

## 2. Repository Structure

```text
ANYmal-ContactAware-RL/
├── README.md
└── anymal_c_contact_estimator/
    ├── __init__.py                        # Isaac Lab Gym environment registration
    ├── velocity_env_cfg.py                # Base ManagerBasedRLEnvCfg with proprioceptive & contact-belief obs
    ├── contact_estimator_wrapper.py       # Frozen ContactNet inference & 16-class marginal belief projection
    ├── play.py                            # Policy evaluation & rollout script
    ├── config/
    │   ├── scene.py                       # Multi-terrain simulation scene configuration
    │   ├── go_1/                          # Unitree Go1 flat & rough terrain PPO configs
    │   ├── anymal_c/                      # ANYmal C flat & rough terrain PPO configs
    │   └── anymal_d/                      # ANYmal D flat & rough terrain PPO configs
    └── mdp/
        ├── observations.py                # 48D proprioceptive builder + estimated_contacts observation term
        ├── contact_estimator_event.py     # Startup init, 150-step rolling buffer update & env reset hooks
        ├── rewards.py                     # Velocity tracking & gait-regularization reward terms
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
    --task Isaac-Velocity-Rough-Anymal-C-ContactEst-v0 --num_envs 32
```

---

## 4. Perception Backbone

- **[Masateemah123/ContactNet](https://github.com/Masateemah123/ContactNet):** 48D IMU-free Dilated 1D-CNN + GRU contact estimator trained on Unitree Go1 and validated zero-shot across outdoor terrains and Unitree Go2.
