# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

"""
PMETS
Proprioceptive Marginal Error Terrain Score

This module contains analysis metrics only.

Nothing here influences PPO training directly.
These functions are intended for:
    • TensorBoard logging
    • Paper figures
    • Thesis evaluation
"""

from __future__ import annotations

import torch

from isaaclab.sensors import ContactSensor


# --------------------------------------------------------
# Ground-truth contacts
# --------------------------------------------------------

def ground_truth_contacts(env):
    """
    Returns
    -------
    (N,4)
    binary contacts
    """

    sensor: ContactSensor = env.scene["contact_forces"]

    net_forces = sensor.data.net_forces_w_history[
        :, :, env.sensor_foot_ids, :
    ]

    return (net_forces.norm(dim=-1).max(dim=1)[0] > 1.0).float()


# --------------------------------------------------------
# Marginal Error
# --------------------------------------------------------

def marginal_error(env):
    """
    Mean estimator error.

    0 = perfect
    1 = completely wrong
    """

    if not hasattr(env, "estimated_contacts"):
        return torch.zeros(env.num_envs, device=env.device)

    gt = ground_truth_contacts(env)

    return torch.abs(
        env.estimated_contacts - gt
    ).mean(dim=1)


# --------------------------------------------------------
# Grey-area score
# --------------------------------------------------------

def grey_area_score(env):
    """
    Contact uncertainty.

    0 = confident
    1 = maximally uncertain
    """

    if not hasattr(env, "estimated_contacts"):
        return torch.zeros(env.num_envs, device=env.device)

    p = env.estimated_contacts.clamp(
        1e-6,
        1.0 - 1e-6,
    )

    entropy = -(
        p * torch.log2(p)
        + (1.0 - p) * torch.log2(1.0 - p)
    )

    return entropy.mean(dim=1)


# --------------------------------------------------------
# Support Margin
# --------------------------------------------------------

def support_margin(env):
    """
    Measures how many feet are confidently supporting.

    0 = no support

    4 = all feet confidently supporting
    """

    if not hasattr(env, "estimated_contacts"):
        return torch.zeros(env.num_envs, device=env.device)

    p = env.estimated_contacts

    confident = (p > 0.7).float()

    return confident.sum(dim=1)


# --------------------------------------------------------
# Foothold Quality
# --------------------------------------------------------

def foothold_quality(env):
    """
    Composite PMETS score.

    Larger = better foothold.
    """

    me = marginal_error(env)

    ga = grey_area_score(env)

    sm = support_margin(env) / 4.0

    quality = (
        0.40 * (1.0 - me)
        + 0.30 * (1.0 - ga)
        + 0.30 * sm
    )

    return quality


# --------------------------------------------------------
# Torque balance
# --------------------------------------------------------

def torque_balance(env):
    """
    Mean torque per leg.

    Returns
    -------
    LF RF LH RH
    """

    robot = env.scene["robot"]

    tau = robot.data.applied_torque

    lf = tau[:, 0:3].abs().mean(dim=1)
    rf = tau[:, 3:6].abs().mean(dim=1)
    lh = tau[:, 6:9].abs().mean(dim=1)
    rh = tau[:, 9:12].abs().mean(dim=1)

    return lf, rf, lh, rh


# --------------------------------------------------------
# PMETS score
# --------------------------------------------------------

def pmets_score(env):
    """
    Main PMETS metric.

    Higher is better.
    """

    return foothold_quality(env)