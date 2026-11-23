# Copyright (c) 2022-2026
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

_DCE_SRC = os.getenv("CONTACTNET_SRC", os.path.abspath(os.path.join(os.path.dirname(__file__), "../../ContactNet/src")))

if _DCE_SRC not in sys.path:
    sys.path.insert(0, _DCE_SRC)


class ContactEstimatorWrapper:

    def __init__(
        self,
        model_path: str,
        num_envs: int,
        #window_size: int = 150,
        
        # contact_estimator_wrapper.py
        window_size: int = 150,  
        input_dim: int = 48,
        device: str = "cuda",
    ):

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.window_size = window_size
        self.input_dim = input_dim
        self.num_envs = num_envs

        # LOAD MODEL
        # ─────────────────────────────────────────────────────
        self.model = self._load_model(model_path)

        # CLASS CONTACT PATTERNS # DOES THIS REALLY CREATE A CONTACT BELIEF IF SO THEN HOW SO, CONVINCE ME MATHEMATICALLY GIVEN THAT I ALREADY KNOW WHAT A BELIEF STATE IS ...
        # Maps class index (0–15) → LF RF LH RH. Shape: (16, 4)
        # ─────────────────────────────────────────────────────
        patterns = []

        for i in range(16):
            bits = [(i >> j) & 1 for j in range(3, -1, -1)]
            patterns.append(bits)

        self.class_patterns = torch.tensor(patterns,dtype=torch.float32,device=self.device)

        # NORMALIZATION CONSTANTS
        # Replace with training stats if available(what does this mean?)
        # ─────────────────────────────────────────────────────
        self.register_norm_constants()

        print("[INFO] Contact estimator loaded successfully.")
        print(f"[INFO] Wrapper ready — {num_envs} envs on {self.device}")

    def register_norm_constants(self):
        """
        Pre-computed normalization constants.
        Shape: (1, 1, 48)
        """

        self.norm_mean = torch.zeros(1,1,48, device=self.device,dtype=torch.float32)
        self.norm_std = torch.ones(1,1,48,device=self.device,dtype=torch.float32)

    @torch.no_grad()
    def predict(self, input_buffer: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_buffer: (N, T, 48)

        Returns:
            contacts: (N, 4)
            soft contact probabilities [0 → 1]
        """
        # NORMALIZE
        # ─────────────────────────────────────────────────────

        normalized = (input_buffer - self.norm_mean) / (self.norm_std + 1e-8)
        #import contact_cnn
        #print(">>> contact_cnn loaded from:", contact_cnn.__file__)
        #print(">>> input_buffer shape:", input_buffer.shape)
        # CNN expects (N, C, T)
        #x = normalized.permute(0, 2, 1).contiguous().float()
        x = normalized.contiguous().float()
        # FORWARD PASS
        # ─────────────────────────────────────────────────────

        logits = self.model(x)
        if isinstance(logits, tuple):
            logits = logits[0]

        # LOGITS → CLASS PROBABILITIES
        # Shape: (N, 16)
        # ─────────────────────────────────────────────────────
        class_probs = torch.softmax(logits, dim=1)
        # Save the complete 16-class probability distribution
        self.last_probs = class_probs.detach()
        #print(x.shape)
        #print(">>> predict() called")
        #print(">>> last_probs stored")
        # CLASS PROBS → PER-FOOT CONTACT PROBS
        # Shape: (N, 4) Values: [0.0 → 1.0]
        # ─────────────────────────────────────────────────────

        contact_probs = class_probs @ self.class_patterns
        entropy = -(class_probs *torch.log(class_probs + 1e-8)).sum(dim=1)
        self.last_entropy = entropy.detach()
        
        # # LOGITS → CLASS PROBABILITIES
        # class_probs = torch.softmax(logits, dim=1)

        # # CLASS PROBS → CONTACT PROBS
        # contact_probs = class_probs @ self.class_patterns

        # # ENTROPY LOGGING
        # entropy = -(class_probs *torch.log(class_probs + 1e-8)).sum(dim=1)
        # self.last_entropy = entropy.detach()

        # RETURN RAW CONTACT PROBABILITIES
        return contact_probs

        # ─────────────────────────────────────────────────────
        # EMA SMOOTHING ( for the next one remove ema completely and just return the raw contact_probs)
        #
        # smoothed = α * prev + (1 - α) * current
        # ─────────────────────────────────────────────────────

        # smoothed_contacts = (
        #     self.ema_alpha * self.prev_contacts
        #     + (1.0 - self.ema_alpha) * contact_probs
        # )

        # # store for next step
        # self.prev_contacts.copy_(smoothed_contacts)

        
        # return smoothed_contacts
    
    

    def reset(self, env_ids=None):
        # WHY DID WE NEED THE EMA FOR ENVS AGAIN JUSTIFY THIS ??? also i have a eeling this doesnt workas prev contacts isnt defined its just a filler 
        """
        Reset EMA state for environments. 

        Call this during env reset.
        """

        if env_ids is None:

            self.prev_contacts.zero_()

        else:

            self.prev_contacts[env_ids] = 0.0

    @staticmethod
    def _decimal_to_binary(x: torch.Tensor) -> torch.Tensor:
        """
        Class index (0–15) → binary contacts (N, 4)
        MSB first: LF RF LH RH
        """

        mask = 2 ** torch.arange(3,-1,-1,
            device=x.device,
            dtype=x.dtype,
        )

        return (
            x.unsqueeze(-1)
            .bitwise_and(mask.long())
            .ne(0)
            .float()
        )

    def _load_model(self, model_path: str) -> nn.Module:

        try:
            from contact_cnn import contact_cnn_gru_progressive

        except ImportError as e:

            raise ImportError(
                f"\nCould not import contact model.\n"
                f"Expected path:\n{_DCE_SRC}\n"
                f"Original error:\n{e}"
            )

        checkpoint = torch.load(
            model_path,
            map_location=self.device,
        )

        if "state_dict" in checkpoint:

            raw = checkpoint["state_dict"]

            state_dict = {
                k.replace("model.", "", 1): v
                for k, v in raw.items()
            }

        elif "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        else:

            raise ValueError(
                f"Unknown checkpoint format. Keys: {checkpoint.keys()}"
            )

        model = contact_cnn_gru_progressive(
            input_channels=48,
            num_classes=16,
        )

        model.load_state_dict(state_dict)

        model.eval()

        model = model.to(self.device)

        return model