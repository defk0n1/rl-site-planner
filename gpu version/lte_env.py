import gym
from gym import spaces
import numpy as np
import torch
from lte_utils import compute_rsrp_map, compute_sinr_map, build_candidate_graph

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LTEPlannerEnv(gym.Env):
    def __init__(self, clutter_map, candidate_positions, clutter_lookup, k=5, resolution=1.0):
        super().__init__()
        self.clutter_map = torch.tensor(clutter_map, dtype=torch.int, device=device)
        self.candidate_positions = candidate_positions  # still list of tuples, that's fine
        self.clutter_lookup = clutter_lookup
        self.k = k
        self.resolution = resolution
        self.num_nodes = len(candidate_positions)
        self.coverage_target = 0.95
        self.max_steps_per_episode = 10
        self.current_step = 0

        self.action_space = spaces.Dict({
            "placement": spaces.MultiBinary(self.num_nodes),
            "height": spaces.Box(low=0, high=1, shape=(self.num_nodes,)),
            "tilt": spaces.Box(low=0, high=1, shape=(self.num_nodes,)),
            # "azimuth": spaces.Box(low=0, high=1, shape=(self.num_nodes,))
            "azimuth":spaces.MultiDiscrete([71] * self.num_nodes)
        })

        self.observation_space = spaces.Dict({
            "x": spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_nodes, 5)),
            "edge_index": spaces.Tuple((
                spaces.Box(low=0, high=self.num_nodes - 1, dtype=np.int64),
                spaces.Box(low=0, high=self.num_nodes - 1, dtype=np.int64)
            ))
        })

    def reset(self):
        self.current_step = 0
        self.active_nodes = torch.zeros((self.num_nodes, 4), device=device)
        return self._get_state()

    def step(self, action):
        self.current_step += 1

        # Convert action to processed format
        placement = torch.tensor(action["placement"], dtype=torch.float32, device=device)
        height = torch.where(
            torch.tensor(action["height"], device=device) > 0.5,
            torch.tensor(45.0, device=device),
            torch.tensor(28.0, device=device)
        )
        tilt = torch.tensor(action["tilt"], device=device) * 12.0
        azimuth = torch.tensor(action["azimuth"], device=device) * 5.0


        processed_action = torch.stack([placement, height, tilt, azimuth], dim=1)

        # Compute metrics on GPU
        rsrp_map = compute_rsrp_map(
            self.clutter_map, self.candidate_positions, processed_action,
            clutter_lookup=self.clutter_lookup, resolution=self.resolution
        )
        sinr_map = compute_sinr_map(rsrp_map, self.candidate_positions, processed_action)

        # Compute coverage
        coverage = (rsrp_map > -95).float().mean().item()

        # Compute reward
        reward = self._compute_reward(rsrp_map, sinr_map, processed_action)

        done = (
            coverage >= self.coverage_target or
            self.current_step >= self.max_steps_per_episode
        )

        info = {
            "rsrp_map": rsrp_map.detach().cpu().numpy(),
            "sinr_map": sinr_map.detach().cpu().numpy(),
            "active_tx": processed_action[placement > 0.5].detach().cpu().numpy()
        }

        return self._get_state(), reward, done, info

    def _get_state(self):
        return build_candidate_graph(self.clutter_map, self.candidate_positions, k=self.k)

    def _compute_reward(self, rsrp_map, sinr_map, action):
        coverage = (rsrp_map > -95).float().mean()
        good_sinr = (sinr_map > 5).float().mean()
        total_power = action[:, 1].sum()
        num_tx = (action[:, 0] > 0.5).sum()

        return (
            2.0 * coverage.item() +
            1.0 * good_sinr.item() -
            0.2 * num_tx.item()
        )
