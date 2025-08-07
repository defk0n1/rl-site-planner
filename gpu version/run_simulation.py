import numpy as np
from lte_env import LTEPlannerEnv
from gnn_models import GNNPolicy, GNNValue
import torch
from torch.optim import Adam
from collections import deque
import matplotlib.pyplot as plt
import random
from visualise import log_episode 
from metrics_tracker import MetricsTracker
from datetime import datetime
import os






def get_experiment_folder(base_dir="experiments"):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(folder, exist_ok=True)
    return folder
    

exp_folder = get_experiment_folder("ppo_logs")
tracker = MetricsTracker(log_dir=exp_folder)



# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Environment Setup
h, w = 500, 500
clutter_map = np.ones((h, w), dtype=int) * 2
clutter_map[150:360, 150:360] = 5
clutter_map[180:330, 180:330] = 6
clutter_map[:100, :100] = 3
clutter_map[420:, 420:] = 1
clutter_map[100:200, 400:500] = 4
clutter_map[300:380, 100:160] = 7
candidate_positions = [(x, y) for x in range(50, 400, 100) for y in range(50, 450, 100)]

clutter_lookup = {
    0: ("urban", 0), 1: ("rural", 1), 2: ("rural", 0), 3: ("rural", 0.5),
    4: ("urban", 6), 5: ("rural", 9), 6: ("rural", 11), 7: ("rural", 15),
    8: ("rural", 5), 9: ("rural", 8), 10: ("suburban", 3), 11: ("urban", 10),
    12: ("suburban", 3), 13: ("urban", 8), 14: ("urban", 18), 15: ("suburban", 5),
    16: ("urban", 7), 17: ("rural", 2), 18: ("suburban", 4), 20: ("urban", 1.5)
}

env = LTEPlannerEnv(clutter_map, candidate_positions, clutter_lookup, resolution=1.0)
policy_net = GNNPolicy().to(device)
value_net = GNNValue().to(device)
optimizer = Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

# PPO Hyperparameters
CLIP_EPSILON = 0.2
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_COEF = 0.01
PPO_EPOCHS = 4
BATCH_SIZE = 64
BUFFER_SIZE = 2048
NUM_EPISODES = 1000

replay_buffer = deque(maxlen=BUFFER_SIZE)

# Helper Functions
def select_action(policy_net, state):
    state = state.to(device)
    with torch.no_grad():
        logits = policy_net(state)
        place_logits = logits[:, 0]
        place_probs = torch.sigmoid(place_logits)
        place_dist = torch.distributions.Bernoulli(probs=place_probs)
        place_sample = place_dist.sample()
        height_probs = torch.sigmoid(logits[:, 1])
        tilt_vals = torch.sigmoid(logits[:, 2])

        # print(logits[:,3:])
        azimuth_vals = torch.softmax(logits[:, 3:] , dim=1)
        azimuth_dist = torch.distributions.Categorical(probs=azimuth_vals)
        azimuth_sample = azimuth_dist.sample()

        action = {
            "placement": place_sample.cpu().numpy(),
            "height": height_probs.cpu().numpy(),
            "tilt": tilt_vals.cpu().numpy(),
            "azimuth": azimuth_sample.cpu().numpy()
        }
        place_log_prob = place_dist.log_prob(place_sample)
        height_log_prob = torch.distributions.Normal(height_probs, 0.1).log_prob(height_probs)
        tilt_log_prob = torch.distributions.Normal(tilt_vals, 0.1).log_prob(tilt_vals)
        azimuth_log_prob = azimuth_dist.log_prob(azimuth_sample)
        # total_log_prob = place_log_prob + height_log_prob + tilt_log_prob + azimuth_log_prob
        total_log_prob = (
            place_log_prob.sum()
            + height_log_prob.sum()
            + tilt_log_prob.sum()
            + azimuth_log_prob.sum()
            )
        return {
            'action_dict': action,
            'action_tensor': torch.cat([
                place_sample.unsqueeze(1),
                height_probs.unsqueeze(1),
                tilt_vals.unsqueeze(1),
                azimuth_sample.float().unsqueeze(1) 
            ], dim=1),
            'log_prob': total_log_prob, 
            'logits': logits,
            'place_probs': place_probs,
            'entropy': place_dist.entropy() + azimuth_dist.entropy()

        }

def compute_gae(rewards, values, dones, next_values):
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages

def ppo_update():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, old_log_probs, rewards, next_states, dones, values = zip(*batch)
    actions = torch.stack(actions).to(device)
    old_log_probs = torch.stack(old_log_probs).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    old_values = torch.stack(values).to(device)
    with torch.no_grad():
        next_values = torch.stack([value_net(s.to(device)).squeeze() for s in next_states])
    advantages = compute_gae(rewards, old_values, dones, next_values)
    returns = advantages + old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        new_log_probs = []
        values_pred = []
        for state in states:
            out = select_action(policy_net, state.to(device))
            new_log_probs.append(out['log_prob'])
            values_pred.append(value_net(state.to(device)).squeeze())
        new_log_probs = torch.stack(new_log_probs)
        values_pred = torch.stack(values_pred)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = 0.5 * (values_pred - returns).pow(2).mean()
        entropies = [select_action(policy_net, s.to(device))['entropy'] for s in states]
        entropy_loss = -ENTROPY_COEF * torch.stack(entropies).mean()
        total_loss = policy_loss + value_loss + entropy_loss
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        optimizer.step()

import matplotlib.pyplot as plt

def plot_metrics(metrics_log, save_path=None):
    episodes = [m['episode'] for m in metrics_log]
    rewards = [m['reward'] for m in metrics_log]
    entropies = [m['entropy'] for m in metrics_log if m['entropy'] is not None]
    kl_divs = [m['kl_div'] for m in metrics_log if m['kl_div'] is not None]
    variances = [m['variance'] for m in metrics_log if m['variance'] is not None]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    axs[0, 0].plot(episodes, rewards, label='Reward')
    axs[0, 0].set_title('Total Reward')
    axs[0, 0].set_xlabel('Episode')
    axs[0, 0].set_ylabel('Reward')
    axs[0, 0].grid(True)

    if entropies:
        axs[0, 1].plot(episodes[:len(entropies)], entropies, label='Entropy', color='orange')
        axs[0, 1].set_title('Entropy')
        axs[0, 1].set_xlabel('Episode')
        axs[0, 1].set_ylabel('Entropy')
        axs[0, 1].grid(True)

    if kl_divs:
        axs[1, 0].plot(episodes[:len(kl_divs)], kl_divs, label='KL Divergence', color='green')
        axs[1, 0].set_title('KL Divergence')
        axs[1, 0].set_xlabel('Episode')
        axs[1, 0].set_ylabel('KL Divergence')
        axs[1, 0].grid(True)

    if variances:
        axs[1, 1].plot(episodes[:len(variances)], variances, label='Action Variance', color='red')
        axs[1, 1].set_title('Action Variance')
        axs[1, 1].set_xlabel('Episode')
        axs[1, 1].set_ylabel('Variance')
        axs[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.show()









# Training Loop
rewards_history = []
for episode in range(NUM_EPISODES):
    print(f"[Episode {episode}] Starting...")
    state = env.reset().to(device)
    done = False
    episode_rewards = []
    episode_rsrp_map = None
    active_sites_history = []  # New: Track active sites per episode
    metrics = {} 


    while not done:
        action_data = select_action(policy_net, state)
        value = value_net(state).squeeze()
        next_state, reward, done, info = env.step(action_data['action_dict'])
        next_state = next_state.to(device)
        replay_buffer.append((
            state,
            action_data['action_tensor'],
            action_data['log_prob'],
            reward,
            next_state,
            done,
            value.detach()
        ))

        if done and 'rsrp_map' in info:
            episode_rsrp_map = info['rsrp_map']

        state = next_state
        # Track active sites
        active_mask = action_data['action_dict']['placement'] > 0.5
        active_sites = [i for i, active in enumerate(active_mask) if active]
        active_sites_history.append(active_sites)

        print(f"Active sites for episode : {len(active_sites)}")

        episode_rewards.append(reward)
        print(f"  Step: reward={reward:.2f}, done={done}")
    
        
    # === Metrics for the episode ===
    # 1. KL Divergence between old and new log probs
    batch = list(replay_buffer)[-BATCH_SIZE:]  # Use last batch
    states, _, old_log_probs, _, _, _, _ = zip(*batch)
    with torch.no_grad():
        old_log_probs = torch.stack(old_log_probs).to(device)
        new_log_probs = torch.stack([
            select_action(policy_net, s.to(device))['log_prob']
            for s in states
        ])
        kl_div = (old_log_probs.exp() * (old_log_probs - new_log_probs)).mean()

    # 2. Entropy
    entropies = torch.stack([
        select_action(policy_net, s.to(device))['entropy']
        for s in states
    ])
    avg_entropy = entropies.mean()

    # 3. Action Variance
    actions = torch.stack([
        select_action(policy_net, s.to(device))['action_tensor']
        for s in states
    ])
    action_var = actions.var(dim=0).mean()


    print("[PPO Update] Starting...")
    ppo_update()
    total_reward = sum(episode_rewards)
    rewards_history.append(total_reward)
    print(f"Episode {episode}: Total Reward = {total_reward:.2f}")

    # === Log metrics ===
    tracker.log(
        episode=episode,
        reward=total_reward,
        entropy=avg_entropy.item(),
        kl_div=kl_div.item(),
        variance=action_var.item()
    )
      

    if episode % 100 == 0:
        log_episode(
            episode, 
            total_reward,
            active_sites_history[-1], 
            episode_rsrp_map,
            candidate_positions=candidate_positions)
        if episode > 0:
            plot_metrics(tracker.metrics, save_path=os.path.join(exp_folder, f"metrics_ep_{episode}.png"))

            
            
        print("episode info: ",info)
        torch.save(policy_net.state_dict(), os.path.join(exp_folder, f"ppo_gnn_policy_{episode}.pth"))
        torch.save(value_net.state_dict(), os.path.join(exp_folder, f"ppo_gnn_value_{episode}.pth"))

# Final reward plot
plt.figure(figsize=(12, 4))
plt.plot(rewards_history)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()
