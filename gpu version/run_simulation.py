from load_clutter_data import load_clutter_map
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

# clutter_map = load_clutter_map("index_clutter.txt")

clutter_map_unmapped = np.load("rastermap.npy",mmap_mode='r')
h, w = clutter_map_unmapped.shape

original_values = np.array([
    256, 512, 768, 1024, 1280, 1536, 1792, 2048, 2304, 2560,
    2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 5120, 61912
])
new_classes = np.where(original_values == 61912, 0, original_values // 256)
mapping = dict(zip(original_values, new_classes))

# --- Extract unique values in the raster ---
unique_vals = np.unique(clutter_map_unmapped)

# --- Build a dict that includes defaults (-1) ---
remap_dict = {val: mapping.get(val, -1) for val in unique_vals}

# --- Vectorized remapping using np.take with indexer ---
# Step 1: make an array of remapped values in the order of unique_vals
remap_array = np.array([remap_dict[v] for v in unique_vals], dtype=np.int32)

# Step 2: use np.searchsorted to map all pixels via unique indices
indices = np.searchsorted(unique_vals, clutter_map_unmapped)
clutter_map = remap_array[indices]


print(clutter_map)
print(clutter_map.shape)




# Environment Setup
# h, w = 500, 500
# clutter_map = np.ones((h, w), dtype=int) * 2
# clutter_map[150:360, 150:360] = 5
# clutter_map[180:330, 180:330] = 6
# clutter_map[:100, :100] = 3
# clutter_map[420:, 420:] = 1
# clutter_map[100:200, 400:500] = 4
# clutter_map[300:380, 100:160] = 7
candidate_positions = [(x, y) for x in range(0, h, 10) for y in range(0, w, 10) if x < h and y < w and clutter_map[x , y] != -1]

print(len(candidate_positions))


clutter_lookup = {
    0: ("urban", 0), 1: ("rural", 1), 2: ("rural", 0), 3: ("rural", 0.5),
    4: ("urban", 6), 5: ("rural", 9), 6: ("rural", 11), 7: ("rural", 15),
    8: ("rural", 5), 9: ("rural", 8), 10: ("suburban", 3), 11: ("urban", 10),
    12: ("suburban", 3), 13: ("urban", 8), 14: ("urban", 18), 15: ("suburban", 5),
    16: ("urban", 7), 17: ("rural", 2), 18: ("suburban", 4), 20: ("urban", 1.5),
    -1:("outofbounds",-1)
}




LEARNING_RATE = 3e-4

env = LTEPlannerEnv(clutter_map, candidate_positions, clutter_lookup, resolution=50)
policy_net = GNNPolicy().to(device)
value_net = GNNValue().to(device)
optimizer = Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=LEARNING_RATE)

# PPO Hyperparameters
CLIP_EPSILON = 0.5
GAMMA = 0.99
LAMBDA = 0.95
ENTROPY_COEF = 0.5
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

    # --------------------------
    # Placement (Bernoulli)
    # --------------------------
    place_logits = logits[:, 0]
    place_probs = torch.sigmoid(place_logits)
    place_dist = torch.distributions.Bernoulli(probs=place_probs)
    place_sample = place_dist.sample()

    # --------------------------
    # Height (Continuous Normal)
    # --------------------------
    height_mean = logits[:, 1]      # raw output
    height_std = 0.5                # or learnable
    height_dist = torch.distributions.Normal(height_mean, height_std)
    height_sample = height_dist.rsample()  # reparameterized sample
    height_env = torch.clamp(height_sample * 12 + 28, 28.0, 45.0)  # map to env range

    # --------------------------
    # Tilt (Continuous Normal)
    # --------------------------
    tilt_mean = logits[:, 2]
    tilt_std = 0.5
    tilt_dist = torch.distributions.Normal(tilt_mean, tilt_std)
    tilt_sample = tilt_dist.rsample()
    tilt_env = torch.clamp(tilt_sample * 12, 0.0, 12.0)

    # --------------------------
    # Azimuth (Categorical)
    # --------------------------
    azimuth_logits = logits[:, 3:]
    azimuth_dist = torch.distributions.Categorical(logits=azimuth_logits)
    azimuth_sample = azimuth_dist.sample()
    azimuth_env = azimuth_sample * 5.0  # map to env range

    # --------------------------
    # Log probabilities
    # --------------------------
    log_prob = (
        place_dist.log_prob(place_sample).sum()
        + height_dist.log_prob(height_sample).sum()
        + tilt_dist.log_prob(tilt_sample).sum()
        + azimuth_dist.log_prob(azimuth_sample).sum()
    )

    entropy = (
        place_dist.entropy().sum()
        + height_dist.entropy().sum()
        + tilt_dist.entropy().sum()
        + azimuth_dist.entropy().sum()
    )

    action_dict = {
        "placement": place_sample.cpu().numpy(),
        "height": height_env.cpu().numpy(),
        "tilt": tilt_env.cpu().numpy(),
        "azimuth": azimuth_env.cpu().numpy()
    }

    action_tensor = torch.cat([
        place_sample.unsqueeze(1),
        height_sample.unsqueeze(1),
        tilt_sample.unsqueeze(1),
        azimuth_sample.float().unsqueeze(1)
    ], dim=1)

    return {
        "action_dict": action_dict,
        "action_tensor": action_tensor,
        "log_prob": log_prob,
        "logits": logits,
        "entropy": entropy,
        "dists": {
        "place": place_probs.detach(),         # store probs as tensor
        "height": (height_mean.detach(), torch.tensor(height_std)), 
        "tilt": (tilt_mean.detach(), torch.tensor(tilt_std)),
        "azimuth": azimuth_logits.detach()     # store logits as tensor
    }



        
    }


def compute_gae(rewards, values, dones, next_values):
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages

# def ppo_update():
#     if len(replay_buffer) < BATCH_SIZE:
#         return
#     batch = list(replay_buffer)[-BATCH_SIZE:]

    
#     states, actions, old_log_probs, rewards, next_states, dones, values = zip(*batch)

#     actions = torch.stack(actions).to(device)
#     old_log_probs = torch.stack(old_log_probs).to(device)
#     rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
#     dones = torch.tensor(dones, dtype=torch.float32, device=device)
#     old_values = torch.stack(values).to(device)


#     with torch.no_grad():
#         next_values = torch.stack([value_net(s.to(device)).squeeze() for s in next_states])


#     advantages = compute_gae(rewards, old_values, dones, next_values)
#     returns = advantages + old_values
#     advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

#     for _ in range(PPO_EPOCHS):
#         new_log_probs = []
#         values_pred = []
#         for state in states:
#             out = select_action(policy_net, state.to(device))
#             new_log_probs.append(out['log_prob'])
#             values_pred.append(value_net(state.to(device)).squeeze())
#         new_log_probs = torch.stack(new_log_probs)
#         values_pred = torch.stack(values_pred)
#         ratio = torch.exp(new_log_probs - old_log_probs)
#         surr1 = ratio * advantages
#         surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
#         policy_loss = -torch.min(surr1, surr2).mean()
#         value_loss = 0.5 * (values_pred - returns).pow(2).mean()
#         entropies = [select_action(policy_net, s.to(device))['entropy'] for s in states]
#         entropy_loss = -ENTROPY_COEF * torch.stack(entropies).mean()
#         total_loss = policy_loss + value_loss + entropy_loss
#         optimizer.zero_grad()
#         total_loss.backward()
#         torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
#         torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
#         optimizer.step()

def ppo_update():
    if len(replay_buffer) < BATCH_SIZE:
        return
    batch = list(replay_buffer)[-BATCH_SIZE:]

    states, actions, old_log_probs, rewards, next_states, dones, values, old_dists = zip(*batch)

    actions = torch.stack(actions).to(device)
    old_log_probs = torch.stack(old_log_probs).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
    dones = torch.tensor(dones, dtype=torch.float32, device=device)
    old_values = torch.stack(values).to(device)

    # bootstrap from last value
    with torch.no_grad():
        next_values = torch.stack([value_net(s.to(device)).squeeze() for s in next_states])

    advantages = compute_gae(rewards, old_values, dones, next_values)
    returns = advantages + old_values
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    for _ in range(PPO_EPOCHS):
        new_log_probs = []
        values_pred = []
        entropies = []
        kl_divs = []

        for state, action, old_dist in zip(states, actions, old_dists):
            logits = policy_net(state.to(device))

            # rebuild distributions
            place_dist = torch.distributions.Bernoulli(logits=logits[:, 0])
            height_dist = torch.distributions.Normal(logits[:, 1], 0.5)
            tilt_dist   = torch.distributions.Normal(logits[:, 2], 0.5)
            azimuth_dist = torch.distributions.Categorical(logits=logits[:, 3:])

            # compute new log prob for stored action
            place_lp   = place_dist.log_prob(action[:, 0])
            height_lp  = height_dist.log_prob(action[:, 1])
            tilt_lp    = tilt_dist.log_prob(action[:, 2])
            azimuth_lp = azimuth_dist.log_prob(action[:, 3].long())

            log_prob = place_lp.sum() + height_lp.sum() + tilt_lp.sum() + azimuth_lp.sum()
            new_log_probs.append(log_prob)

            # value
            values_pred.append(value_net(state.to(device)).squeeze())

            # entropy
            entropy = (
                    place_dist.entropy().sum() +
                    height_dist.entropy().sum() +
                    tilt_dist.entropy().sum() +
                    azimuth_dist.entropy().sum()
                )
            entropies.append(entropy)

            

        new_log_probs = torch.stack(new_log_probs)
        values_pred = torch.stack(values_pred)
        entropies = torch.stack(entropies)

        print(new_log_probs.shape , old_log_probs.shape)

        # PPO surrogate loss
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - CLIP_EPSILON, 1 + CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # value loss
        value_loss = 0.5 * (values_pred - returns).pow(2).mean()

        # entropy bonus
        entropy_loss = -ENTROPY_COEF * entropies.mean()

        total_loss = policy_loss + value_loss + entropy_loss

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        optimizer.step()

        # log for debugging
        print(f"Entropy: {entropies.mean().item():.4f} "
              f"Policy Loss: {policy_loss.item():.4f}, Value Loss: {value_loss.item():.4f}")




import matplotlib.pyplot as plt

def plot_metrics(metrics_log, save_path=None):
    episodes = [m['episode'] for m in metrics_log]
    rewards = [m['reward'] for m in metrics_log]
    entropies = [m['entropy'] for m in metrics_log if m['entropy'] is not None]
    kl_divs = [m['kl_div'] for m in metrics_log if m['kl_div'] is not None]
    variances = [m['variance'] for m in metrics_log if m['variance'] is not None]

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))

    print(episodes , rewards)

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
            value.detach(),
            action_data['dists']   # store distribution parameters

        ))

        if done and 'rsrp_map' in info:
            episode_rsrp_map = info['rsrp_map']

        state = next_state
        # Track active sites
        active_mask = action_data['action_dict']['placement'] > 0.5
        active_sites = [i for i, active in enumerate(active_mask) if active]
        active_sites_history.append(active_sites)

        print(f"Active sites for episode : {len(active_sites)}")

        episode_rewards.append(reward.item())
        print(f"  Step: reward={reward:.2f}, done={done}")
    
        
# # === Metrics for the episode ===
# # 1. KL Divergence between old and new log probs
        batch = list(replay_buffer)[-BATCH_SIZE:]  # Use last batch
        states, actions, old_log_probs, rewards, next_states, dones, values , old_dists = zip(*batch)

#     print(old_dists)  
#     old_place_probs = torch.stack([d["place"] for d in old_dists]).to(device)
#     old_place_dist = torch.distributions.Bernoulli(probs=old_place_probs)

#     old_height_means = torch.stack([d["height"][0] for d in old_dists]).to(device)
#     old_height_stds = torch.stack([d["height"][1] for d in old_dists]).to(device)
#     old_height_dist = torch.distributions.Normal(old_height_means, old_height_stds)

#     old_tilt_means = torch.stack([d["tilt"][0] for d in old_dists]).to(device)
#     old_tilt_stds = torch.stack([d["tilt"][1] for d in old_dists]).to(device)
#     old_tilt_dist = torch.distributions.Normal(old_tilt_means, old_tilt_stds)

#     old_azimuth_logits = torch.stack([d["azimuth"] for d in old_dists]).to(device)
#     old_azimuth_dist = torch.distributions.Categorical(logits=old_azimuth_logits)

#     with torch.no_grad():
#         new_dists = [select_action(policy_net, s.to(device))["dists"] for s in states]
#     # Example new place distribution
#     new_place_probs = torch.stack([d["place"] for d in new_dists]).to(device)
#     new_place_dist = torch.distributions.Bernoulli(probs=new_place_probs)

#     new_height_means = torch.stack([d["height"][0] for d in new_dists]).to(device)
#     new_height_stds  = torch.stack([d["height"][1] for d in new_dists]).to(device)
#     new_height_dist  = torch.distributions.Normal(new_height_means, new_height_stds)

#     new_tilt_means = torch.stack([d["tilt"][0] for d in new_dists]).to(device)
#     new_tilt_stds  = torch.stack([d["tilt"][1] for d in new_dists]).to(device)
#     new_tilt_dist  = torch.distributions.Normal(new_tilt_means, new_tilt_stds)

#     new_az_logits = torch.stack([d["azimuth"] for d in new_dists]).to(device)
#     new_az_dist   = torch.distributions.Categorical(logits=new_az_logits)
# # Now compute KL (do this per-head and sum if needed)
#     kl_place = torch.distributions.kl_divergence(old_place_dist, new_place_dist).mean()
#     kl_height = torch.distributions.kl_divergence(old_height_dist, new_height_dist).mean()
#     kl_tilt   = torch.distributions.kl_divergence(old_tilt_dist, new_tilt_dist).mean()
#     kl_az     = torch.distributions.kl_divergence(old_azimuth_dist, new_az_dist).mean()

#     kl_total = kl_place + kl_height + kl_tilt + kl_az

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
        # kl_div=kl_div.item(),
        variance=action_var.item()
    )
      

    if episode % 10 == 0:
        log_episode(
            episode, 
            total_reward,
            active_sites_history[-1], 
            episode_rsrp_map,
            candidate_positions=candidate_positions)
        if episode > 0:
            plot_metrics(tracker.metrics, save_path=os.path.join(exp_folder, f"metrics_ep_{episode}.png"))

            
            
        print("episode info: ", info )
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
