import numpy as np
from lte_env import LTEPlannerEnv
from gnn_models import GNNPolicy , GNNValue
import torch
from torch.optim import Adam
from collections import deque
import matplotlib.pyplot as plt
import random


h,w = 500 , 500  

clutter_map = np.ones((h, w), dtype=int) * 2  # open land
clutter_map[150:360, 150:360] = 5  # urban
clutter_map[180:330, 180:330] = 6  # dense urban
clutter_map[:100, :100] = 3        # forest
clutter_map[420:, 420:] = 1        # water
clutter_map[100:200, 400:500] = 4  # suburban
clutter_map[300:380, 100:160] = 7  # industrial
# Dummy input data
# clutter_map = np.random.randint(0, 3, size=(50, 50))
candidate_positions = [(x, y) for x in range(50, 400, 100) for y in range(50, 450, 100)]

print(len(candidate_positions))

clutter_lookup = {
    0: ("urban", 0),
    1: ("rural", 1),
    2: ("rural", 0),
    3: ("rural", 0.5),
    4: ("urban", 6),
    5: ("rural", 9),
    6: ("rural", 11),
    7: ("rural", 15),
    8: ("rural", 5),
    9: ("rural", 8),
    10: ("suburban", 3),
    11: ("urban", 10),
    12: ("suburban", 3),
    13: ("urban", 8),
    14: ("urban", 18),
    15: ("suburban", 5),
    16: ("urban", 7),
    17: ("rural", 2),
    18: ("suburban", 4),
    20: ("urban", 1.5)
}
env = LTEPlannerEnv(clutter_map, candidate_positions, clutter_lookup, resolution=1.0)
policy_net = GNNPolicy()
value_net = GNNValue(in_dim=3, hidden_dim=64)
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

# Replay buffer
replay_buffer = deque(maxlen=BUFFER_SIZE)

def select_action(policy_net, state):
    """Returns action dict compatible with our modified action space"""
    with torch.no_grad():
        logits = policy_net(state)
        
        # Binary placement
        place_logits = logits[:, 0]
        place_probs = torch.sigmoid(place_logits)
        place_dist = torch.distributions.Bernoulli(probs=place_probs)
        place_sample = place_dist.sample()
        
        # Continuous parameters (already normalized to [0,1])
        height_probs = torch.sigmoid(logits[:, 1])
        tilt_vals = torch.sigmoid(logits[:, 2])
        azimuth_vals = torch.sigmoid(logits[:, 3])
        
        # Create action dictionary
        action = {
            "placement": place_sample.numpy(),
            "height": height_probs.numpy(),
            "tilt": tilt_vals.numpy(),
            "azimuth": azimuth_vals.numpy()
        }
        
        # Calculate log probs
        place_log_prob = place_dist.log_prob(place_sample)
        height_log_prob = torch.distributions.Normal(height_probs, 0.1).log_prob(height_probs)
        tilt_log_prob = torch.distributions.Normal(tilt_vals, 0.1).log_prob(tilt_vals)
        azimuth_log_prob = torch.distributions.Normal(azimuth_vals, 0.1).log_prob(azimuth_vals)
        total_log_prob = place_log_prob + height_log_prob + tilt_log_prob + azimuth_log_prob
        
        # return action, total_log_prob, logits, place_probs, place_dist.entropy()
        return  {
        'action_dict': {  # For env.step()
            "placement": place_sample.numpy(),
            "height": height_probs.numpy(),
            "tilt": tilt_vals.numpy(),
            "azimuth": azimuth_vals.numpy()
        },
        'action_tensor': torch.cat([  # For gradient computation
            place_sample.unsqueeze(1),
            height_probs.unsqueeze(1),
            tilt_vals.unsqueeze(1),
            azimuth_vals.unsqueeze(1)
        ], dim=1),
        'log_prob': total_log_prob,
        'logits': logits,
        'place_probs': place_probs,
        'entropy': place_dist.entropy()
    }
       

def log_episode(episode, total_reward, active_sites, rsrp_map):
    """Comprehensive episode logging"""
    print(f"\n=== Episode {episode} Summary ===")
    print(f"Total Reward: {total_reward:.2f}")
    print(f"Active Transmitters: {len(active_sites)}/{len(candidate_positions)}")
    print("Active Site Indices:", active_sites)
    
    if rsrp_map is not None:
        print("\nCoverage Statistics:")
        print(f"- Area > -95dBm: {(rsrp_map > -95).mean()*100:.1f}%")
        print(f"- Area > -100dBm: {(rsrp_map > -100).mean()*100:.1f}%")
    
    # Visualizations
    plot_active_sites(candidate_positions, active_sites, episode)
    if rsrp_map is not None:
        plot_rsrp_coverage(rsrp_map, episode)


def plot_active_sites(candidate_positions, active_indices, episode):
    """Visualize transmitter placements"""
    plt.figure(figsize=(10, 8))
    all_x, all_y = zip(*candidate_positions)
    
    # Plot all candidates
    plt.scatter(all_x, all_y, c='gray', marker='x', label='Inactive Sites')
    
    # Highlight active ones
    if active_indices:
        active_x = [candidate_positions[i][0] for i in active_indices]
        active_y = [candidate_positions[i][1] for i in active_indices]
        plt.scatter(active_x, active_y, c='red', marker='o', s=100, label='Active TX')
    
    plt.title(f"Episode {episode}: Active Transmitters")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_rsrp_coverage(rsrp_map, episode):
    """Visualize RSRP heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(rsrp_map, cmap='viridis', vmin=-120, vmax=-70)
    plt.colorbar(label='RSRP (dBm)')
    plt.title(f"Episode {episode}: RSRP Coverage")
    plt.show()


def compute_gae(rewards, values, dones, next_values):
    """Compute Generalized Advantage Estimation"""
    advantages = torch.zeros_like(rewards)
    gae = 0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + GAMMA * next_values[t] * (1 - dones[t]) - values[t]
        gae = delta + GAMMA * LAMBDA * (1 - dones[t]) * gae
        advantages[t] = gae
    return advantages

def ppo_update():
    """Perform PPO update using samples from replay buffer"""
    if len(replay_buffer) < BATCH_SIZE:
        return
    
    # Sample batch
    batch = random.sample(replay_buffer, BATCH_SIZE)
    states, actions, old_log_probs, rewards, next_states, dones, values = zip(*batch)
    
    # Convert to tensors
    states = [s for s in states]  # List of Data objects
    actions = torch.stack(actions)
    old_log_probs = torch.stack(old_log_probs)
    rewards = torch.tensor(rewards, dtype=torch.float32)
    next_states = [s for s in next_states]
    dones = torch.tensor(dones, dtype=torch.float32)
    old_values = torch.stack(values)
    
    # Compute advantages
    with torch.no_grad():
        next_values = torch.stack([value_net(s).squeeze() for s in next_states])
    advantages = compute_gae(rewards, old_values, dones, next_values)
    returns = advantages + old_values
    
    # Normalize advantages
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    # PPO epochs
    for _ in range(PPO_EPOCHS):
        # Get new policy's log probs and values
        new_log_probs = []
        values_pred = []
        for state in states:
            _, log_prob, _, _, _ = select_action(policy_net, state)
            new_log_probs.append(log_prob)
            values_pred.append(value_net(state).squeeze())
        
        new_log_probs = torch.stack(new_log_probs)
        values_pred = torch.stack(values_pred)
        
        # Policy loss (clipped)
        ratio = torch.exp(new_log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1-CLIP_EPSILON, 1+CLIP_EPSILON) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        value_loss = 0.5 * (values_pred - returns).pow(2).mean()
        
        # Entropy bonus
        entropies = []
        for state in states:
            _, _, _, _, entropy = select_action(policy_net, state)
            entropies.append(entropy)
        entropy_loss = -ENTROPY_COEF * torch.stack(entropies).mean()
        
        # Total loss
        total_loss = policy_loss + value_loss + entropy_loss
        
        # Optimize
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 0.5)
        torch.nn.utils.clip_grad_norm_(value_net.parameters(), 0.5)
        optimizer.step()

# Training loop
rewards_history = []
losses_history = []

for episode in range(NUM_EPISODES):
    state = env.reset()
    done = False
    episode_rewards = []
    active_sites_history = []  # New: Track active sites per episode
    rsrp_maps = []            # New: Store RSRP maps for visualization
    episode_rsrp_map = None

    while not done:
        # Get action and value
        action_data = select_action(policy_net, state)

        # print(action_data)
        value = value_net(state).squeeze()

        print(f"Get action and value for episode : {episode}")

        
        # Step environment
        next_state, reward, done, info = env.step(action_data['action_dict'])
        episode_rewards.append(reward)
        
        print(f"Step environment for episode : {episode}")

        # Store transition
        replay_buffer.append((
            state,
            action_data['action_tensor'],
            action_data['log_prob'],
            reward,
            next_state,
            done,
            value
        ))

        print(f"Current state of replay buffer : {replay_buffer} ")
        if done and 'rsrp_map' in info:
            episode_rsrp_map = info['rsrp_map']
        
        state = next_state
        # Track active sites
        active_mask = action_data['action_dict']['placement'] > 0.5
        active_sites = [i for i, active in enumerate(active_mask) if active]
        active_sites_history.append(active_sites)

        print(f"Active sites for episode : {episode}")

        
        # Capture final RSRP map
        if done and 'rsrp_map' in info:
            episode_rsrp_map = info['rsrp_map']
    
    log_episode(episode, total_reward, active_sites_history[-1], rsrp_maps[-1])

    # Update networks
    ppo_update()
    
    # Logging
    total_reward = sum(episode_rewards)
    rewards_history.append(total_reward)
    print(f"Episode {episode}: Total Reward = {total_reward:.2f}")
    # Enhanced logging
    if 10 % 10 == 0:  # Log every 10 episodes
        log_episode(
            episode, 
            total_reward,
            active_sites_history[-1], 
            episode_rsrp_map
        )
    if episode % 100 == 0:
        # Visualize and save models
        torch.save(policy_net.state_dict(), f"ppo_gnn_policy_{episode}.pth")
        torch.save(value_net.state_dict(), f"ppo_gnn_value_{episode}.pth")

# Plot training progress
plt.figure(figsize=(12, 4))
plt.plot(rewards_history)
plt.title('Training Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.grid(True)
plt.show()








# def plot_episode(clutter_map, candidate_positions, action_np, episode):
#     plt.figure(figsize=(6, 6))
#     plt.imshow(clutter_map, cmap='tab20', origin='lower')
#     xs, ys = zip(*candidate_positions)
#     plt.scatter(xs, ys, c='gray', marker='x', label='Candidate Sites')

#     # Highlight active transmitters
#     for (x, y), (place, *_rest) in zip(candidate_positions, action_np):
#         if place > 0.5:
#             plt.scatter(x, y, c='red', marker='o', s=100, label='Active Site')

#     plt.title(f"Episode {episode}: Activated Sites")
#     plt.legend(loc='upper right')
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()   

# def select_action(policy_net, state):
#     logits = policy_net(state)  # shape [num_candidates, 4]
    
#     place_logits = logits[:, 0]            # [N]
#     place_probs = torch.sigmoid(place_logits)  # [N]

#     configs = torch.sigmoid(logits[:, 1:])  # [N, 3] → power, height, tilt ∈ [0, 1]

#     place_dist = torch.distributions.Bernoulli(probs=place_probs)
#     place_sample = place_dist.sample()      # [N] binary: 0 or 1

#     entropy = place_dist.entropy().sum()


#     log_prob = place_dist.log_prob(place_sample)  # [N]

#     action = torch.cat([place_sample.unsqueeze(1), configs], dim=1)  # [N, 4]

#     return action, log_prob.sum(), logits, place_probs , entropy




# def plot_place_probs(place_probs, episode):
#     plt.figure(figsize=(6, 2))
#     plt.bar(range(len(place_probs)), place_probs.detach().numpy(), color='skyblue')
#     plt.ylim(0, 1)
#     plt.xlabel("Candidate Transmitter Index")
#     plt.ylabel("Placement Probability")
#     plt.title(f"Episode {episode} - Placement Probabilities")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()


# def print_weight_deltas(model, name="PolicyNet"):
#     for pname, p in model.named_parameters():
#         if p.grad is not None:
#             print(f"[{name}] Δ {pname}: {p.grad.abs().sum().item():.6f}")
#         else:
#             print(f"[{name}] {pname} has no gradient!")


# rewards = []
# losses = []
# num_txs = []




# for episode in range(1000):
#     state = env.reset()
#     action, log_prob, logits, place_probs , entropy = select_action(policy_net, state)
#     action_np = action.detach().numpy()
#     next_state, reward, done, _ = env.step(action_np)

#     loss = -log_prob * reward -0.01 * entropy


#     optimizer.zero_grad()


#     loss.backward()
#     print_weight_deltas(policy_net)

#     optimizer.step()

#     rewards.append(reward)
#     losses.append(loss.item())

#     print(f"Episode {episode}: Reward = {reward:.2f}, Loss = {loss.item():.4f}")
#     if episode % 100 == 0 : 
#         plot_episode(clutter_map, candidate_positions, action_np, episode)
#         plot_place_probs(place_probs, episode)
#     # Print state node features and edge list
#         print(f"\nEpisode {episode} - Graph State:")
#         print("Node features (x):")
#         print(state.x)
#         print("Edge index:")
#         print(state.edge_index)
#         print("Active Transmitters Configurations:")
#         for (i, ((x, y), (place, power, height, tilt))) in enumerate(zip(candidate_positions, action_np)):
#             if place > 0.5:
#                 print(f"  TX {i}: Pos=({x}, {y}) | Power={power:.2f} | Height={height:.2f} | Downtilt={tilt:.2f}")


# # Plot training progress<wwwwwwwwwwwwwwww
# plt.figure(figsize=(12, 4))
# plt.subplot(1, 3, 1)
# plt.plot(rewards)
# plt.title('Reward per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Reward')

# plt.subplot(1, 3, 2)
# plt.plot(losses)
# plt.title('Loss per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Loss')

# plt.subplot(1, 3, 3)
# plt.plot(num_txs)
# plt.title('Active Transmitters per Episode')
# plt.xlabel('Episode')
# plt.ylabel('Number TX')

# plt.tight_layout()
# plt.show()
    

# torch.save(policy_net.state_dict(), "gnn_policy_weights.pth")
# print("Model saved successfully.")

