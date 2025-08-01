import numpy as np
import torch
import matplotlib.pyplot as plt

from lte_env import LTEPlannerEnv
from gnn_models import GNNPolicy
from lte_utils import build_candidate_graph, compute_rsrp_map, compute_sinr_map

# === Load saved model ===
def load_model(path="gnn_policy_weights.pth"):
    model = GNNPolicy()
    model.load_state_dict(torch.load(path))
    model.eval()
    print(f"✅ Model loaded from {path}")
    return model

# === New clutter map and candidate positions ===
clutter_map = np.ones((500, 500), dtype=int) * 2

# === Define realistic zones ===
clutter_map[50:200, 50:200] = 5      # Urban area
clutter_map[100:160, 100:160] = 6    # Dense urban core in the urban area
clutter_map[300:450, 300:450] = 4    # Suburban area
clutter_map[350:420, 350:420] = 10   # Residential suburb inside suburban area
clutter_map[0:100, 400:500] = 3      # Forest patch
clutter_map[400:500, 0:80] = 1       # Water
clutter_map[200:280, 100:180] = 7    # Industrial zone
clutter_map[150:180, 250:290] = 14   # Dense urban business district
clutter_map[300:330, 100:150] = 13   # Older urban
clutter_map[120:150, 300:350] = 15   # Suburban residential
clutter_map[450:500, 450:500] = 17   # Rural edge
clutter_map[180:220, 400:450] = 18   # Semi-suburban
clutter_map[200:240, 240:280] = 20   # Industrial/Commercial mix

# === Candidate transmitter positions on a grid ===
candidate_positions = [(x, y) for x in range(50, 450, 100) for y in range(50, 450, 100)]

# === Clutter mapping (merged) ===
clutter_to_env = {
    1: 'urban', 2: 'urban', 3: 'suburban', 4: 'urban',
    5: 'rural', 6: 'rural', 7: 'rural', 8: 'rural',
    9: 'rural', 10: 'suburban', 11: 'urban', 12: 'suburban',
    13: 'urban', 14: 'urban', 15: 'suburban', 16: 'urban',
    17: 'rural', 18: 'suburban', 20: 'urban'
}
clutter_loss_table = {
    0: 0, 1: 1, 2: 0, 3: 0.5, 4: 6, 5: 9, 6: 11, 7: 15,
    8: 5, 9: 8, 10: 3, 11: 10, 12: 3, 13: 8, 14: 18,
    15: 5, 16: 7, 17: 2, 18: 4, 20: 1.5
}
clutter_lookup = {
    i: (clutter_to_env.get(i, "urban"), clutter_loss_table.get(i, 5))
    for i in range(0, 21)
}

# === Create environment and policy model ===
env = LTEPlannerEnv(clutter_map, candidate_positions, clutter_lookup, resolution=1.0)
policy_net = load_model("gnn_policy_weights.pth")

# === Run the policy on the new state ===
state = env.reset()
with torch.no_grad():
    logits = policy_net(state)
    place_probs = torch.sigmoid(logits[:, 0])
    configs = torch.sigmoid(logits[:, 1:])
    placements = (place_probs > 0.5).float()
    action = torch.cat([placements.unsqueeze(1), configs], dim=1)

# === Evaluate action in environment ===
_, reward, _, _ = env.step(action.numpy())
print(f"\n📈 Reward on new map: {reward:.2f}")
print(f"📍 Active Transmitters: {int((action[:, 0] > 0.5).sum())} / {len(candidate_positions)}")

# === Plotting ===
def plot_transmitters(clutter_map, candidate_positions, action, title="Transmitter Placement"):
    plt.figure(figsize=(6, 6))
    plt.imshow(clutter_map, cmap='gray', origin='lower')
    for i, (x, y) in enumerate(candidate_positions):
        if action[i, 0] > 0.5:
            plt.plot(x, y, 'ro', label='Active' if i == 0 else "")
        else:
            plt.plot(x, y, 'bo', label='Inactive' if i == 0 else "")
    plt.title(title)
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.show()

# === Visualize placement ===
plot_transmitters(clutter_map, candidate_positions, action, title="Test Episode - Transmitters")

# === Visualize RSRP and SINR maps ===
rsrp_map = compute_rsrp_map(clutter_map, candidate_positions, action.numpy(), clutter_lookup)
sinr_map = compute_sinr_map(rsrp_map, candidate_positions, action.numpy())

plt.figure(figsize=(6, 5))
plt.imshow(rsrp_map, cmap='viridis', origin='lower')
plt.title("RSRP Map (dBm)")
plt.colorbar()
plt.show()

plt.figure(figsize=(6, 5))
plt.imshow(sinr_map, cmap='plasma', origin='lower')
plt.title("SINR Map (dB)")
plt.colorbar()
plt.show()
