import matplotlib.pyplot as plt

def log_episode(episode, total_reward, active_sites, rsrp_map , candidate_positions):
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
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()

def plot_rsrp_coverage(rsrp_map, episode):
    """Visualize RSRP heatmap"""
    plt.figure(figsize=(10, 8))
    plt.imshow(rsrp_map, cmap='viridis', vmin=-120, vmax=-70)
    plt.colorbar(label='RSRP (dBm)')
    plt.title(f"Episode {episode}: RSRP Coverage")
    plt.show()
    # plt.show(block=False)
    # plt.pause(2)
    # plt.close()