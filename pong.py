import gym
import numpy as np
import warnings

# Initialize environment with specified render mode
env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset()  # Extract the state value
print(env.render())  # Use print for ANSI rendering

q_table = np.zeros([env.observation_space.n, env.action_space.n])
alpha = 0.1
gamma = 0.6
epsilon = 0.1

episodes = 1200

for episode in range(episodes):
    state, _ = env.reset()  # Extract the state value
    done = False
    while not done:
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(q_table[state])
            
        next_state, reward, terminated, truncated, info = env.step(action)  # Unpack the tuple
        done = terminated or truncated
        q_table[state, action] = q_table[state, action] + alpha * (
            reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
        )
        state = next_state
    
    if (episode + 1) % 100 == 0:
        print(f"Episode {episode + 1} completed")

print("Training finished!\n")


# now testing

def test_q_learning(env, q_table, episodes):
    total_epochs, total_penalties = 0, 0
    
    for episode in range(episodes):
        state, _ = env.reset()  # Extract the state value
        epochs, penalties, reward = 0, 0, 0
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env.step(action)  # Unpack the tuple
            done = terminated or truncated
            
            if reward == -10:
                penalties += 1
            
            epochs += 1
        
        total_penalties += penalties
        total_epochs += epochs
        
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    print(f"Results after {episodes} episodes:")
    print(f"Average timesteps per episode: {total_epochs / episodes}")
    print(f"Average penalties per episode: {total_penalties / episodes}")


test_q_learning(env, q_table, 100)