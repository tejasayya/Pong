import gym
import numpy as np
import warnings

# Function to train and test the agent with specific hyperparameters
def train_and_test(alpha, gamma, epsilon, train_episodes=1000, test_episodes=100):
    # Initialize environment
    env = gym.make("Taxi-v3", render_mode="ansi")
    state, _ = env.reset()
    
    # Initialize Q-table
    q_table = np.zeros([env.observation_space.n, env.action_space.n])
    
    # Training the agent
    for episode in range(train_episodes):
        state, _ = env.reset()
        done = False
        while not done:
            if np.random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore
            else:
                action = np.argmax(q_table[state])  # Exploit
            
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            q_table[state, action] = q_table[state, action] + alpha * (
                reward + gamma * np.max(q_table[next_state]) - q_table[state, action]
            )
            state = next_state

    # Testing the agent
    total_epochs, total_penalties = 0, 0
    for episode in range(test_episodes):
        state, _ = env.reset()
        epochs, penalties = 0, 0
        done = False
        
        while not done:
            action = np.argmax(q_table[state])
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            if reward == -10:
                penalties += 1
            
            epochs += 1
        
        total_penalties += penalties
        total_epochs += epochs
    
    # Suppress warnings
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    
    # Return performance metrics
    return total_epochs / test_episodes, total_penalties / test_episodes

# Define hyperparameter combinations
alphas = [0.1, 0.5, 0.9]
gammas = [0.5, 0.8, 0.99]
epsilons = [0.1, 0.3, 0.5]

# Grid search over hyperparameters
results = []
for alpha in alphas:
    for gamma in gammas:
        for epsilon in epsilons:
            avg_timesteps, avg_penalties = train_and_test(alpha, gamma, epsilon)
            results.append((alpha, gamma, epsilon, avg_timesteps, avg_penalties))
            print(f"Alpha: {alpha}, Gamma: {gamma}, Epsilon: {epsilon} => Avg Timesteps: {avg_timesteps}, Avg Penalties: {avg_penalties}")

# Display results
print("\nHyperparameter Tuning Results:")
print("Alpha | Gamma | Epsilon | Avg Timesteps | Avg Penalties")
for res in results:
    print(f"{res[0]:<5} | {res[1]:<5} | {res[2]:<7} | {res[3]:<14.2f} | {res[4]:<13.2f}")
