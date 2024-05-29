import gym
from stable_baselines3 import DQN
from pyep import eprun, EPProblem, readidf
import numpy as np
from pyswarm import pso
from stable_baselines3.common.evaluation import evaluate_policy
from gym import spaces

class EnergyPlusEnv(gym.Env):
def __init__(self):
        def __init__(self, size):
        self.size = size
        self.action_space = spaces.Discrete(4)  # low, mid, high. off
        self.observation_space = spaces.Box(
                low=np.array([0, -30, 0, 0]),
                high=np.array([50, 50, 100, 1]),
                dtype=np.float32)  # Define observation space: indoor temp, outdoor temp, energy consumption, occupancy status
        self.reset()
        
        print("Action space:", self.action_space)
        print("Observation space:", self.observation_space)

def step(self, action):
        # Implement step function
        # Placeholder implementation for the step function
        # Here, you would integrate with EnergyPlus to perform a simulation step
        new_state = np.array([22, 10, 30, 1])  # Example state values
        reward = -1  # Example reward
        done = False  # Example termination condition
        
        # Print debug information
        print("Action taken:", action)
        print("New state:", new_state)
        print("Reward:", reward)
        print("Done:", done)
        
        # Return the new state, reward, done flag, and additional info
        return new_state, reward, done, {}

def reset(self):
        # Implement reset function
        # Placeholder implementation for the reset function
        # Here, you would reset the EnergyPlus simulation to the initial state
        initial_state = np.array([20, 10, 0, 1])  # Example initial state values
        return initial_state

        # Print debug information
        print("Environment reset")
        print("Initial state:", initial_state)

def render(self, mode='human'):
        # Implement render function
        pass
def close(self):
        # Placeholder implementation for the close function
        pass

# Register the custom environment with Gym
gym.register(id='MixedUseHouse', entry_point='__main__:EnergyPlusEnv')
env = gym.make('MixedUseHouse')

check_env(MixedUseHouse)

# Define the Q-learning model
model = DQN('MlpPolicy', env, verbose=1)

# Train the Q-learning model
model.learn(total_timesteps=10000)

# Save the trained model
model.save("q_learning_model")

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"Mean reward: {mean_reward} +/- {std_reward}")

# Define the fitness function for PSO
def fitness(params):
    learning_rate, gamma = params
    
    # Define the Q-learning model with the given parameters
    model = DQN('MlpPolicy', env, learning_rate=learning_rate, gamma=gamma, verbose=0)
    
    # Train the model
    model.learn(total_timesteps=5000)
    
    # Evaluate the trained model
    mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=10)
    
    # Return the negative mean reward as PSO minimizes the function
    return -mean_reward

# Set the bounds for the parameters: learning rate and gamma
lb = [1e-5, 0.8]  # Lower bounds for learning rate and gamma
ub = [1e-2, 0.99]  # Upper bounds for learning rate and gamma

# Perform PSO to find the optimal parameters
best_params, best_fitness = pso(fitness, lb, ub, swarmsize=10, maxiter=10)

print("Best Parameters Found: Learning Rate = {}, Gamma = {}".format(best_params[0], best_params[1]))
print("Best Fitness (Negative Mean Reward):", best_fitness)

# Train the final Q-learning model using the optimized parameters
optimal_learning_rate, optimal_gamma = best_params
final_model = DQN('MlpPolicy', env, learning_rate=optimal_learning_rate, gamma=optimal_gamma, verbose=1)
final_model.learn(total_timesteps=10000)

# Evaluate the final model
mean_reward, std_reward = evaluate_policy(final_model, env, n_eval_episodes=10)
print(f"Final Model Mean Reward: {mean_reward} +/- {std_reward}")