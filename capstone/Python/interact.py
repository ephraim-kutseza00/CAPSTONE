import gym
import numpy as np
from stable_baselines3 import DQN

# Load the trained model
final_model = DQN.load("q_learning_model")

# Function to interact with the model
def interact_with_model(env, model):
    print("Enter the state variables for the building:")
    indoor_temp = float(input("Indoor Temperature (0-50): "))
    outdoor_temp = float(input("Outdoor Temperature (-30 to 50): "))
    energy_consumption = float(input("Energy Consumption (0-100): "))
    occupancy_status = int(input("Occupancy Status (0 or 1): "))

    # Create the state array
    state = np.array([indoor_temp, outdoor_temp, energy_consumption, occupancy_status])
    state = np.reshape(state, (1, -1))  # Reshape for the model
    
    # Get the model's action
    action, _ = model.predict(state)
    
    # Map the action to a human-readable format
    action_mapping = {0: "Low Thermostat Setting", 1: "Medium Thermostat Setting", 2: "High Thermostat Setting"}
    action_readable = action_mapping.get(action[0], "Unknown Action")
    
    print(f"The recommended action is: {action_readable}")

# Create the environment instance
env = gym.make('EnergyPlus-v0')

# Call the interaction function
interact_with_model(env, final_model)
