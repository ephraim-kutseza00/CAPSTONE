# app.py

from flask import Flask, request, jsonify, render_template
from stable_baselines3 import DQN
import numpy as np
import gym

# Create the Flask app
app = Flask(__name__)

# Load the trained model
model = DQN.load("q_learning_model")

# Register the custom environment
class EnergyPlusEnv(gym.Env):
    def __init__(self):
        super(EnergyPlusEnv, self).__init__()
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([0, -30, 0, 0]),
            high=np.array([50, 50, 100, 1]),
            dtype=np.float32
        )

    def step(self, action):
        new_state = np.array([22, 10, 30, 1])
        reward = -1
        done = False
        return new_state, reward, done, {}

    def reset(self):
        initial_state = np.array([20, 10, 0, 1])
        return initial_state

    def render(self, mode='human'):
        pass

    def close(self):
        pass

gym.register(id='EnergyPlus-v0', entry_point='__main__:EnergyPlusEnv')
env = gym.make('EnergyPlus-v0')

# Define the route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define the route for model interaction
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    indoor_temp = data['indoor_temp']
    outdoor_temp = data['outdoor_temp']
    energy_consumption = data['energy_consumption']
    occupancy_status = data['occupancy_status']
    
    state = np.array([indoor_temp, outdoor_temp, energy_consumption, occupancy_status])
    state = np.reshape(state, (1, -1))
    
    action, _ = model.predict(state)
    
    action_mapping = {0: "Low Thermostat Setting", 1: "Medium Thermostat Setting", 2: "High Thermostat Setting"}
    action_readable = action_mapping.get(action[0], "Unknown Action")
    
    return jsonify({'action': action_readable})

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
