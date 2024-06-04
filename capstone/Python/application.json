import tkinter as tk
from tkinter import messagebox
from stable_baselines3 import DQN
import numpy as np
import gym

# Load the trained model
final_model = DQN.load("q_learning_model")

# Function to interact with the model
def interact_with_model(env, model, indoor_temp, outdoor_temp, energy_consumption, occupancy_status):
    # Create the state array
    state = np.array([indoor_temp, outdoor_temp, energy_consumption, occupancy_status])
    state = np.reshape(state, (1, -1))  # Reshape for the model
    
    # Get the model's action
    action, _ = model.predict(state)
    
    # Map the action to a human-readable format
    action_mapping = {0: "Low Thermostat Setting", 1: "Medium Thermostat Setting", 2: "High Thermostat Setting"}
    action_readable = action_mapping.get(action[0], "Unknown Action")
    
    return action_readable

# Create the environment instance
env = gym.make('EnergyPlus-v0')

# Create the GUI application
app = tk.Tk()
app.title("EnergyPlus Q-Learning Model")

# Define and place the labels and entry widgets
tk.Label(app, text="Indoor Temperature (0-50):").grid(row=0, column=0)
indoor_temp_entry = tk.Entry(app)
indoor_temp_entry.grid(row=0, column=1)

tk.Label(app, text="Outdoor Temperature (-30 to 50):").grid(row=1, column=0)
outdoor_temp_entry = tk.Entry(app)
outdoor_temp_entry.grid(row=1, column=1)

tk.Label(app, text="Energy Consumption (0-100):").grid(row=2, column=0)
energy_consumption_entry = tk.Entry(app)
energy_consumption_entry.grid(row=2, column=1)

tk.Label(app, text="Occupancy Status (0 or 1):").grid(row=3, column=0)
occupancy_status_entry = tk.Entry(app)
occupancy_status_entry.grid(row=3, column=1)

# Define the action when the button is clicked
def on_submit():
    try:
        indoor_temp = float(indoor_temp_entry.get())
        outdoor_temp = float(outdoor_temp_entry.get())
        energy_consumption = float(energy_consumption_entry.get())
        occupancy_status = int(occupancy_status_entry.get())
        
        action = interact_with_model(env, final_model, indoor_temp, outdoor_temp, energy_consumption, occupancy_status)
        messagebox.showinfo("Recommended Action", f"The recommended action is: {action}")
    except ValueError:
        messagebox.showerror("Invalid Input", "Please enter valid input values.")

# Define and place the submit button
submit_button = tk.Button(app, text="Get Recommended Action", command=on_submit)
submit_button.grid(row=4, columnspan=2)

# Run the application
app.mainloop()
