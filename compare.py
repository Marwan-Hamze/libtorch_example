import torch
import torch.nn as nn

import numpy as np

import sys
# Importing the path to the original python project, where several libraries were used to create the models and envs
sys.path.append('/home/yoshidalab/devel/src/reinforcement-learning-hrp')

# Load the models saved with torch.save

actor1 = torch.load("actor_saved.pt")
critic1 = torch.load("critic_saved.pt")

# Load the models saved with torch.jit.trace

actor2 = torch.jit.load("kaleido_standing_actor.pt")
critic2 = torch.jit.load("kaleido_standing_critic.pt")

# Define a 39-element Observation Vector, which is compatible with a simple policy I trained
observation = np.array([1, 0, 0, 0, # quaternion orientation
                       0, 0, 0, # angular velocity
                       0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, # joint positions
                       0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, # Joint Velocities
                       0.5, -0.85, # Clock
                       1.0, 0.0, 0.0, # Mode
                       0.0, 0.0, 0.0])  # Modref: velocities (yaw, root_x, root_y)

# Add a random noise to the observation
# observation += np.random.uniform(-0.02, 0.02, 39)

# Transform the Observation into a Tensor to use it as input to all the models
tensor_obs = torch.tensor(observation, dtype= torch.float) #Specifying dtype=float, because this is how the models' parameters were saved (not double)
# print("TYPE: ", tensor_obs.dtype)

# Putting the models in eval mode and performing a forward pass without gradient computation
actor1.eval()
critic1.eval()

actor2.eval()
critic2.eval()

with torch.no_grad():
    action1 = actor1.forward(tensor_obs)
    value1 = critic1.forward(tensor_obs)
    action2 = actor2.forward(tensor_obs)
    value2 = critic2.forward(tensor_obs)

# Printing the Outputs to see if they match, hence the models are identical to each other
print("Actions from Model 1 (Actor):\n", action1)
print("Actions from Model 2 (Actor):\n", action2)

print("Actions from Model 1 (Critic):\n", value1)
print("Actions from Model 2 (Critic):\n", value2)