import sys
sys.path.append('..')
import numpy as np
import pandas as pd
import src as odyn

model = odyn.OpinionNetworkModel(probabilities = [.45,.1,.45])
model.populate_model(num_agents = 1000)
sim = odyn.NetworkSimulation()
sim.run_simulation(model = model,
           store_results = True
          )
sim.plot_simulation_results()