#%%
import pypsa
from pyomo.core import ComponentUID
import numpy as np
import pyomo.environ as pyomo_env
import pickle 
# %% Create pypsa model 
network = pypsa.Network()

n_snapshots = 100
network.set_snapshots(range(n_snapshots))


network.add("Carrier",'ocgt',co2_emissions=1)
network.add("Carrier","onwind")
network.add("Carrier","offwind")
network.add("Carrier","wind")
network.add("Carrier","solar")

#add three buses
n_buses = 10
for i in range(n_buses):
    network.add("Bus","My bus {}".format(i),
                x=i,y=i%2+1)

#add three lines in a ring
for i in range(n_buses):
    network.add("Link","My line {}".format(i),
                bus0="My bus {}".format(i),
                bus1="My bus {}".format((i+1)%n_buses),
                p_nom=0,
                p_nom_extendable=True,
                length=1,
                capital_cost=np.random.rand()+0.1,)

# Add generators 
for i in range(n_buses):
    for gen in ['solar','wind','ocgt']:
        network.add("Generator","{}{}".format(gen,i),
                    bus="My bus {}".format(i),
                    p_nom_max=10,
                    type = gen,
                    carrier = gen,
                    p_max_pu = 0 if gen == 'solar' and i%2 ==0 else 1,
                    #p_max_pu = float((np.random.rand()>0.5)*0.8),
                    p_nom = 0,
                    capital_cost = np.random.rand()+0.1,
                    marginal_cost = np.random.rand()*0.1,
                    p_nom_extendable	= True)    


    network.add("Load","My load {}".format(i),
            bus="My bus {}".format(i),
            p_set=np.random.rand(n_snapshots)*10)



network.add("GlobalConstraint","co2_limit",
              sense="<=",
              carrier_attribute="co2_emissions",
              constant=50)
# %% Solve pyomo model and add MGA constraint

#network.lopf(formulation="cycles")
#model = pypsa.opf.network_lopf_build_model(network)
network.lopf()
old_objective_value = network.objective
model = network.model
#%%
MGA_slack = 0.1
# Add the MGA slack constraint.
model.mga_constraint = pyomo_env.Constraint(expr=model.objective.expr <= 
                                        (1 + MGA_slack) * old_objective_value)

# Saving model as .lp file
_, smap_id = model.write("test.lp",)
# Creating symbol map, such that variables can be maped back from .lp file to pyomo model
symbol_map = model.solutions.symbol_map[smap_id]
#%%
tmp_buffer = {} # this makes the process faster
symbol_cuid_pairs = dict(
        (symbol, ComponentUID(var_weakref(), cuid_buffer=tmp_buffer))
        for symbol, var_weakref in symbol_map.bySymbol.items())


# %% Pickeling variable pairs 
with open('var_pairs.pickle', 'wb') as handle:
    pickle.dump(symbol_cuid_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)

# %%
