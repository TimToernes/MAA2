#%%
# from IPython import get_ipython
#from IPython.display import display, clear_output
import pandas as pd
import numpy as np
#from scipy.spatial import ConvexHull,  Delaunay
#from scipy.interpolate import griddata,interpn
import pypsa
from pypsa.linopt import get_var, linexpr, join_exprs, define_constraints, get_dual, get_con, write_objective
import os 
dir_path = os.path.dirname(os.path.realpath(__file__))
os.chdir(dir_path)
import pyomo.environ as pyomo_env
from pyomo.core import ComponentUID
import pickle
import sys 
import logging

#%%
def setup_logging():
    #logger = logging.getLogger('MAA')
    logging.basicConfig(level=logging.INFO, filename='log.log')
    logger = logging.getLogger(__name__)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)
    return logger



def import_network(Snapshots):
    network = pypsa.Network()
    network.import_from_hdf5('euro_95')
    network.snapshots = network.snapshots[0:Snapshots]
    return network



#%%

def initialize_network(network):
    network.lopf(network.snapshots, 
                solver_name='gurobi',
                solver_options={'LogToConsole':1,
                                            'crossover':0,
                                            #'presolve': 2,
                                            #'NumericFocus' : 3,
                                            'method':2,
                                            'threads':8,
                                            #'NumericFocus' : numeric_focus,
                                            'BarConvTol' : 1.e-6,
                                            'FeasibilityTol' : 1.e-2},
                pyomo=False,
                keep_references=True,
                formulation='kirchhoff',
                solver_dir = os.getcwd()
                ),
    network.old_objective = network.objective
    return network


#%%


def mga_constraint(network, snapshots, options):
    scale = 1e-6
    # This function creates the MGA constraint 
    gen_capital_cost   = linexpr((scale*network.generators.capital_cost,get_var(network, 'Generator', 'p_nom'))).sum()
    gen_marginal_cost  = linexpr((scale*network.generators.marginal_cost,get_var(network, 'Generator', 'p'))).sum().sum()
    #store_capital_cost = linexpr((scale*network.storage_units.capital_cost,get_var(network, 'StorageUnit', 'p_nom'))).sum()
    link_capital_cost  = linexpr((scale*network.links.capital_cost,get_var(network, 'Link', 'p_nom'))).sum()
    # total system cost
    #cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,store_capital_cost,link_capital_cost]))
    cost_scaled = join_exprs(np.array([gen_capital_cost,gen_marginal_cost,link_capital_cost]))
    #cost_scaled = linexpr((scale,cost))
    #cost_increase = cost_scaled[0]+'-'+str(network.old_objective*scale)
    # MGA slack
    if options['mga_slack_type'] == 'percent':
        slack = network.old_objective*options['mga_slack']+network.old_objective
    elif options['mga_slack_type'] == 'fixed':
        slack = options['baseline_cost']*options['mga_slack']+options['baseline_cost']

    define_constraints(network,cost_scaled,'<=',slack*scale,'GlobalConstraint','MGA_constraint')


def mga_objective(network,snapshots,direction,options):
    mga_variables = options['mga_variables']
    expr_list = []
    for i,variable in enumerate(mga_variables):
        if variable == 'transmission':
            expr_list.append(linexpr((direction[i],get_var(network,'Link','p_nom'))).sum())
        if variable == 'co2_emission':
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p').filter(network.generators.index[network.generators.type == 'ocgt']))).sum().sum())
        elif variable == 'H2' or variable == 'battery':
            expr_list.append(linexpr((direction[i],get_var(network,'StorageUnit','p_nom').filter(network.storage_units.index[network.storage_units.carrier == variable]))).sum())
        elif variable == 'wind' or variable == 'solar' or variable == 'ocgt': 
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p_nom').filter(network.generators.index[network.generators.type == variable]))).sum())
        else :
            expr_list.append(linexpr((direction[i],get_var(network,'Generator','p_nom').filter(network.generators.index[network.generators.index == variable]))))


    mga_obj = join_exprs(np.array(expr_list))
    #print(mga_obj)
    write_objective(network, mga_obj)

def extra_functionality(network, snapshots, direction, options):
    mga_constraint(network, snapshots, options)
    mga_objective(network, snapshots, direction,options)


# %%

def evaluate(network,x):

    options = dict(mga_variables=network.generators.index.values, mga_slack_type='percent',mga_slack=0.1)
    direction = x

    network.lopf(network.snapshots,
                                pyomo=False,
                                solver_name='gurobi',
                                solver_options={'LogToConsole':0,
                                                'crossover':0,
                                                #'presolve': 0,
                                                'ObjScale' : 1e6,
                                                #'Aggregate' : 0,
                                                'NumericFocus' : 3,
                                                'method':2,
                                                'threads':4,
                                                'BarConvTol' : 1.e-6,
                                                'FeasibilityTol' : 1.e-2},
                                keep_references=False,
                                skip_objective=True,
                                formulation='kirchhoff',
                                extra_functionality=lambda network, snapshots: extra_functionality(network, snapshots, direction,options))

    y = network.generators.p_nom_opt.values

    return y

#%%
def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0], rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list


      

#%%
__name__ = '__main__'
mode = 'sampling'
if __name__ == '__main__':
    logger = setup_logging()

    mode = 'sampling'
    try :
        logger.info(sys.argv[1] )
        Snapshots = int(sys.argv[1] )
    except :
        Snapshots = 50
    try :
        n_rand_points = sys.argv[2]
    except :
        n_rand_points = 30
    logger.info('Using {} snapshots'.format(Snapshots))
    logger.info("{} random samples, {} snapshots".format(n_rand_points,Snapshots))

    network = import_network(Snapshots)

    if mode == 'sampling':

        #model = pypsa.opf.network_lopf_build_model(network)
        #test = model.write("test.lp")

        network.lopf(formulation='angles',solver_name='gurobi',)
        old_objective_value = network.objective
        model = network.model
        MGA_slack = 0.5
        # Add the MGA slack constraint.
        model.mga_constraint = pyomo_env.Constraint(expr=model.objective.expr <= 
                                                (1 + MGA_slack) * old_objective_value)

        # Saving model as .lp file
        _, smap_id = model.write("model_small.lp",)
        logger.info('saved model .lp file')
        # Creating symbol map, such that variables can be maped back from .lp file to pyomo model
        symbol_map = model.solutions.symbol_map[smap_id]
        tmp_buffer = {} # this makes the process faster
        symbol_cuid_pairs = dict(
                (symbol, ComponentUID(var_weakref()))
                for symbol, var_weakref in symbol_map.bySymbol.items())

        # Pickeling variable pairs 
        with open('model_small_vars.pickle', 'wb') as handle:
            pickle.dump(symbol_cuid_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        logger.info('saved var_pairs as pickle')

    elif mode == 'metamodel':
        network = initialize_network(network)


        # Creating X
        dim = 111
        directions = np.concatenate([np.diag(np.ones(dim)),-np.diag(np.ones(dim))],axis=0)
        for i in range(n_rand_points):
            directions = np.append(directions,[rand_split(111)],axis=0) 

        X = pd.DataFrame(data = directions)
        X = X.sample(frac=1).reset_index(drop=True)
        X.to_csv('X')
        logger.info('Saved X')
        # Creating Y

        Y = pd.DataFrame(columns=range(111))
        for x in X.iterrows():
            Y.loc[len(Y)] = evaluate(x[1].values)

        Y.to_csv('Y')
        logger.info('Saved Y')

        # Training metamodel
        logger.info("Training metamodel")
        import metamodel

# %%

# %%
