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
from multiprocessing import Lock, Process, Queue, current_process, set_start_method
import multiprocessing 
import queue
import time 

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
    network.import_from_hdf5('models/euro_test')
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

def rand_split(n):
    
    rand_list = np.random.random(n-1)
    rand_list.sort()
    rand_list = np.concatenate([[0], rand_list,[1]])
    rand_list = np.diff(rand_list)

    return rand_list


def get_symbol_cuid_pairs_seriel(symbol_map):
    symbol_cuid_pairs = dict(
                (symbol, ComponentUID(var_weakref()))
                for symbol, var_weakref in symbol_map.bySymbol.items())
    return symbol_cuid_pairs

def get_symbol_cuid_pairs(symbol_map,n_jobs=4):

    # Generate dict with var number and reference
    symbol_refs = dict((symbol, var_weakref())
                    for symbol, var_weakref in symbol_map.bySymbol.items())

    # Split this dict in to n_jobs chuncks
    symbol_refs_lists = []
    idx = np.linspace(0,len(symbol_refs),n_jobs+1)
    for i in range(n_jobs):
        i_start = int(idx[i])
        i_end = int(idx[i+1])
        symbol_refs_lists.append(dict(list(symbol_refs.items())[i_start:i_end]))

    # Define job function to get variable names 
    def job(ref_lst,q_done):
        d = dict()
        for symbol, var_weakref in ref_lst.items():
            d[symbol]=ComponentUID(var_weakref) 
        q_done.put(d,block=True)
        return

    # Start a queue for done jobs 
    q_done = Queue()
    # Start n_proceses 
    processes = []
    for ref_lst in symbol_refs_lists:
        p = Process(target=job, args=(ref_lst,q_done))
        processes.append(p)
        p.start()
        logger.info('{} started'.format(p.name))
    
    # Wait for proceses to finish 
    logger.info('Waiting for jobs to finish')
    while q_done.qsize() < n_jobs:
        time.sleep(1)

    # Join all sub proceses
    for p in processes:
        logger.info('waiting to join {}'.format(p.name))
        try :
            p.join(1)
        except :
            p.terminate()
            p.join(60)
            logger.info('killed {}'.format(p.name))
        else :
            logger.info('joined {}'.format(p.name))

    # Create symbol cuid pairs dict from part results 
    symbol_cuid_pairs = dict()
    while not q_done.empty():
        symbol_cuid_pairs.update(q_done.get())

    if len(symbol_cuid_pairs) == len(symbol_refs):
        logger.info('symbol cuid pars == symbol refs')
    else :
        logger.warning('symbol cuid pars != symbol refs')

    return symbol_cuid_pairs



def add_MGA_constraint(network,MGA_slack=0.2):
        network.lopf(formulation='angles',solver_name='gurobi',)
        old_objective_value = network.objective
        model = network.model
        MGA_slack = 0.2
        # Add the MGA slack constraint.
        model.mga_constraint = pyomo_env.Constraint(expr=model.objective.expr*1e-9 <= 
                                                ((1 + MGA_slack) * old_objective_value)*1e-9 )
        return network,model

def save_model(network,model,name):
        # Saving model as .lp file
        _, smap_id = model.write(name+".lp",)
        logger.info('saved model .lp file')
        # Creating symbol map, such that variables can be maped back from .lp file to pyomo model
        symbol_map = model.solutions.symbol_map[smap_id]
        # parallel processing - only work on linux
        #symbol_cuid_pairs = get_symbol_cuid_pairs(symbol_map,n_jobs=os.cpu_count())
        # seriel processing
        symbol_cuid_pairs = get_symbol_cuid_pairs_seriel(symbol_map)

        # Pickeling variable pairs 
        with open(name+'.pickle', 'wb') as handle:
            pickle.dump(symbol_cuid_pairs, handle, protocol=pickle.HIGHEST_PROTOCOL)  
        logger.info('saved var_pairs as pickle')


#%%

mode = 'sampling'
if __name__ == '__main__':
    #multiprocessing.set_start_method('spawn')
    logger = setup_logging()

    try :
        logger.info(sys.argv[1] )
        Snapshots = int(sys.argv[1] )
    except :
        Snapshots = 2



    logger.info('Using {} snapshots'.format(Snapshots))
    logger.info("{} snapshots".format(Snapshots))

    network = import_network(Snapshots)

    if mode == 'sampling':

        network,model = add_MGA_constraint(network,MGA_slack=0.2)
        save_model(network,model,name='models/large_model')


###############################################################################
    elif mode == 'metamodel':
        try :
            n_rand_points = sys.argv[2]
        except :
            n_rand_points = 30


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


