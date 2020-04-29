# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:12:53 2019

@author: Henrik Kortum
"""

# Import of the needed packages
from ortools.sat.python import cp_model
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def generate_random_case(min_number_dl: int, 
                         max_number_dl: int, 
                         min_number_sp: int, 
                         max_number_sp: int,
                         min_number_il: int,
                         max_number_il: int):
    """Generates a random case, containing a random number of demandlocations with randomized parameters for demand locations, service providers and inventory locations.
    returns the following objects as pandas dataframe: demandlocation, serviceprovider, inventory location, transportation costs for service provider, transportation cost für service provider,
    transportation cost for inventory location and transportation time for inventory location
    """
    
    # generate random number of dl sp il
    random_number_dl = np.random.randint(min_number_dl, max_number_dl)
    random_number_sp = np.random.randint(min_number_sp, max_number_sp)
    random_number_il = np.random.randint(min_number_il, max_number_il)
    
    demandlocation = pd.DataFrame(index=np.arange(random_number_dl)) 
    serviceprovider = pd.DataFrame(index=np.arange(random_number_sp))
    inventorylocation = pd.DataFrame(index=np.arange(random_number_il))
    
    # generate and randomy fill the columns for all demandlocations
    demandlocation['ID'] = demandlocation.index+1
    demandlocation['Annual demand for spare parts at location i'] = np.random.randint(10000, 20001, demandlocation.shape[0])
    demandlocation['Downtime costs per hour at location i'] = np.random.randint(10, 60, demandlocation.shape[0])
    demandlocation['Min Annual production capacity of an AM machine'] = np.random.randint(0, 1, demandlocation.shape[0])
    demandlocation['Max Annual production capacity of an AM machine'] = np.random.randint(1000, 5000, demandlocation.shape[0])
    demandlocation['Lifespan of an AM machine in location i '] = np.random.randint(1, 2, demandlocation.shape[0])
    demandlocation['Production time at location i per spare part [in hours]'] = np.random.randint(1, 2, demandlocation.shape[0])
    demandlocation['Acquisition costs for an AM machine in location i including IT infrastructure'] = np.random.randint(1000000, 3000000, demandlocation.shape[0])
    demandlocation['Production costs in i per spare part below step mu'] = np.random.randint(20, 30, demandlocation.shape[0])
    demandlocation['Production costs in i per spare part above step mu'] = demandlocation['Production costs in i per spare part below step mu'] -np.random.randint(0,3,demandlocation.shape[0])

    # generate and randomy fill the columns for all service providers     
    serviceprovider['ID'] = serviceprovider.index+1
    serviceprovider['Min Annual production capacity of an AM machine'] = np.random.randint(0, 1, serviceprovider.shape[0])
    serviceprovider['Max Annual production capacity of an AM machine'] = np.random.randint(5000, 20000, serviceprovider.shape[0])
    serviceprovider['Lifespan of an AM machine in location j'] = np.random.randint(1, 2, serviceprovider.shape[0])
    serviceprovider['Production time at location j per spare part [in hours]'] = np.random.randint(1, 2, serviceprovider.shape[0])
    serviceprovider['Acquisition costs for an AM machine in location j including IT infrastructure'] = np.random.randint(0, 1, serviceprovider.shape[0])
    serviceprovider['Production costs in j per spare part above step mu'] = np.random.randint(10, 30, serviceprovider.shape[0])

    # generate and randomy fill the columns for all inventory locations
    inventorylocation['ID'] = inventorylocation.index+1
    inventorylocation['Production costs in k per spare part'] = np.random.randint(50,900, inventorylocation.shape[0])
    inventorylocation['Inventory carrying costs in k per spare part'] = np.random.randint(5, 200, inventorylocation.shape[0])
    
    transportationtime_sp = pd.DataFrame(index=np.arange(random_number_dl))
    transportationcost_sp = pd.DataFrame(index=np.arange(random_number_dl))
    transportationtime_sp['ID'] = transportationtime_sp.index+1
    transportationcost_sp['ID'] = transportationcost_sp.index+1
    transportationtime_sp['ID'] = transportationtime_sp['ID'].apply(lambda x: 'DL'+str(x))
    transportationcost_sp['ID'] = transportationcost_sp['ID'].apply(lambda x: 'DL'+str(x))
    
    transportationtime_il = pd.DataFrame(index=np.arange(random_number_dl))
    transportationcost_il = pd.DataFrame(index=np.arange(random_number_dl))

    transportationtime_il['ID'] = transportationtime_il.index+1
    transportationcost_il['ID'] = transportationcost_il.index+1
    transportationtime_il['ID'] = transportationtime_il['ID'].apply(lambda x: 'DL'+str(x))
    transportationcost_il['ID'] = transportationcost_il['ID'].apply(lambda x: 'DL'+str(x))


    for sp in range(1, len(serviceprovider)+1):
        transportationtime_sp['SP'+str(sp)] = np.random.randint(1, 3, transportationtime_sp.shape[0], dtype='int64')
        transportationcost_sp['SP'+str(sp)] = np.random.randint(1, 3, transportationcost_sp.shape[0], dtype='int64')

    for il in range(1, len(inventorylocation)+1):
        transportationtime_il['IL'+str(il)] = np.random.randint(1, 5, transportationtime_il.shape[0], dtype='int64')
        transportationcost_il['IL'+str(il)] = np.random.randint(1, 10, transportationcost_il.shape[0], dtype='int64')
                
        
    return(demandlocation,serviceprovider, inventorylocation, transportationcost_sp,transportationtime_sp, transportationcost_il, transportationtime_il)

def calculate_model(demandlocation,
                    serviceprovider, 
                    inventorylocation, 
                    transportationcost_sp,
                    transportationtime_sp,
                    transportationcost_il,
                    transportationtime_il):
    """Calculates the model and optimzises the production volume for each dl,sp and il.
    Returns the model itself, the solver and the status.
    """
    model = cp_model.CpModel()

    # Generating the Variable for optimized production volume demandlocation
    demandlocation['production volume'] = demandlocation.apply(lambda x: model.NewIntVar(lb = int(x['Min Annual production capacity of an AM machine']),
                                                                                         ub = int(x['Max Annual production capacity of an AM machine']), 
                                                                                         name='opt_'+str(x['ID'])), 
                                                                                         axis=1)

    # Generating the Dummy Variable wether a Machine is placed at a demand location
    demandlocation['dummy'] = demandlocation.apply(lambda x: model.NewBoolVar('dummy_'+str(x['ID'])), 
                                                                               axis=1)

    # Generating the Variable for optimized production volume service provider
    ## Create new Column for Each DL and genrate a Variable
    prod_vol_idx = []
    for i in range(0, len(demandlocation)):
        serviceprovider['production volume for DL ' +str(i)] = serviceprovider.apply(lambda x: model.NewIntVar(lb = int(x['Min Annual production capacity of an AM machine']),
                                                                                                               ub = int(x['Max Annual production capacity of an AM machine']), 
                                                                                                               name='opt_'+str(x['ID'])+'_for_' +str(i)), 
                                                                                                               axis=1)
        prod_vol_idx.append(serviceprovider.columns.get_loc('production volume for DL ' +str(i)))

    # Generating the Dummy Variable wether a Machine is placed at a demand location
    serviceprovider['dummy'] = serviceprovider.apply(lambda x: model.NewBoolVar('dummy_'+str(x['ID'])), 
                                                                        axis=1)
    
    
    # Gerate the variable for optimized volume inventory location
    for i in range(0, len(demandlocation)):
        inventorylocation['production volume for DL ' +str(i)] = inventorylocation.apply(lambda x: model.NewIntVar(lb = 0, 
                                                                                                                 ub = 100000, 
                                                                                                                 name='opt_'+str(x['ID'])+'_for_' +str(i)), 
                                                                                                                 axis=1)


    #demandfullfillment
    demandlocation['demand_over_lifetime'] = demandlocation['Annual demand for spare parts at location i'] * demandlocation['Lifespan of an AM machine in location i ']

    ## Add Restriction to model
    ### Annual demand = annual production on machine at DL + production for DL at all Service providers
    for entry in range(0, len(demandlocation)):
        model.Add(demandlocation['production volume'].iloc[entry] + 
                  sum(serviceprovider['production volume for DL '+ str(entry)]) +
                  sum(inventorylocation['production volume for DL ' + str(entry)])
                  == demandlocation['demand_over_lifetime'].iloc[entry])

    # downtime
    demandlocation['downtime'] = 0
    ## Add Column for Downtime at Demandlocation
    for entry in range (0, len(demandlocation)):
        demandlocation['downtime'].iloc[entry] = sum([(demandlocation['production volume'].iloc[entry] * demandlocation['Production time at location i per spare part [in hours]'].iloc[entry]) , 
                                                      sum(serviceprovider['production volume for DL '+ str(entry)] * transportationtime_sp.drop('ID', axis=1).iloc[entry].values),
                                                      sum(serviceprovider['production volume for DL '+ str(entry)] * serviceprovider['Production time at location j per spare part [in hours]']),
                                                      sum(inventorylocation['production volume for DL '+ str(entry)] * transportationtime_il.drop('ID', axis=1).iloc[entry].values)
                                                     ])

    ## Kapazität der AM Maschine an Demandlocation i über die Gesamte Lebens
    demandlocation['AM Capacity'] = demandlocation['dummy'] * demandlocation['Max Annual production capacity of an AM machine'] * demandlocation['Lifespan of an AM machine in location i ']

    ## Kapazität der AM Maschine bei Serviceprovider j über die Gesamte Lebens
    serviceprovider['AM Capacity'] = serviceprovider['dummy'] * serviceprovider['Max Annual production capacity of an AM machine'] * serviceprovider['Lifespan of an AM machine in location j']

    ## Add Restriction to model
    ### Annual demand = annual production on machine at DL + production for DL at all Service providers
    for entry in range(0, len(serviceprovider)):
        model.Add(sum(serviceprovider.iloc[entry, prod_vol_idx]) 
                  <= serviceprovider['AM Capacity'].iloc[entry])

    for dl in range(0,len(demandlocation)):
        model.Add(demandlocation['production volume'].iloc[dl] == 0).OnlyEnforceIf(demandlocation['dummy'].iloc[dl].Not())
        model.Add(demandlocation['production volume'].iloc[dl] > 0).OnlyEnforceIf((demandlocation['dummy'].iloc[dl]))
        #print('added dummy logic for demandlocation ' +str(dl))

    for sp in range(0,len(serviceprovider)):
        for dl in range(0,len(demandlocation)):
            model.Add(serviceprovider['production volume for DL ' + str(dl)].iloc[sp] == 0).OnlyEnforceIf(serviceprovider['dummy'].iloc[sp].Not())
            model.Add(serviceprovider['production volume for DL ' + str(dl)].iloc[sp] >= 0).OnlyEnforceIf((serviceprovider['dummy'].iloc[sp]))
            #print('added dummy logic for DL ' +str(dl) + ' and SP ' + str(sp))

    #Demandlocation Costs
    demandlocation['downtime_costs'] = demandlocation['downtime'] * demandlocation['Downtime costs per hour at location i'] 
    demandlocation['acquisition_costs'] = demandlocation['dummy'] * demandlocation['Acquisition costs for an AM machine in location i including IT infrastructure']
    demandlocation['production_costs'] = demandlocation['production volume'] * demandlocation['Production costs in i per spare part above step mu']

    #Serviceprovider Costs
    serviceprovider['acquisition_costs'] = serviceprovider['dummy'] * serviceprovider['Acquisition costs for an AM machine in location j including IT infrastructure']

    sp_prod_cost_idx = []
    for dl in range(0,len(demandlocation)):
        serviceprovider['production_costs_' + str(dl)] = serviceprovider['Production costs in j per spare part above step mu'] * serviceprovider['production volume for DL ' +str(dl)]
        sp_prod_cost_idx.append(serviceprovider.columns.get_loc('production_costs_' + str(dl)))

    sp_trans_cost_idx = []    
    for dl in range(0,len(demandlocation)):
        serviceprovider['transportation_costs_' + str(dl)] = serviceprovider['production volume for DL '+ str(dl)] * transportationcost_sp.drop('ID', axis=1).iloc[dl].values
        sp_trans_cost_idx.append(serviceprovider.columns.get_loc('transportation_costs_' + str(dl)))

    ## Inventory Location Costs
    for dl in range(0,len(demandlocation)):
        inventorylocation['production_costs_' + str(dl)] = inventorylocation['Production costs in k per spare part'] * inventorylocation['production volume for DL ' +str(dl)]
        inventorylocation['inventory_carrying_costs_' + str(dl)] = inventorylocation['Inventory carrying costs in k per spare part'] * inventorylocation['production volume for DL ' +str(dl)]
        inventorylocation['transportation_costs_' + str(dl)] = inventorylocation['production volume for DL '+ str(dl)] * transportationcost_il.drop('ID', axis=1).iloc[dl].values

    ## Cost Function
    total_costs_dl = sum([demandlocation['downtime_costs'], demandlocation['acquisition_costs'], demandlocation['production_costs']])
    
    total_costs_sp = sum([serviceprovider['acquisition_costs'],
       sum(sum(serviceprovider.iloc[:, sp_prod_cost_idx].values)),
       sum(sum(serviceprovider.iloc[:, sp_trans_cost_idx].values))])
    
    total_costs_il = sum([sum(inventorylocation.filter(regex='production_cost').values),
                      sum(inventorylocation.filter(regex='inventory_carrying_cost').values),
                      sum(inventorylocation.filter(regex='transportation_cost').values)])

    total_cost = sum(total_costs_dl.values) + sum(total_costs_sp.values) + sum(total_costs_il)

    model.Minimize(total_cost)

    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    return(model, solver, status)

def sensitivity_calculator(df, column:str, factor:int, increase:bool):
    if increase==True:
        df[column] = df[column] + factor
    else:
        df[column] = df[column] + factor
    return(df)


demandlocation, serviceprovider, inventorylocation, cost_sp, time_sp, cost_il, time_il = generate_random_case(5, 10, 10, 12, 10, 16)

model, solver, status = calculate_model(demandlocation, serviceprovider, inventorylocation, cost_sp, time_sp, cost_il, time_il)

def cost_sensitivity_calculator_alt():
    relevant_columns = ['Downtime costs per hour at location i', 
                        'Production time at location i per spare part [in hours]',
                        'Production costs in i per spare part below step mu',
                        'Production costs in i per spare part above step mu',
                        'production_costs']
    model, solver, status = calculate_model(demandlocation, serviceprovider, inventorylocation, cost_sp, time_sp, cost_il, time_il)
    base_cost_for_case = solver.ObjectiveValue()
    
    def sensitivity_calculator(df, column:str, factor:int, increase:bool):
        if increase==True:
            df[column] = df[column] + factor
        else:
            df[column] = df[column] + factor
        return(df)

    for col in relevant_columns:
        df = sensitivity_calculator(demandlocation, col, 1, True)
        model, solver, status = calculate_model(df, serviceprovider, inventorylocation, cost_sp, time_sp, cost_il, time_il)
        print('Parameter: ' + str(col) + 'Kostenveränderung = %i' % (solver.ObjectiveValue() -base_cost_for_case))
    
def print_results():
    if status == cp_model.OPTIMAL:
        for dl in range (0,len(demandlocation)):
            print('optimale Menge an Demandlocation  '+ str(dl) + ' = %i' % solver.Value(demandlocation['production volume'].iloc[dl]))
            #print('Wert der Dummy Variable an Demandlocation ' + str(dl) + ' = %i' % solver.Value(demandlocation['dummy'].iloc[dl]))
            
        for dl in range(0,len(demandlocation)):            
            for sp in range(0,len(serviceprovider)):
                if solver.Value(serviceprovider['production volume for DL ' + str(dl)].iloc[sp]) > 0:
                    print('optimale Menge bei Service Provider '+ str(sp) + ' für Demand Location ' + str(dl) + ' = %i' % solver.Value(serviceprovider['production volume for DL ' + str(dl)].iloc[sp]))
                 #  print('Wert der Dummy Variable an Demandlocation ' + str(sp) + ' = %i' % solver.Value(serviceprovider['dummy'].iloc[sp]))
                    
        for dl in range(0,len(demandlocation)):    
            for il in range(0,len(inventorylocation)):
                if solver.Value(inventorylocation['production volume for DL ' + str(dl)].iloc[il]) > 0:
                    print('optimale Menge bei Inventory Location '+ str(il) + ' für Demand Location ' + str(dl) + ' = %i' % solver.Value(inventorylocation['production volume for DL ' + str(dl)].iloc[il]))
                    #print('Wert der Dummy Variable an Demandlocation ' + str(il) + ' = %i' % solver.Value(inventorylocation['dummy'].iloc[il]))
                    
def cost_sensitivity_calculator_neu(demandlocation,
                                    
                                    serviceprovider, 
                                    inventorylocation, 
                                    transportationcost_sp,
                                    transportationtime_sp,
                                    transportationcost_il,
                                    transportationtime_il, 
                                    idx_sp=2,):
    sensitivity_df = pd.DataFrame(columns=['independend_variable', 'value', 'rate_of_change', 'costs'])
    serviceprovider_intern = serviceprovider
    decreasefactor = 0
    boundary = serviceprovider_intern['Production costs in j per spare part above step mu'].iloc[idx_sp].copy()
    while decreasefactor <= boundary:
        decreasefactor += 1
        serviceprovider_intern['Production costs in j per spare part above step mu'].iloc[idx_sp] = boundary.copy() - decreasefactor
        new_row = pd.DataFrame([['cost_per_spare_part', 
                   serviceprovider_intern['Production costs in j per spare part above step mu'].iloc[idx_sp],
                   0,
                   'later']], columns=['independend_variable', 'value', 'rate_of_change', 'costs'])
        model, solver, status = calculate_model(demandlocation,
                        serviceprovider_intern, 
                        inventorylocation, 
                        transportationcost_sp,
                        transportationtime_sp,
                        transportationcost_il,
                        transportationtime_il)
        new_row['costs'] = solver.ObjectiveValue()
        sensitivity_df = sensitivity_df.append(new_row, ignore_index=True)
    return(sensitivity_df)


def generate_plot_tables():
    cities = pd.read_csv(r'/Users/henrikkortum/Downloads/simplemaps_worldcities_basicv1/worldcities.csv')
    
    demandlocation[['DL_city', 'DL_lat', 'DL_lng']] = cities[['city', 'lat', 'lng']].sample(len(demandlocation)).reset_index(drop=True)
    serviceprovider[['SP_city', 'SP_lat', 'SP_lng']] = cities[['city', 'lat', 'lng']].sample(len(serviceprovider)).reset_index(drop=True)
    inventorylocation[['IL_city', 'IL_lat', 'IL_lng']] = cities[['city', 'lat', 'lng']].sample(len(inventorylocation)).reset_index(drop=True)
    
    for dl in range(0,len(demandlocation)):
        inventorylocation['production volume for DL ' + str(dl)] = inventorylocation['production volume for DL ' + str(dl)].apply(lambda x: solver.Value(x))
    melted_inventorylocation = pd.concat([inventorylocation[['ID','IL_city', 'IL_lat', 'IL_lng']],inventorylocation.filter(regex='production volume for ')], axis=1)
    melted_inventorylocation = melted_inventorylocation.melt(id_vars= ['ID', 'IL_city', 'IL_lat', 'IL_lng'],
                 var_name='DL',
                 value_name ='value')
    melted_inventorylocation['DL'] = melted_inventorylocation['DL'].apply(lambda x: x.replace('production volume for DL ', ''))
    melted_inventorylocation['DL'] = melted_inventorylocation['DL'].astype('int')+1
    melted_inventorylocation = pd.merge(melted_inventorylocation, demandlocation[['ID','DL_city', 'DL_lat', 'DL_lng']], 
                                                                      how = 'left', 
                                                                      left_on = 'DL',
                                                                      right_on = 'ID')
    
    
    for dl in range(0,len(demandlocation)):
        serviceprovider['production volume for DL ' + str(dl)] = serviceprovider['production volume for DL ' + str(dl)].apply(lambda x: solver.Value(x))
    melted_serviceprovider = pd.concat([serviceprovider[['ID','SP_city', 'SP_lat', 'SP_lng']],serviceprovider.filter(regex='production volume for ')], axis=1)
    melted_serviceprovider = melted_serviceprovider.melt(id_vars= ['ID', 'SP_city', 'SP_lat', 'SP_lng'],
                 var_name='DL',
                 value_name ='value')
    melted_serviceprovider['DL'] = melted_serviceprovider['DL'].apply(lambda x: x.replace('production volume for DL ', ''))
    melted_serviceprovider['DL'] = melted_serviceprovider['DL'].astype('int')
    melted_serviceprovider = pd.merge(melted_serviceprovider, demandlocation[['ID','DL_city', 'DL_lat', 'DL_lng']], 
                                                                      how = 'left', 
                                                                      left_on = 'DL',
                                                                      right_on = 'ID')
    melted_serviceprovider = melted_serviceprovider.dropna(axis=0)
    return demandlocation, serviceprovider, inventorylocation, melted_inventorylocation, melted_serviceprovider