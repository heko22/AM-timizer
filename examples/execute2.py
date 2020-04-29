# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:03:02 2020

@author: Henrik Kortum
"""

import sys
#sys.path.append('E:\Daten\DataScience\or\logistic_costfunction\script\AMtimizer')
sys.path.append('/Users/henrikkortum/Documents/Projekte/logistic_costfunction/script/AMtimizer')
import AMtimizer as amt
from ortools.sat.python import cp_model

###################

demandlocation,serviceprovider, inventorylocation, transportationcost_sp,transportationtime_sp, transportationcost_il, transportationtime_il = amt.generate_random_case(8, 10, 5, 8, 5, 9)

model, solver, status = amt.calculate_model(demandlocation,serviceprovider, inventorylocation, transportationcost_sp,transportationtime_sp, transportationcost_il, transportationtime_il)

amt.print_results()

######

demandlocation, serviceprovider, inventorylocation, plot_inventorylocation, plot_serviceprovider = amt.generate_plot_tables() 

print(plot_inventorylocation[plot_inventorylocation['value'] > 0]['value'])
print(plot_serviceprovider[plot_serviceprovider['value'] > 0]['value'])

import plotly.graph_objects as go
import pandas as pd
from plotly.subplots import make_subplots


fig = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "mapbox"}, {"type": "pie"}]],
)

#fig = go.Figure()



### Plot Demandlocations ####
fig.add_trace(go.Scattergeo(
    name = 'Demandlocation (Nachfragestandorte)',
    locationmode = 'ISO-3',
    lon = demandlocation['DL_lng'],
    lat = demandlocation['DL_lat'],
    hoverinfo = 'text',
    text = demandlocation[['DL_city',
                          'Annual demand for spare parts at location i']],
    mode = 'markers',
    marker = dict(
        size = 12,
        color = 'rgb(0, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(0, 0, 0, 0)'
        )
    )))

### Plot Serviceproviders ####
fig.add_trace(go.Scattergeo(
    name = 'AM-Serviceprovider',
    locationmode = 'ISO-3',
    lon = serviceprovider['SP_lng'],
    lat = serviceprovider['SP_lat'],
    hoverinfo = 'text',
    text = serviceprovider[['SP_city',
                           'Max Annual production capacity of an AM machine']],
    mode = 'markers',
    marker = dict(
        size = 12,
        color = 'rgb(0, 255, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 168, 68, 0)'
        )
    )))


### Plot Inventorylocation ####
fig.add_trace(go.Scattergeo(
    name = 'Lagerstandorte',
    locationmode = 'ISO-3',
    lon = inventorylocation['IL_lng'],
    lat = inventorylocation['IL_lat'],
    hoverinfo = 'text',
    text = inventorylocation['IL_city'],
    mode = 'markers',
    marker = dict(
        size = 12,
        color = 'rgb(255, 0, 0)',
        line = dict(
            width = 3,
            color = 'rgba(68, 68, 168, 0)'
        )
    )))

## Plot Delivery Connections ##

plot_inventorylocation_active = plot_inventorylocation[plot_inventorylocation['value']>0].reset_index(drop=True)
plot_serviceprovider_active = plot_serviceprovider[plot_serviceprovider['value']>0].reset_index(drop=True)


if len(plot_inventorylocation_active) > 0:
    for i in range(len(plot_inventorylocation_active)):
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'ISO-3',
                lon = [plot_inventorylocation_active['IL_lng'][i], plot_inventorylocation_active['DL_lng'][i]],
                lat = [plot_inventorylocation_active['IL_lat'][i], plot_inventorylocation_active['DL_lat'][i]],
                mode = 'lines',
                line = dict(width = 1,color = 'red'),
                showlegend=False
                #opacity = float(df_flight_paths_1['cnt'][i]) / float(df_flight_paths_1['cnt'].max()),
            )
        )

if len(plot_serviceprovider_active) > 0: 
    for i in range(len(plot_serviceprovider_active)):
        fig.add_trace(
            go.Scattergeo(
                locationmode = 'ISO-3',
                lon = [plot_serviceprovider_active['SP_lng'][i], plot_serviceprovider_active['DL_lng'][i]],
                lat = [plot_serviceprovider_active['SP_lat'][i], plot_serviceprovider_active['DL_lat'][i]],
                mode = 'lines',
                line = dict(width = 1,color = 'green'),
                showlegend=False
                #opacity = float(df_flight_paths_2['cnt'][i]) / float(df_flight_paths_2['cnt'].max()),
            )
        )

fig.update_layout(
    title_text = 'Optimierte Bezugsplanung für das Ersatzteil Test-1 <br>(Mouseover für Detailinformationen zum jeweiligen Standort)',
    showlegend = True,
    geo = go.layout.Geo(
        scope = 'world',
        projection_type = 'natural earth',
        showland = True,
        landcolor = 'rgb(243, 243, 243)',
        countrycolor = 'rgb(204, 204, 204)',
    ),
)

fig.add_trace(go.Pie(values=[2, 3, 1]),
              row=2, col=1)

fig.show()