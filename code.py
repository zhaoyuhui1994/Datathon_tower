#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

import pyarrow as pa
import pyarrow.parquet as pq


import matplotlib.pylab as plt
import igraph as ig
import sys
import os
import geopandas as gpd


# In[2]:


files=['./fhvhv_tripdata_2019-04.parquet',
       './fhvhv_tripdata_2020-04.parquet',
       './fhvhv_tripdata_2021-04.parquet',
       './fhvhv_tripdata_2022-04.parquet',
       './fhvhv_tripdata_2023-04.parquet']


# In[9]:


'''Step 1-1: Taxi orders during commuting hours--------------------------------------------------------'''

for f in files:
    print(f)
    df = pd.read_parquet(f,engine="pyarrow")[['pickup_datetime','dropoff_datetime','trip_miles','PULocationID','DOLocationID']]
    pt=[]
    dt=[]
    week=[]
    for i in range(len(df)):
        pt.append(df['pickup_datetime'][i].hour)
        dt.append(df['dropoff_datetime'][i].hour)
        week.append(df['pickup_datetime'][i].weekday())
    df['pickHour']=pt
    df['dropHour']=dt
    df['weekday']=week
    df_work=df.query('(weekday<6 and weekday>0) and ((pickHour>5 and pickHour<10) or (dropHour>16 and dropHour<22))')  #weekday commuting
    df_work.to_parquet("./trip_commuting"+f[17:24]+".parquet",engine="pyarrow")
    
    
'''Step 1-2: Weekend entertainment trip'''
for f in files:
    print(f)
    df = pd.read_parquet(f,engine="pyarrow")[['pickup_datetime','dropoff_datetime','trip_miles','PULocationID','DOLocationID']]
    pt=[]
    dt=[]
    week=[]
    for i in range(len(df)):
        pt.append(df['pickup_datetime'][i].hour)
        dt.append(df['dropoff_datetime'][i].hour)
        week.append(df['pickup_datetime'][i].weekday())
    df['pickHour']=pt
    df['dropHour']=dt
    df['weekday']=week
    df_unwork=df.query('weekday==6 or weekday==0')  #weekend
    df_unwork.to_parquet("./trip_weekend"+f[17:24]+".parquet",engine="pyarrow")


# In[25]:


'''Step 2: OD network----------------------------------------------------------------------------'''
# weekday
for f in files:
    print(f)
    df = pd.read_parquet('./trip_commuting'+f[17:24]+'.parquet',engine="pyarrow")
    df_od=df.groupby(['PULocationID','DOLocationID']).size().reset_index(name='count')
    df_od.to_csv('./od_'+f[17:24]+'.ncol', header=None, index=False, sep=" ")
#weekend   
for f in files:
    print(f)
    df = pd.read_parquet('./trip_weekend'+f[17:24]+'.parquet',engine="pyarrow")
    df_od=df.groupby(['PULocationID','DOLocationID']).size().reset_index(name='count')
    df_od.to_csv('./Weekend_od_'+f[17:24]+'.ncol', header=None, index=False, sep=" ")


# In[27]:


'''Step 3-1: community detection-----------------------------------------------------------------'''

def infomap_save(G,community,pathout):
    df = pd.DataFrame(columns=['community_ID','LocationID'])
    comm_id = []
    grid_id = []
    for index,comm in enumerate(community):
        for item in comm:
            name=G.vs[item]['name']
            comm_id.append(index)
            grid_id.append(name)
    df['community_ID'] = list(comm_id)
    df['LocationID'] = list(grid_id)
    df2=df.groupby('community_ID').size().reset_index(name='count')
    df3=df2.sort_values('count',ascending=False).reset_index(drop=True)
    df3['N_community_ID']=range(len(df3))
    dictionary = dict(zip(df3['community_ID'].values, df3['N_community_ID'].values))
    df['community_ID']= np.vectorize(dictionary.get)(df['community_ID'].values)
    df=df.sort_values('community_ID',ascending=True).reset_index(drop=True)
    df_temp=df[['community_ID','LocationID']]
    df_temp.to_csv(pathout,index=False)  
for f in files:
    G = ig.Graph.Read_Ncol('./od_'+f[17:24]+'.ncol',names=True,weights=True,directed=True)
    #G = ig.Graph.Read_Ncol('./Weekend_od_'+f[17:24]+'.ncol',names=True,weights=True,directed=True)
    weights = G.es["weight"]
    community = G.community_infomap(edge_weights=weights,vertex_weights=None,trials=20)
    infomap_save(G,community,'./shp/'+f[17:24]+'.csv') 

'''Step 3-2 output community result using shapefile-----------------------------------------------------------'''
for f in files:
    df_temp=pd.read_csv('./shp/'+f[17:24]+'.csv')
    df_out=gpd.read_file('./NYCTaxiZones/geo_export_3f8c3d91-09a8-4acc-aaf3-484abc4900c9.shp')
    df_out['community_ID'] = -1
    index=df_temp['community_ID'][0]
    flag=0
    for j in range(len(df_temp)):
        n=df_temp['LocationID'][j]
        index_list = df_out.query('location_i==@n').index.tolist()
        if(index != df_temp['community_ID'][j]):
            flag=flag+1 
            index=df_temp['community_ID'][j]
        df_out.loc[index_list, 'community_ID'] = flag

    df_out.to_file('./shp/comm_'+f[17:24]+'.shp',driver='ESRI Shapefile',encoding='utf-8')
    #df_out.to_file('./shp/Weekend_comm_'+f[17:24]+'.shp',driver='ESRI Shapefile',encoding='utf-8')


# In[28]:


'''Step 4 Shift in overall mileage covered'''
for f in files:
    df = pd.read_parquet('./trip_commuting'+f[17:24]+'.parquet',engine="pyarrow")
    df_out=gpd.read_file('./shp/comm_'+f[17:24]+'.shp')
    #temp=list(df.groupby(['PULocationID'])['trip_miles'].sum())
    temp=[]
    for i in df_out['location_i']:
        temp.append(int(df.query('PULocationID==@i')['trip_miles'].sum()))
    df_out['outTripMile'] = temp
    df_out.to_file('./shp/comm_'+f[17:24]+'.shp',driver='ESRI Shapefile',encoding='utf-8')


# In[30]:


'''Step 5 Shift in overall trip numbers'''
for f in files:
    df = pd.read_parquet('./trip_commuting'+f[17:24]+'.parquet',engine="pyarrow")
    df_out=gpd.read_file('./shp/comm_'+f[17:24]+'.shp')
    temp=[]
    for i in df_out['location_i']:
        temp.append(len(df.query('DOLocationID==@i')))
    df_out['visitNum'] = temp
    df_out.to_file('./shp/comm_'+f[17:24]+'.shp',driver='ESRI Shapefile',encoding='utf-8')