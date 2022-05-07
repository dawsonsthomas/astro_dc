import numpy as np
import pandas as pd
import sys

###### For Catch/Machine learning
import graphtools
from CATCH import catch
import matplotlib.pyplot as plt
import scprep

import phate
import sklearn
from sklearn.cluster import KMeans

import tasklogger
import collections
import warnings
from collections import defaultdict

from scipy.spatial.distance import pdist, cdist, squareform
warnings.simplefilter("ignore")

###### For the astronomy data
import astropy.units as u
from astropy.coordinates.sky_coordinate import SkyCoord
from astropy.units import Quantity
from astroquery.gaia import Gaia

def build_data_matrix_from_query(job_results, features):
  ''' After performing a query and getting the results, this function builds an n x d data matrix from the given d features in that order
  
  Parameters:
  ----------------
  job_results : astropy.table.table.Table
    results of a query after running .get_results()
  features : iterable of strings
    which fetures to include in the data matrix. Raises an exception if there is not a column with the given name
 
  Returns:
  ----------------
  X : an n x d ndarray
  '''

  cols = [np.array(job_results[f]) for f in features]
  X = np.column_stack(cols)
  
  return X

def build_dataframe_from_query(job_results, features=None):
  ''' After performing a query and getting the results, this function builds an pandas dataframe from the given features in that order
  
  Parameters:
  ----------------
  job_results : astropy.table.table.Table
    results of a query after running .get_results()
  features : iterable of strings
    which fetures to include in the data matrix. By default the function returns all of them
 
  Returns:
  ----------------
  X : pandas dataframe
    with the given feaures, by default all of them
  '''

  df = job_results.to_pandas()

  if features==None:
    return df

  return df[features]


def visualize_topology(catch_op, topological_activity=True, trees=False):
  ''' Visualizes topolical activity and, if indicated, condensation homology trees at 4 different granularities
  '''
  levels = catch_op.transform()
  tree = catch_op.build_tree()

  if topological_activity:
    print(levels)
    plt.plot(catch_op.gradient)
    plt.xlabel("Iterations")
    plt.ylabel("Topological Activity")
    plt.scatter(len(catch_op.NxTs)+levels, catch_op.gradient[levels+1], c='r')
    plt.show()

  if trees:
    fig = plt.figure(figsize=(8, 10))
    for i in range(4):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        tree_clusters = catch_op.map_clusters_to_tree(catch_op.NxTs[levels[-2*i-1]])
        
        scprep.plot.scatter3d(tree, c = tree_clusters, ax=ax, title = 'Granularity '+str(len(catch_op.NxTs)+levels[-2*i-1]),
                          fontsize=16, s = 20, legend = False,
                          ticks=False, label_prefix="PHATE", figsize=(6,10))
        ax.set_axis_off()

    fig.tight_layout()

    return levels, tree


def plot_phate_granularities(X, catch_op=None, real_idxs=None, custom_levels=None):
  levels = catch_op.transform()
  phate_op = phate.PHATE()
  data_phate = phate_op.fit_transform(X)

  fig, axes = plt.subplots(2,2, figsize=(12, 8))

  for i, ax in enumerate(axes.flatten()):
    if custom_levels != None:
        lev = levels[custom_levels[i]]
    else:
        lev = levels[-2*i-1]
      
    scprep.plot.scatter2d(data_phate, c=catch_op.NxTs[lev], legend_anchor=(1,1), ax=ax,
                            title='Granularity '+str(len(catch_op.NxTs)+lev),
                            xticks=False, yticks=False, label_prefix="PHATE", fontsize=10, s=3)
                            
    if real_idxs != None:
        xs = np.array([data_phate[i][0] for i in real_idxs])
        ys = np.array([data_phate[i][1] for i in real_idxs])
        special_pts = np.column_stack((xs, ys))
        scprep.plot.scatter2d(special_pts, c='black', ax=ax, s=10)

  fig.tight_layout()


def granulated_feature_plot(j, catch_op, features=['pmra', 'pmdec'], real_idxs=None, custom_levels=None):
  ''' Creates a 2-d scatter plot of features [x, y] colored according to 4 different granularities
  '''
  levels = catch_op.transform()
  fig, axes = plt.subplots(2,2, figsize=(12, 8))

  X = build_data_matrix_from_query(j, features)
  df = build_dataframe_from_query(j, features)

  for i, ax in enumerate(axes.flatten()):
      if custom_levels != None:
        lev = levels[custom_levels[i]]
      else:
        lev = levels[-2*i-1]

      scprep.plot.scatter2d(X, c=catch_op.NxTs[lev], ax=ax,
                            title='Granularity '+str(len(catch_op.NxTs)+lev),
                            xticks=True, yticks=True, label_prefix="PM", fontsize=10, s=3)
      if real_idxs != None:
        xs = np.array([list(df[features[0]])[i] for i in real_idxs])
        ys = np.array([list(df[features[1]])[i] for i in real_idxs])
        special_pts = np.column_stack((xs, ys))
        scprep.plot.scatter2d(special_pts, c='black', ax=ax, s=10)
      
      if features==['pmra', 'pmdec']:
        ax.set_xlim(-60,80)
        ax.set_ylim(-120,30)

  fig.tight_layout()
  
def k_means_feature_plot(X, Ns=[1, 2, 3, 4], features=['pmra', 'pmdec'], real_idxs=None):
  ''' Make a feature plot colored by k-means clustering at 4 different values of k

    Parameters:
    ----------------
    X : pandas dataframe
      with columns whose features are used to train k means
    Ns : iterable of ints
      number of clusters
    features : iterable of strings
      which features plot
    real_idxs : iteratble of ints
      indices of stars of interest to plot in black
  
    Returns:
    ----------------
  '''
  fig, axes = plt.subplots(2,2, figsize=(12, 9))
  for i, ax in enumerate(axes.flatten()):
        kmeans = KMeans(n_clusters=Ns[i], random_state=0).fit(X.to_numpy())
        preds = kmeans.predict(X.to_numpy())
        
        scprep.plot.scatter2d(X.loc[0:, features], c=preds, ax=ax, title='kmeans '+str(Ns[i]),
                              xticks=True, yticks=True, fontsize=10, s=3)
        if real_idxs != None:
          xs = np.array([list(X[features[0]])[i] for i in real_idxs])
          ys = np.array([list(X[features[1]])[i] for i in real_idxs])
          special_pts = np.column_stack((xs, ys))
          scprep.plot.scatter2d(special_pts, c='black', ax=ax, s=10)
        
        if features == ['pmra', 'pmdec']:
          ax.set_xlim(-60,80)
          ax.set_ylim(-120,30)

        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])

def get_common_source_idxs(j, ids):
  X=build_dataframe_from_query(j)
  reals = []
  locs = []
  for i in ids:
    if i in list(j['source_id']):
      reals.append(i)

  for r in reals:
    locs.append(X[X['source_id']==r].index.values.item())

  return locs
