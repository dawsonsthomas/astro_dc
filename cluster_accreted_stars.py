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

#Import the utility functions
from astro_dc_utils import *

#read the file
Stars = pd.read_hdf('/home/phys678_dst29/project/PHYS678_2022/final_project/Public_SelectedByNetwork_6D.h5')
Stars = Stars.sample(n=8000)
print(Stars)


#convert the pmra and pmdec to velocities in km/s
au_yr = 1.49598e+8 / 3.15576e+7
pmra_to_kms = (Stars['pmra']/Stars['parallax']) * au_yr
pmdec_to_kms = (Stars['pmdec']/Stars['parallax']) * au_yr
Stars['v_phi'] = pmra_to_kms
Stars['v_z'] = pmra_to_kms



vs = [pmra_to_kms, pmdec_to_kms]
ylabels = ['$v_\phi$ [km/s]', '$v_z$ [km/s]'] #Check these!

#Plot Before clustering:
fig, axes = plt.subplots(1,2, figsize=(16, 8))

for i, ax in enumerate(axes.flatten()):
  ax.scatter(Stars['radial_velocity'], vs[i], color='b', alpha=0.2, s=10)
  ax.set_xlim(-500, 500)
  ax.set_ylim(-500, 500)
  ax.set_aspect('equal')
  ax.set_xlabel('$v_R$ [km/s]')
  ax.set_ylabel(ylabels[i])


plt.savefig('figs/velocity_plots_before_clustering')


#run catch
#select features to cluster on
features = ['v_phi', 'v_z', 'radial_velocity', 'l', 'b', 'phot_g_mean_mag'] #'phot_bp_mean_mag', 'phot_rp_mean_mag', 'v_phi', 'v_z'
X = Stars[features]
print(X)

catch_op = catch.CATCH(knn=20, random_state=18, n_pca=len(features), n_jobs=1)
catch_op.fit(X)

#visualize topology
plt.clf()
visualize_topology(catch_op, topological_activity=True, trees=False)
plt.savefig('figs/topological_activity')



#visualize granularities
levels = catch_op.transform()
custom_levels = [0, 1, 2, 3]

for j in range(2):
    fig, axes = plt.subplots(2,2, figsize=(12, 8))
    for i, ax in enumerate(axes.flatten()):
        if custom_levels != None:
            lev = levels[custom_levels[i]]
        else:
            lev = levels[-2*i-1]

        scprep.plot.scatter2d(np.column_stack((Stars['radial_velocity'], vs[j])), c=catch_op.NxTs[lev], ax=ax, title='Granularity '+str(len(catch_op.NxTs)+lev), xticks=True, yticks=True, label_prefix="PM", fontsize=10, s=3, alpha=0.6, legend=False)
        ax.set_xlim(-500, 500)
        ax.set_ylim(-500, 500)
        ax.set_aspect('equal')
        ax.set_xlabel('$v_R$ [km/s]')
        ax.set_ylabel(ylabels[j])
        
    plt.savefig('figs/granulated_feature_plot'+str(j))



#compare to k-means
#k_means_feature_plot(X, Ns=[3, 4, 5, 6], features=['pmra', 'pmdec'], real_idxs=pleiades)

#visualize with phate
plot_phate_granularities(X, catch_op, custom_levels=custom_levels)
plt.savefig('figs/phate_visualizations')
