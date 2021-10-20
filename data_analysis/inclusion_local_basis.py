import pandas as pd
from data_analysis import static as da

"""
    ToDo:
    [ ] create test data and param file in ~/Colloid/Data/eshelby_rotation
        [+] recompile ~/Colloid/Data/eshelby_rotation.ipynb to make sure the cells I wrote still compile if called in seq
        - synthetic data from output of mathematica nb
            - load mathematica output which already has the strain computed
            - apply a rotation
            - find high von Mises and confirm discont distrubtion of von Mises strains (core and matrix)
            - copy and shift up to create two inclusions
            - find high strain on duplicated dataset
            - cluster using graph adjacency to get two clusters.
            - diagonalize using numpy routines or da.strainDiag to give formatted dataframe indexed by particle id
            - there is some subtle point about eigenvectors and their permutations that I have forgotten and I likely
              never figured out how to determine the permutation matrix.
                  >> u6,v6,w6 = np.array([-0.5,-0.5,-0.707107]), np.array([-0.5,-0.5,0.707107]),np.array([-0.707107,0.707107,0])
                  >> u1,v1,w1 = np.array([0,1,0]),np.array([0.707107,0,0.707107]),np.array([0.707107,0,-0.707107])
                  >> perm = np.array([[1,0,0],[0,0,1],[0,1,0]])
                  >> (R.T@np.array([u6,v6,w6])).T
                  >> (R.T@np.array([u6,v6,w6])).T @ (-1*perm), np.array([u1,v1,w1]).T
                  >> ((np.array([u6,v6,w6])).T), (R@np.array([u1,v1,w1])).T @(-1 * np.array([[0,1,0],[1,0,0],[0,0,1]]))
        - isolate inclusion that was found manually from glass shear step f in the correct coordinates (rheo sed depth)
        - param test file
    - outline functions to carry out the following steps
        - load particle positions, as needed, from hdf file
        - compute strain as needed
        - find particles with high local von Mises strain between 
          specified (likely short 3 to 5) time frames
        - cluster high strain particles using adjacency matrix on nnb distances. 
        - find local eigen basis for each cluster...should operate on an iterable. 
        - transform strain around cluster into local basis
        - output rotation matrices or some other representation of local basis in the sample
        - average strain fields accounting for local rotation
        - visualize clusters and strain components in ovito 
    - write or transcribe functions from jupyter notebook
    - encapsulate functions into class as methods
        - data structures: 
             - particle dataframe (index is particle id, and queried from pyTables) 
             - cluster of particles (index shared with particle dataFrame, and just a column of cluster id)
             - transformation matrices for each cluster (also a dataFrame with index as cluster id)
        - 
    - encapsulate attibutes commonly used
        - file paths
        - 
    - complete refactoring to allow instantiating the class in jupyter and
      carrying out the rest of the analysis, including figure saving within jupyter.
    -  
"""

