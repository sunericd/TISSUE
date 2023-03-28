import numpy as np
import pandas as pd
import anndata as ad
import os


def large_save(adata, dirpath):
    '''
    Saves anndata objects by saving each obsm value with its {key}.csv as pandas dataframe
    Then saves the anndata object with obsm removed.
    
    adata - AnnData object to save
    dirpath [str] - path to directory for where to save the h5ad and csv files; will create if not existing
        adata will be saved as {dirpath}/adata.h5ad
        obsm will be saved as {dirpath}/{key}.csv
    '''
    # check if dirpath exists; else create it
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    # extract the obsm metadata and save it as separate csv files
    for key, value in adata.obsm.items():
        df = pd.DataFrame(value)
        df.to_csv(os.path.join(dirpath, f"{key}.csv"), index=False)

    # remove the obsm metadata from the anndata object
    adatac = adata.copy()
    adatac.obsm = {}

    # save the new anndata object
    adatac.write(os.path.join(dirpath, "adata.h5ad"))



def large_load(dirpath):
    '''
    Loads in anndata and associated pandas dataframe csv files to be added to obsm metadata.
    Input is the directory path to the output directory of large_save()
    '''
    # read h5ad anndata object
    adata = ad.read_h5ad(os.path.join(dirpath, "adata.h5ad"))
    
    # read and load in obsm from CSV files
    for fn in os.listdir(dirpath):
        if ".csv" in fn:
            df = pd.read_csv(os.path.join(dirpath, fn))
            df.index = adata.obs_names
            key = fn.split(".")[0]
            adata.obsm[key] = df
            
    return(adata)
    
    
def nan_weighted_std(values, weights):
    """
    Return the weighted standard deviation (omitting nan values)
    """
    values_f = values[np.isfinite(values)]
    weights_f = weights[np.isfinite(values)]
    average = np.average(values_f, weights=weights_f)
    variance = np.average((values_f-average)**2, weights=weights_f)
    return (np.sqrt(variance))