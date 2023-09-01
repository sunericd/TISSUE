# Contains utility functions for TISSUE

import numpy as np
import pandas as pd
import anndata as ad
import os


def large_save(adata, dirpath):
    '''
    Saves anndata objects by saving each obsm value with its {key}.csv as pandas dataframe
    Saves each uns value that is a dataframe with uns/{key}.csv as pandas dataframe
    Then saves the anndata object with obsm removed.
    
    Parameters
    ----------
        adata [AnnData] - AnnData object to save
        
        dirpath [str] - path to directory for where to save the h5ad and csv files; will create if not existing
            adata will be saved as {dirpath}/adata.h5ad
            obsm will be saved as {dirpath}/{key}.csv
        
    Returns
    -------
        Saves anndata object in "large" folder format
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
    
    # extract the uns metadata and save it as separate csv files
    del_keys = []
    for key, value in adatac.uns.items():
        if isinstance(value, pd.DataFrame):
            if not os.path.exists(os.path.join(dirpath,"uns")):
                os.makedirs(os.path.join(dirpath,"uns"))
            df = pd.DataFrame(value)
            df.to_csv(os.path.join(dirpath,"uns",f"{key}.csv"), index=False)
            del_keys.append(key)
    
    # remove uns metadata from the anndata object
    for key in del_keys:
        del adatac.uns[key]

    # save the new anndata object
    adatac.write(os.path.join(dirpath, "adata.h5ad"))



def large_load(dirpath):
    '''
    Loads in anndata and associated pandas dataframe csv files to be added to obsm metadata and uns metadata.
    Input is the directory path to the output directory of large_save()
    
    Parameters
    ----------
        dirpath [str] - path to directory for where outputs of large_save() are located
    
    Returns
    -------
        adata - AnnData object loaded from dirpath along with all obsm and uns key values added to metadata
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
            
    # read and load any usn metadata from CSV files
    if os.path.isdir(os.path.join(dirpath,"uns")):
        for fn in os.listdir(os.path.join(dirpath,"uns")):
            if ".csv" in fn:
                df = pd.read_csv(os.path.join(dirpath,"uns",fn))
                key = fn.split(".")[0]
                adata.uns[key] = df
            
    return(adata)
    
    
# def nan_weighted_std(values, weights):
    # """
    # Return the weighted standard deviation (omitting nan values)
    
    # Parameters
    # ----------
        # values [array] - values for which to compute weighted standard deviation
    
        # weights [array] - weights for each of the values
    
    # Returns
    # -------
        # Weighted standard deviation of the values [float]
    # """
    # values_f = values[np.isfinite(values)]
    # weights_f = weights[np.isfinite(values)]
    # average = np.average(values_f, weights=weights_f)
    # variance = np.average((values_f-average)**2, weights=weights_f)
    
    # return (np.sqrt(variance))