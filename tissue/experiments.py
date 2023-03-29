import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import anndata as ad
import warnings
import os
import gc

from tissue.utils import large_save, large_load


def group_conformalize_from_intermediate(dataset_name, methods, symmetric, alpha_levels,
                                         save_alpha=[0.05], savedir="SCPI"):
    '''
    Function for taking intermediate fold predictions and running group conformalization for all different alpha values
    
    Returns a results dictionary with calibration quality (res_dict) and the AnnData with CI for all folds at alpha of save_alpha [float]

    '''
    # read in spatial data
    if os.path.isfile("DataUpload/"+dataset_name+"/Metadata.txt"):
        adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                  "DataUpload/"+dataset_name+"/Locations.txt",
                                   spatial_metadata = "DataUpload/"+dataset_name+"/Metadata.txt")
    else:
        adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                  "DataUpload/"+dataset_name+"/Locations.txt")
    adata.var_names = [x.lower() for x in adata.var_names]
    
    # results dict
    res_dict = {}
    
    for method in methods:

        res_dict[method] = {}
        res_dict[method]['ind_gene_results'] = {}

        calibration_weight = 0 # for computing weighted average
        test_weight = 0

        dirpath = savedir+"/"+dataset_name+"_intermediate/"+method
        
        folds = np.load(os.path.join(savedir+"/"+dataset_name+"_intermediate/"+method,"folds.npy"), allow_pickle=True)

        # subset spatial data into shared genes
        gene_names = np.concatenate(folds)
        adata = adata[:, gene_names]

        for i, fold in enumerate(folds):

            # load adata within fold
            sub_adata = large_load(os.path.join(dirpath, "fold"+str(i)))
            target_genes = list(fold)

            # subset data
            predicted = method+"_predicted_expression"
            test_genes = target_genes.copy()
            calib_genes = [gene for gene in gene_names if gene not in test_genes]
            test_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in test_genes]
            calib_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]

            # get uncertainties and scores from saved adata
            scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (sub_adata, predicted)

            # init dict for individual gene results
            for g in test_genes:
                if g not in res_dict[method]['ind_gene_results'].keys():
                    res_dict[method]['ind_gene_results'][g] = {}
                    res_dict[method]['ind_gene_results'][g]['1-alpha'] = 1-alpha_levels
                    res_dict[method]['ind_gene_results'][g]['test'] = []

            # iterate over different alphas for conformalization
            test_perc = []
            calib_perc = []

            for alpha_level in alpha_levels:
                sub_adatac = sub_adata.copy()
                conformalize_prediction_interval (sub_adatac, predicted, calib_genes, alpha_level=alpha_level,
                                                  symmetric=symmetric, return_scores_dict=False)
                
                prediction_sets = (sub_adatac.obsm[predicted+"_lo"].values, sub_adatac.obsm[predicted+"_hi"].values)
                
                test_perc.append(np.nanmean(((adata[:,test_genes].X>prediction_sets[0][:,test_idxs]) & (adata[:,test_genes].X<prediction_sets[1][:,test_idxs]))[(G[:,test_idxs]!=0)&(G_stdev[:,test_idxs]!=0)&(adata[:,test_genes].X!=0)]))
                calib_perc.append(np.nanmean(((adata[:,calib_genes].X>prediction_sets[0][:,calib_idxs]) & (adata[:,calib_genes].X<prediction_sets[1][:,calib_idxs]))[(G[:,calib_idxs]!=0)&(G_stdev[:,calib_idxs]!=0)&(adata[:,calib_genes].X!=0)]))

                # Compute individual calibration curves for each gene
                for ti, tg in zip(test_idxs, test_genes):
                    if sub_adatac.obsm[predicted].columns[ti] != tg:
                        raise Warning ("ti not equal to tg: "+str(adata.var_names[ti])+" != "+str(tg))
                    ind_test_ci = np.nanmean(((adata[:,tg].X>prediction_sets[0][:,ti]) & (adata[:,tg].X<prediction_sets[1][:,ti]))[(G[:,ti]!=0)&(G_stdev[:,ti]!=0)&(adata[:,tg].X!=0)])
                    res_dict[method]['ind_gene_results'][tg]['test'].append(ind_test_ci)
                    
                del sub_adatac
                del prediction_sets
                del ind_test_ci
                gc.collect()

            # weighted average
            if i == 0:
                calibration_ci = np.array(calib_perc) * len(calib_genes)
                calibration_weight += len(calib_genes)
                test_ci = np.array(test_perc) * len(test_genes)
                test_weight += len(test_genes)
            else:
                calibration_ci += np.array(calib_perc) * len(calib_genes)
                calibration_weight += len(calib_genes)
                test_ci += np.array(test_perc) * len(test_genes)
                test_weight += len(test_genes)
                
            # Add new predictions
            for si, s_alpha in enumerate(save_alpha):
                #sub_adatac = sub_adata.copy()
                conformalize_prediction_interval (sub_adata, predicted, calib_genes, alpha_level=s_alpha,
                                                  symmetric=symmetric, return_scores_dict=False)
                
                if i == 0:
                    if si == 0: # to avoid overwriting multiple times
                        adata.obsm[predicted] = pd.DataFrame(sub_adata.obsm[predicted][fold].values,
                                                          columns=fold,
                                                          index=adata.obs_names)
                    adata.obsm[predicted+f"_lo_{round((1-s_alpha)*100)}"] = pd.DataFrame(sub_adata.obsm[predicted+"_lo"][fold].values,
                                                      columns=fold,
                                                      index=adata.obs_names)
                    adata.obsm[predicted+f"_hi_{round((1-s_alpha)*100)}"] = pd.DataFrame(sub_adata.obsm[predicted+"_hi"][fold].values,
                                                      columns=fold,
                                                      index=adata.obs_names)
                else:
                    if si == 0: # to avoid overwriting multiple times
                        adata.obsm[predicted][fold] = sub_adata.obsm[predicted][fold].values.copy()
                    adata.obsm[predicted+f"_lo_{round((1-s_alpha)*100)}"][fold] = sub_adata.obsm[predicted+"_lo"][fold].values.copy()
                    adata.obsm[predicted+f"_hi_{round((1-s_alpha)*100)}"][fold] = sub_adata.obsm[predicted+"_hi"][fold].values.copy()
                
            del sub_adata
            gc.collect()

        # add results
        calibration_ci = calibration_ci / calibration_weight
        test_ci = test_ci / test_weight

        res_dict[method]['1-alpha'] = 1-alpha_levels
        res_dict[method]['calibration'] = calibration_ci
        res_dict[method]['test'] = test_ci
        
    return(res_dict, adata)



def group_multiple_imputation_testing_from_intermediate(dataset_name, methods, symmetric, condition, n_imputations=100,
                                                        group1=None, group2=None, savedir="SCPI"):
    '''
    Function for taking intermediate fold predictions and running multiple imputation t-tests
    
    Returns AnnData object with all test results saved in adata.var
    ''' 
    # read in spatial data
    if os.path.isfile("DataUpload/"+dataset_name+"/Metadata.txt"):
        adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                  "DataUpload/"+dataset_name+"/Locations.txt",
                                   spatial_metadata = "DataUpload/"+dataset_name+"/Metadata.txt")
    else:
        adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                  "DataUpload/"+dataset_name+"/Locations.txt")
    adata.var_names = [x.lower() for x in adata.var_names]
    
    for method in methods:

        dirpath = savedir+"/"+dataset_name+"_intermediate/"+method
        
        folds = np.load(os.path.join(savedir+"/"+dataset_name+"_intermediate/"+method,"folds.npy"), allow_pickle=True)

        # subset spatial data into shared genes
        gene_names = np.concatenate(folds)
        adata = adata[:, gene_names]

        for i, fold in enumerate(folds):

            # load adata within fold
            sub_adata = large_load(os.path.join(dirpath, "fold"+str(i)))
            target_genes = list(fold)

            # subset data
            predicted = method+"_predicted_expression"
            test_genes = target_genes.copy()
            calib_genes = [gene for gene in gene_names if gene not in test_genes]
            test_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in test_genes]
            calib_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]

            # run multiple imputation test
            keys_list = multiple_imputation_testing (sub_adata, predicted, calib_genes, condition, n_imputations=n_imputations,
                                                     group1=group1, group2=group2, symmetric=symmetric, return_keys=True)
            
            if i == 0:
                for key in keys_list:
                    adata.var[key] = np.zeros(adata.shape[1])
                    
                adata.obsm[predicted] = pd.DataFrame(sub_adata.obsm[predicted][fold].values,
                                                  columns=fold,
                                                  index=adata.obs_names)
            for key in keys_list:
                adata.var[key][[np.where(adata.var_names==gene)[0][0] for gene in fold]] = sub_adata.uns[key][fold].values.flatten()
                adata.obsm[predicted][fold] = sub_adata.obsm[predicted][fold].values.copy()
                
    return(adata)
    
def leiden_clustering(adata, pca=True, inplace=False, **kwargs):
    '''
    Performs Leiden clustering using settings in the PBMC3K tutorial from Scanpy:
    
    https://scanpy-tutorials.readthedocs.io/en/latest/pbmc3k.html
    
    Adds under key "leiden" in adata.obs
    '''
    if inplace is False:
        adata = adata.copy()
    if pca is True:
        adata.X[np.isnan(adata.X)] = 0
        adata.X[adata.X < 0] = 0
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        sc.tl.pca(adata, svd_solver='arpack')
    else:
        adata.obsm['X_pca'] = adata.X
    sc.pp.neighbors(adata)#, n_neighbors=10, n_pcs=15)
    sc.tl.leiden(adata, **kwargs)
    
    return (adata.obs['leiden'].copy(), adata.obsm['X_pca'].copy())
    
def pca_correlation(pcs1, pcs2):
    '''
    Computes Pearson correlation between concatenated/flattened pcs1 and pcs2
    
    Effectively equal to variance-weighted average of the column-wise correlations
    '''
    corr, p = pearsonr(pcs1.flatten(), pcs2.flatten())
    
    return(corr)