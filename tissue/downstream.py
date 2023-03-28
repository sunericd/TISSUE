import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import StandardScaler
import anndata as ad
import os

from main import build_calibration_scores, get_spatial_uncertainty_scores_from_metadata



def multiple_imputation_testing (adata, predicted, calib_genes, condition, n_imputations=100,
                                 group1=None, group2=None, symmetric=False, return_keys=False):
    '''
    Uses multiple imputation with the score distributions to perform hypothesis testing
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm that corresponds to predicted gene expression
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        condition [str] - key in adata.obs for which to compute the hypothesis test
            group1 [value] - value in adata.obs[condition] identifying the first comparison group
                             if None, will perform group vs all comparisons for all unique values in adata.obs[condition]
            group2 [value] - value in adata.obs[condition] identifying the second comparison group
                             if None, will compare against all values that are not group1
        n_imputations [int] - number of imputations to use
        symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
    '''
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    ### Building calibration sets for scores
    
    scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric,
                                                     include_zero_scores=True, trim_quantiles=[None, 0.8]) # trim top 20% scores
    
    ### Multiple imputation
    
    # init dictionary to hold results (for independent two-sample t-test)
    stat_dict = {}
    stat_dict["mean_difference"] = {}
    stat_dict["standard_deviation"] = {}
    
    for m in range(n_imputations):
        
        # generate new imputation
        new_G = sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=symmetric)
    
        # calculate statistics for the imputation using approach from Palmer & Peer, 2016
        
        if group1 is None: # pairwise comparisons against all
            
            for g1 in np.unique(adata.obs[condition]):
                
                key = str(g1)+"_all"
            
                if m == 0: # init list
                    stat_dict["mean_difference"][key] = []
                    stat_dict["standard_deviation"][key] = []
                
                g1_bool = (adata.obs[condition] == g1) # g1
                g2_bool = (adata.obs[condition] != g1) # all other
                
                mean_diff, pooled_sd = get_ttest_stats(new_G, g1_bool, g2_bool) # get ttest stats
                stat_dict["mean_difference"][key].append(mean_diff)
                stat_dict["standard_deviation"][key].append(pooled_sd)
                
        elif group2 is None: # group1 vs all
        
            key = str(group1)+"_all"
            
            if m == 0: # init list
                stat_dict["mean_difference"][key] = []
                stat_dict["standard_deviation"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] != group1) # all other
            
            mean_diff, pooled_sd = get_ttest_stats(new_G, g1_bool, g2_bool) # get ttest stats
            stat_dict["mean_difference"][key].append(mean_diff)
            stat_dict["standard_deviation"][key].append(pooled_sd)
            
        else: # group1 vs group2
            
            key = str(group1)+"_"+str(group2)
            
            if m == 0: # init list
                stat_dict["mean_difference"][key] = []
                stat_dict["standard_deviation"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] == group2) # g2
            
            mean_diff, pooled_sd = get_ttest_stats(new_G, g1_bool, g2_bool) # get ttest stats
            stat_dict["mean_difference"][key].append(mean_diff)
            stat_dict["standard_deviation"][key].append(pooled_sd)

    # pool statistics and perform t-test
    pooled_results_dict = pool_multiple_stats(stat_dict)#, adata.obsm[predicted].columns, adata.var_names)
     
    # add prediction intervals to adata
    keys_list = []
    for key_measure in pooled_results_dict.keys():
        for key_comparison in pooled_results_dict[key_measure].keys():
            adata.uns[predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure] = pd.DataFrame(pooled_results_dict[key_measure][key_comparison][None,:],
                                                                                                 columns=adata.obsm[predicted].columns)
            keys_list.append(predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure)
    
    if return_keys is True:
    
        return(keys_list)


def sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=False):
    '''
    Creates a new imputation by sampling from scores and adding to G
    '''
    new_scores = np.zeros(G.shape) # init array for sampled scores
    new_add_sub = np.zeros(G.shape) # init array for add/subtract coefs
    
    # for each group, sample calibration score and corresponding imputations
    unique_groups, unique_counts = np.unique(groups[~np.isnan(groups)], return_counts=True)
    
    for ui, group in enumerate(unique_groups):
        count = unique_counts[ui] # get number of values in group
        
        # sample scores and add/sub indicators
        if symmetric is True:
            scores_flattened = scores_flattened_dict[str(group)] # get scores
            if len(scores_flattened) < 100: # default to full set if <100 in group
                scores_flattened = scores_flattened_dict[str(np.nan)]
            sampled_scores = np.random.choice(scores_flattened, count, replace=True) # with replacement, sample scores
            add_sub = np.random.choice([-1,1], count, replace=True) # add or subtract
        else:
            scores_lo_flattened = scores_flattened_dict[str(group)][0]
            scores_hi_flattened = scores_flattened_dict[str(group)][1]
            if (len(scores_lo_flattened) < 100) or (len(scores_hi_flattened) < 100): # default to full set if <100 in group
                scores_lo_flattened = scores_flattened_dict[str(np.nan)][0]
                scores_hi_flattened = scores_flattened_dict[str(np.nan)][1]
            scores_flattened = np.concatenate((scores_lo_flattened, scores_hi_flattened))
            lo_hi_indicators = np.concatenate(([-1]*len(scores_lo_flattened), [1]*len(scores_hi_flattened)))
            # sample indices
            sampled_idxs = np.random.choice(np.arange(len(scores_flattened)), count, replace=True) # with replacement
            sampled_scores = scores_flattened[sampled_idxs]
            add_sub = lo_hi_indicators[sampled_idxs]
        
        # append to new_scores and new_add_sub
        new_scores[groups==group] = sampled_scores
        new_add_sub[groups==group] = add_sub
        
    # calculate new imputation
    new_G = G + new_add_sub*(new_scores*G_stdev)

    return (new_G)


def get_ttest_stats(G, g1_bool, g2_bool):
    '''
    Computes mean_diff and pooled SD for each column of G independently
    '''
    mean_diff = np.nanmean(G[g1_bool,:], axis=0) - np.nanmean(G[g2_bool,:], axis=0)
    #pooled_sd = np.sqrt(2/n) * np.sqrt( 1/2*(np.nanvar(G[g1_bool,:],axis=0) + np.nanvar(G[g2_bool,:],axis=0)) )
    n1 = np.count_nonzero(~np.isnan(G[g1_bool,:]), axis=0)
    n2 = np.count_nonzero(~np.isnan(G[g2_bool,:]), axis=0)
    sp = np.sqrt( ( (n1-1)*(np.nanvar(G[g1_bool,:],axis=0)) + (n2-1)*(np.nanvar(G[g2_bool,:],axis=0)) ) / (n1+n2-2) )
    pooled_sd = np.sqrt(1/n1 + 1/n2) * sp
    
    return(mean_diff, pooled_sd)


def two_sample_ttest (G, g1_bool, g2_bool):
    '''
    Computes two-sample t-test for unequal sample sizes
    
    Use get_ttest_stats
    '''
    from scipy import stats
    # calculate t-stat
    mean_diff, pooled_sd = get_ttest_stats(G, g1_bool, g2_bool)
    tt = mean_diff/pooled_sd
    # calculate dof
    n1 = np.count_nonzero(~np.isnan(G[g1_bool,:]), axis=0)
    n2 = np.count_nonzero(~np.isnan(G[g2_bool,:]), axis=0)
    dof = n1+n2-2
    # calculate p-value
    pp = 2*(1 - stats.t.cdf(np.abs(tt), dof))
    
    return(tt, pp)


def pool_multiple_stats(stat_dict):
    '''
    Pool stats across multiple imputations for t-test
    
    stat_dict [dict] - dictionary containing statistical testing results
    #target_vars [arr] - target order of variable names
    #current_vars [arr] - current order of variable names
    '''
    from scipy import stats
    
    results_dict = {}
    results_dict["tstat"] = {}
    results_dict["pvalue"] = {}
    
    results_dict["varw"] = {}
    results_dict["varb"] = {}
    results_dict["poolmean"] = {}
    
    for key in stat_dict["mean_difference"].keys():
        
        d = len(stat_dict["mean_difference"][key])
        
        pooled_mean = np.mean(np.vstack(stat_dict["mean_difference"][key]), axis=0)
        var_w = np.mean(np.vstack(stat_dict["standard_deviation"][key])**2, axis=0) # within-draw sample variance
        var_b = 1/(d-1) * np.sum((np.vstack(stat_dict["mean_difference"][key])-pooled_mean)**2, axis=0) # between-draw sample variance
        var_MI = var_w + (1+1/d)*var_b # multiple imputation variance
        
        test_stat = pooled_mean / np.sqrt(var_MI) # pooled t statistic
        
        # compute pvalue from T distribution
        dof = (d-1)*(1+(d*var_w)/((d+1)*var_b))**2 # degrees of freedom for T distribution
        pval = 2*(1 - stats.t.cdf(np.abs(test_stat), dof))
        
        # reorder accordingly so genes match up to target_vars
        # _,sorting_idxs = np.where(target_vars[:,None] == current_vars)
        # test_stat = test_stat[sorting_idxs]
        # pval = pval[sorting_idxs]
        
        # Add test statistic and pvalue
        results_dict["tstat"][key] = test_stat
        results_dict["pvalue"][key] = pval
        
        # Add intermediate stats (for debugging, etc)
        results_dict["varw"][key] = var_w
        results_dict["varb"][key] = var_b
        results_dict["poolmean"][key] = pooled_mean
    
    return(results_dict)



def weighted_PCA(adata, imp_method, pca_method="wpca", weighting="inverse_pi_width", quantile_cutoff=None,
                 n_components=15, replace_inf=None, binarize=False, binarize_ratio=100, log_transform=False,
                 scale=True, tag="", return_weights=False,):
    '''
    Runs weighted PCA using the "wpca" package: https://github.com/jakevdp/wpca
    
        adata [AnnData] - should be the AnnData after running conformalize_prediction_interval()
                        - must include in obsm: {imp_method}_predicted_expression,
                                                {imp_method}_predicted_expression_lo,
                                                {imp_method}_predicted_expression_hi
        imp_method [str] - specifies which imputation method to return PCA for (e.g. 'knn', 'spage', 'tangram')
        pca_method [str] - "wpca" for WPCA (Delchambre, 2014), "empca" for EMPCA (Bailey, 2012), "pca" for PCA
        weighting [str] - "uniform" (regular PCA)
                          "inverse_pi_width" (weights are 1/(prediction interval width))
                          "inverse_norm_pi_width" (weights are predicted expression/(prediction interval width))
        quantile_cutoff [None or float] - quantile (between 0 and 1) for which to set a ceiling for the weights
        n_components [int] - number of principal components
        replace_inf [None, str, float] - what to replace np.inf with (after all other weight transforms); if None, keeps np.inf
                                         can also be "max" or "min" to replace with the max or min weights
        binarize [bool] - binarizes the weights with Otsu threshold -- if larger than threshold, set to 1; else 1e-2
        binarize_ratio [int or float] - how much to "upweight" values greater than the binarized threshold
        log_transform [bool] - whether to log1p transform weights (will be done before binarization if binarize=True)
        scale [bool - whether to scale data with StandardScaler() before running WPCA
        tag [str] - additional tag to append to the obsm key for storing the PCs
        return_weights [bool] - whether to return weights
        
    Stores the result in adata.obsm["{imp_method}_predicted_expression_PC{n_components}_{tag}"]
    
    Refer to postprocess_weights() for order for weight calculations
    '''
    from wpca import PCA, WPCA, EMPCA
    
    predicted = f"{imp_method}_predicted_expression"
    
    # get gene names/order
    genes = adata.obsm[predicted].columns
    
    # determine weights
    if weighting == "inverse_pi_width":
        weights = 1/(adata.obsm[predicted+'_hi'][genes].values-adata.obsm[predicted+'_lo'][genes].values)
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    elif weighting == "inverse_norm_pi_width":
        #weights = adata.obsm[predicted][genes].values/(adata.obsm[predicted+'_hi'][genes].values-adata.obsm[predicted+'_lo'][genes].values)
        weights = 1/(adata.obsm[predicted+'_hi'][genes].values-adata.obsm[predicted+'_lo'][genes].values)
        weights = weights / np.nanmean(weights, axis=0)
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    elif weighting == "uniform":
        weights = np.ones(adata.obsm[predicted].shape)
    elif weighting == "inverse_residual":
        weights = 1/np.abs(adata.obsm[predicted][genes].values - np.array(adata[:,genes].X))
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    elif weighting == "inverse_norm_residual":
        #weights = adata.obsm[predicted][genes].values / np.abs(adata.obsm[predicted][genes].values - np.array(adata[:,genes].X))
        weights = 1/np.abs(adata.obsm[predicted][genes].values - np.array(adata[:,genes].X))
        weights = weights / np.nanmean(weights, axis=0)
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    else:
        raise Exception("weighting not recognized")
    
    # scaling
    if scale is True:
        X = StandardScaler().fit_transform(adata.obsm[predicted].values)
    else:
        X = adata.obsm[predicted].values
    
    # run weighted PCA
    if pca_method == "wpca":
        X_red = WPCA(n_components=n_components).fit_transform(X, weights=weights)
    elif pca_method == "empca":
        X_red = EMPCA(n_components=n_components).fit_transform(X, weights=weights)
    elif pca_method == "pca":
        X_red = PCA(n_components=n_components).fit_transform(X)
    elif pca_method == "gwpca": # gene-weighted PCA
        weights = np.nanmean(weights, axis=0)
        X_red = PCA(n_components=n_components).fit_transform(X * weights)
    else:
        raise Exception("pca_method not recognized")
        
    # add PCs to adata
    adata.obsm[predicted+f"_PC{n_components}_{tag}"] = X_red

    if return_weights is True:
        return(weights)


def postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform):
    '''
    Method for postprocessing weights (filter with cutoff, replace inf, etc) for weighted_PCA()
    '''
    # cutoff weights
    if quantile_cutoff is not None:
        cutoff = np.nanquantile(weights, quantile_cutoff)
        weights[np.isfinite(weights) & (weights >= cutoff)] = cutoff
    
    # log-transform
    if log_transform is True:
        weights = np.log1p(weights)
    
    # binarize weights
    if binarize is True:
        from skimage.filters import threshold_otsu
        #cutoff = threshold_otsu(np.unique(weights[np.isfinite(weights)]))
        cutoff = threshold_otsu(weights[np.isfinite(weights)])
        weights[np.isfinite(weights) & (weights >= cutoff)] = 1
        weights[np.isfinite(weights) & (weights < cutoff)] = 1/binarize_ratio
    elif binarize is False:
        pass
    elif isinstance(binarize, float) or isinstance(binarize, int):
        cutoff = np.nanquantile(weights, binarize)
        weights[np.isfinite(weights) & (weights >= cutoff)] = 1
        weights[np.isfinite(weights) & (weights < cutoff)] = 1/binarize_ratio
        
    # deal with infs (from division by zero)
    if replace_inf == "max":
        weights[~np.isfinite(weights)] = np.nanmax(weights[np.isfinite(weights)])
    elif replace_inf == "min":
        weights[~np.isfinite(weights)] = np.nanmin(weights[np.isfinite(weights)])
    elif replace_inf == "mean":
        weights[~np.isfinite(weights)] = np.nanmean(weights[np.isfinite(weights)])
    elif replace_inf == "median":
        weights[~np.isfinite(weights)] = np.nanmedian(weights[np.isfinite(weights)])
    elif isinstance(replace_inf, float) or isinstance(replace_inf, int):
        weights[~np.isfinite(weights)] = replace_inf
    
    return(weights)


def detect_uncertain_cells (X, proportion=0.05, stratification=None):
    '''
    Method for dropping a portion of the most uncertain cells from the input. 
    
        X [2d numpy array] - array of uncertainty values 
        proportion [float] - between 0 and 1; proportion of most uncertain cells to drop
        stratification [None or 1d numpy array] - array of values to stratify the drop by
                                                - same length as number of rows in X
                                                - if None, no stratification
        
    Returns:
        keep_idxs [list] - array of row indices after dropping most uncertain cells
    '''
    from scipy.stats import zscore
    
    if stratification is not None:
    
        drop_idxs = []
        
        for strata in np.unique(stratification):
        
            X_strat = X[stratification==strata,:].copy() # calc gene z-scores
            orig_idxs = np.arange(X.shape[0])[stratification==strata]
            cell_scores = np.nanmean(zscore(X_strat, axis=0), axis=1) # average z-score for each cell
            
            if (isinstance(proportion, float)) or (isinstance(proportion, int)):
                cutoff_idx = int(np.ceil(proportion*len(cell_scores))) # number of cells to drop
                strata_drop_idxs = np.argsort(cell_scores)[::-1][:cutoff_idx]
            elif proportion == "otsu":
                from skimage.filters import threshold_otsu
                cutoff = threshold_otsu(cell_scores)
                strata_drop_idxs = [i for i in range(len(cell_scores)) if cell_scores[i] > cutoff]
            else:
                raise Exception("proportion specified not valid")
                
            drop_idxs.append(orig_idxs[strata_drop_idxs]) # get idxs of highest scores
            
        drop_idxs = list(np.concatenate(drop_idxs))
    
    else:
    
        cell_scores = zscore(X, axis=0).mean(axis=1) # average z-score for each cell
        
        if (isinstance(proportion, float)) or (isinstance(proportion, int)):
            cutoff_idx = int(np.ceil(proportion*len(cell_scores))) # number of cells to drop
            drop_idxs = list(np.argsort(cell_scores)[::-1][:cutoff_idx]) # get idxs of highest scores
        elif proportion == "otsu":
            from skimage.filters import threshold_otsu
            cutoff = threshold_otsu(cell_scores)
            drop_idxs = [i for i in range(len(cell_scores)) if cell_scores[i] > cutoff] 
        else:
            raise Exception("proportion specified not valid")
    
    keep_idxs = [i for i in range(X.shape[0]) if i not in drop_idxs]
    
    return (keep_idxs)