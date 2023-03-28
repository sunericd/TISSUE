import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold
import anndata as ad
import warnings
import os

from .utils import nan_weighted_std


def load_paired_datasets (spatial_counts, spatial_loc, RNAseq_counts, spatial_metadata = None,
                          min_cell_prevalence_spatial = 0.0, min_cell_prevalence_RNAseq = 0.0,
                          min_gene_prevalence_spatial = 0.0, min_gene_prevalence_RNAseq = 0.01):
    '''
    Uses datasets in the format specified by Li et al. (2022)
        See: https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J
        
        spatial_counts [str] - path to spatial counts file; rows are cells
        spatial_loc [str] - path to spatial locations file; rows are cells
        RNAseq_counts [str] - path to RNAseq counts file; rows are genes
        spatial_metadata [None or str] - if not None, then path to spatial metadata file (will be read into spatial_adata.obs)
        min_cell_prevalence_spatial [float between 0 and 1] - minimum prevalence among cells to include gene in spatial anndata object, default=0
        min_cell_prevalence_RNAseq [float between 0 and 1] - minimum prevalence among cells to include gene in RNAseq anndata object, default=0
        min_gene_prevalence_spatial [float between 0 and 1] - minimum prevalence among genes to include cell in spatial anndata object, default=0
        min_gene_prevalence_RNAseq [float between 0 and 1] - minimum prevalence among genes to include cell in RNAseq anndata object, default=0.01
    
    Returns AnnData objects with counts and location (if applicable) in metadata
    '''
    # Spatial data loading
    spatial_adata = load_spatial_data (spatial_counts,
                                     spatial_loc,
                                     spatial_metadata = spatial_metadata,
                                     min_cell_prevalence_spatial = min_cell_prevalence_spatial,
                                     min_gene_prevalence_spatial = min_gene_prevalence_spatial)
    
    # RNAseq data loading
    RNAseq_adata = load_rnaseq_data (RNAseq_counts,
                                     min_cell_prevalence_RNAseq = min_cell_prevalence_RNAseq,
                                     min_gene_prevalence_RNAseq = min_gene_prevalence_RNAseq)

    return(spatial_adata, RNAseq_adata)


def load_spatial_data (spatial_counts, spatial_loc, spatial_metadata=None,
                       min_cell_prevalence_spatial = 0.0, min_gene_prevalence_spatial = 0.0):

    df = pd.read_csv(spatial_counts,header=0,sep="\t")
    
    # filter lowly expressed genes
    cells_prevalence = np.mean(df>0, axis=0)
    df = df.loc[:,cells_prevalence > min_cell_prevalence_spatial]
    del cells_prevalence
    # filter sparse cells
    genes_prevalence = np.mean(df>0, axis=1)
    df = df.loc[genes_prevalence > min_gene_prevalence_spatial, :]
    del genes_prevalence
    # create AnnData
    spatial_adata = ad.AnnData(X=df)
    spatial_adata.obs_names = df.index.values
    spatial_adata.obs_names = spatial_adata.obs_names.astype(str)
    spatial_adata.var_names = df.columns
    del df
    
    # add spatial locations
    locations = pd.read_csv(spatial_loc,header=0,delim_whitespace=True)
    #locations.index = spatial_adata.obs_names
    spatial_adata.obsm["spatial"] = locations.values
    
    # add metadata
    if spatial_metadata is not None:
        metadata_df = pd.read_csv(spatial_metadata)
        metadata_df.index = spatial_adata.obs_names
        spatial_adata.obs = metadata_df
    
    # make unique obs_names and var_names
    spatial_adata.obs_names_make_unique()
    spatial_adata.var_names_make_unique()
    
    return (spatial_adata)


def load_rnaseq_data (RNAseq_counts, min_cell_prevalence_RNAseq = 0.0, min_gene_prevalence_RNAseq = 0.0):

    df = pd.read_csv(RNAseq_counts,header=0,index_col=0,sep="\t")
    # filter lowly expressed genes
    cells_prevalence = np.mean(df>0, axis=0)
    df = df.loc[:,cells_prevalence > min_cell_prevalence_RNAseq]
    del cells_prevalence
    # filter sparse cells
    genes_prevalence = np.mean(df>0, axis=1)
    df = df.loc[genes_prevalence > min_gene_prevalence_RNAseq, :]
    del genes_prevalence
    # create AnnData
    RNAseq_adata = ad.AnnData(X=df.T)
    RNAseq_adata.obs_names = df.T.index.values
    RNAseq_adata.var_names = df.T.columns
    del df
    
    # make unique obs_names and var_names
    RNAseq_adata.obs_names_make_unique()
    RNAseq_adata.var_names_make_unique()
    
    return (RNAseq_adata)



def preprocess_data (adata, standardize=False, normalize=False):
    '''
    Preprocesses adata inplace:
        1. sc.pp.normalize_total() if normalize is True
        2. sc.pp.log1p() if normalize is True
        3. Not recommended: standardize each gene (subtract mean, divide by standard deviation)
        
        standardize [Boolean] - whether to standardize genes; default is False
        normalize [Boolean] - whether to normalize data; default is False (based on finding by Li et al., 2022)
        
    NOTE: Under current default settings, this method does nothing to adata
    '''
    if normalize is True:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    
    if standardize is True:
        adata.X = np.divide(adata.X - np.mean(adata.X, axis=0), np.std(adata.X, axis=0))


def build_spatial_graph (adata, method="delaunay_radius", spatial="spatial", radius=None, n_neighbors=20, set_diag=True):
    '''
    Builds a spatial graph from AnnData according to specifications:
        adata [AnnData] - spatial data, must include adata.obsm[spatial]
        method [str]:
            - "radius" (all cells within radius are neighbors)
            - "delaunay" (triangulation)
            - "delaunay_radius" (triangulation with pruning by max radius; DEFAULT)
            - "fixed" (the k-nearest cells are neighbors determined by n_neighbors)
            - "fixed_radius" (knn by n_neighbors with pruning by max radius)
        spatial [str] - column name for adata.obsm to retrieve spatial coordinates
        radius [None or float/int] - radius around cell centers for which to detect neighbor cells; defaults to Q3+1.5*IQR of delaunay (or fixed for fixed_radius) neighbor distances
        n_neighbors [None or int] - number of neighbors to get for each cell (if method is "fixed" or "fixed_radius" or "radius_fixed"); defaults to 20
        set_diag [True or False] - whether to have diagonal of 1 in adjacency (before normalization); False is identical to theory and True is more robust; defaults to True
    
    Performs all computations inplace. Uses SquidPy implementations for graphs.
    '''
    # delaunay graph
    if method == "delaunay": # triangulation only
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
    
    # radius-based methods
    elif method == "radius": # radius only
        if radius is None: # compute 90th percentile of delaunay triangulation
            sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", set_diag=set_diag)
    elif method == "delaunay_radius":
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
        if radius is None:
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0
    elif method == "fixed_radius":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
        if radius is None:
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0
            
    # fixed neighborhood size methods
    elif method == "fixed":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
            
    else:
        raise Exception ("method not recognized")
        
        
def calc_adjacency_weights (adata, method="cosine", beta=0.0, confidence=None):
    '''
    Creates a normalized adjacency matrix containing edges weights for spatial graph
        adata [AnnData] = spatial data, must include adata.obsp['spatial_connectivities'] and adata.obsp['spatial_distances']
        method [str] = "binary" (weight is binary - 1 if edge exists, 0 otherwise); "cluster" (one weight for same-cluster and different weight for diff-cluster neighbors); "cosine" (weight based on cosine similarity between neighbor gene expressions)
        beta [float] = only used when method is "cluster"; between 0 and 1; specifies the non-same-cluster edge weight relative to 1 (for same cluster edge weight)
        confidence [None or str] = if [str], then will weight edges with respective node confidences using calc_confidence_weights()
            - confidence [str] is the key for adata.obsm[confidence] for the predicted expression of confidence genes from predict_gene_expression()
    
    Adds adata.obsp["S"]:
        S [numpy matrix] = normalized weight adjacency matrix; nxn where n is number of cells in adata
    '''
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    
    # adjacency matrix from adata
    if isinstance(adata.obsp["spatial_connectivities"],np.ndarray):
        A = adata.obsp['spatial_connectivities'].copy()
    else:
        A = adata.obsp['spatial_connectivities'].toarray().copy()
    #print("Average degree: "+str(np.mean(np.count_nonzero(A,axis=0))))
    #print("Median degree: "+str(np.median(np.count_nonzero(A,axis=0))))
    
    # compute weights
    if method == "binary":
        pass
    elif method == "cluster":
        # cluster AnnData if not already clustered
        if "cluster" not in adata.obs.columns:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_pcs=15)
            sc.tl.leiden(adata, key_added = "cluster")
        # init same and diff masks
        cluster_ids = adata.obs['cluster'].values
        same_mask = np.zeros(A.shape)
        for i in range(A.shape[1]):
            same_mask[:,i] = [1 if cid==cluster_ids[i] else 0 for cid in cluster_ids]
        diff_mask = np.abs(same_mask-1)
        # construct cluster-based adjacency matrix
        A = A*same_mask + A*diff_mask*beta
    elif method == "cosine":
        # PCA reduced space
        scaler = StandardScaler()
        pca = PCA(n_components=5, svd_solver='full')
        if isinstance(adata.X,np.ndarray):
            pcs = pca.fit_transform(scaler.fit_transform(adata.X))
        else:
            pcs = pca.fit_transform(scaler.fit_transform(adata.X.toarray()))
        # cosine similarities
        cos_sim = cosine_similarity(pcs)
        # update adjacency matrix
        A = A*cos_sim
        A[A < 0] = 0
        #print("After cosine weighting:")
        #print("Average degree: "+str(np.mean(np.count_nonzero(A,axis=1))))
        #print("Median degree: "+str(np.median(np.count_nonzero(A,axis=1))))
    else:
        raise Exception ("weighting must be 'binary', 'cluster', 'cosine'")
    
    # Compute confidence weights if specified
    if confidence is None:
        pass
    else:
        calc_confidence_weights(adata, confidence)
        A = A*adata.obs["confidence_score"].values # row-wise multiplication
        A[A < 0] = 0
    
    #print("After confidence weighting:")
    #print("Average degree: "+str(np.mean(np.count_nonzero(A,axis=1))))
    #print("Median degree: "+str(np.median(np.count_nonzero(A,axis=1))))
    
    # normalized adjacency matrix
    S = normalize(A, norm='l1', axis=1)
    
    # update adata
    adata.obsp["S"] = S


def predict_gene_expression (spatial_adata, RNAseq_adata,
                             target_genes, conf_genes=None,
                             method="spage", n_folds=None, random_seed=444, **kwargs):
    '''
    Leverages one of several methods to predict spatial gene expression from a paired spatial and scRNAseq dataset
        spatial_adata [AnnData] = spatial data
        RNAseq_adata [AnnData] = RNAseq data, RNAseq_adata.var_names should be superset of spatial_adata.var_names
        target_genes [list of str] = genes to predict spatial expression for; must be a subset of RNAseq_adata.var_names
        conf_genes [list of str] = genes in spatial_adata.var_names to use for confidence measures; Default is to use all genes in spatial_adata.var_names
        method [str] = baseline imputation method
            "onn" (uses nearest neighbor in RNAseq data on Harmony joint space)
            "knn" (uses average of k-nearest neighbors in RNAseq data on Harmony joint space)
            "spage" (SpaGE imputation by Abdelaal et al., 2020)
            "tangram" (Tangram cell positioning by Biancalani et al., 2021)
            Others TBD
        n_folds [None or int] = number of cv folds to use for conf_genes, cannot exceed number of conf_genes, None is keeping each gene in its own fold
        random_seed [int] = used to see n_folds choice (defaults to 444)
    
    Adds to adata the [numpy matrix]: spatial_adata.obsm["predicted_expression"], spatial_adata.obsm["combined_loo_expression"]
        - matrix of predicted gene expressions (same number of rows as spatial_adata, columns are target_genes)
    '''
    # change all genes to lower
    target_genes = [t.lower() for t in target_genes]
    spatial_adata.var_names = [v.lower() for v in spatial_adata.var_names]
    RNAseq_adata.var_names = [v.lower() for v in RNAseq_adata.var_names]
    
    # drop duplicates if any (happens in Dataset14)
    if RNAseq_adata.var_names.duplicated().sum() > 0:
        RNAseq_adata = RNAseq_adata[:,~RNAseq_adata.var_names.duplicated()].copy()
    if spatial_adata.var_names.duplicated().sum() > 0:
        spatial_adata = spatial_adata[:,~spatial_adata.var_names.duplicated()].copy()
    
    # raise warning if any target_genes in spatial data already
    if any(x in target_genes for x in spatial_adata.var_names):
        warnings.warn("Some target_genes are already measured in the spatial_adata object!")
    
    # first pass over all genes
    if method == "onn":
        predicted_expression_target = knn_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,n_neighbors=1,**kwargs)
    elif method == "knn":
        predicted_expression_target = knn_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "spage":
        predicted_expression_target = spage_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "gimvi":
        predicted_expression_target = gimvi_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "tangram":
        predicted_expression_target = tangram_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "stplus":
        predicted_expression_target = stplus_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    else:
        raise Exception ("method not recognized")
        
    # second pass over conf_genes
        # predictions done with a leave-gene-out approach
    if conf_genes is None:
        conf_genes = list(spatial_adata.var_names)
    conf_genes = [c.lower() for c in conf_genes]
    conf_genes_unique = [c for c in conf_genes if c not in target_genes] # removes any conf_genes also in target_genes
    if len(conf_genes_unique) < len(conf_genes):
        print("Found "+str(len(conf_genes)-len(conf_genes_unique))+" duplicate conf_gene in target_genes.")
    conf_genes_RNA = [c for c in conf_genes_unique if c in RNAseq_adata.var_names] # remove any conf genes not in RNAseq
    if len(conf_genes_RNA) < len(conf_genes_unique):
        print("Found "+str(len(conf_genes_unique)-len(conf_genes_RNA))+" conf_gene not in RNAseq_adata.")
    conf_genes = conf_genes_RNA
    
    # raise error if no conf_genes
    if len(conf_genes) == 0:
        raise Exception ("No suitable conf_genes specified!")
    
    # OLD: 1/11/2023 -- changed to accomodate n_folds argument
    # for gi, gene in enumerate(conf_genes):
        # if method == "onn":
            # loo_expression = knn_impute(spatial_adata[:,~spatial_adata.var_names.isin([gene])],RNAseq_adata,genes_to_predict=[gene]+target_genes,n_neighbors=1,**kwargs)
        # elif method == "knn":
            # loo_expression = knn_impute(spatial_adata[:,~spatial_adata.var_names.isin([gene])],RNAseq_adata,genes_to_predict=[gene]+target_genes,**kwargs)
        # elif method == "spage":
            # loo_expression = spage_impute(spatial_adata[:,~spatial_adata.var_names.isin([gene])],RNAseq_adata,genes_to_predict=[gene]+target_genes,**kwargs)
        # elif method == "gimvi":
            # loo_expression = gimvi_impute(spatial_adata[:,~spatial_adata.var_names.isin([gene])],RNAseq_adata,genes_to_predict=[gene]+target_genes,**kwargs)
        # elif method == "tangram":
            # loo_expression = tangram_impute(spatial_adata[:,~spatial_adata.var_names.isin([gene])],RNAseq_adata,genes_to_predict=[gene]+target_genes,**kwargs)
        # elif method == "stplus":
            # loo_expression = stplus_impute(spatial_adata[:,~spatial_adata.var_names.isin([gene])],RNAseq_adata,genes_to_predict=[gene]+target_genes,**kwargs)
        # else:
            # raise Exception ("method not recognized")
    
    # create folds if needed
    if n_folds is None:
        n_folds = len(conf_genes)
    elif n_folds > len(conf_genes):
        raise Warning ("n_folds in predict_gene_expression() is greater than length of conf_genes...")
        n_folds = len(conf_genes)

    np.random.seed(random_seed)
    np.random.shuffle(conf_genes)
    folds = np.array_split(conf_genes, n_folds)
    
    for gi, fold in enumerate(folds):
        if method == "onn":
            loo_expression = knn_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,n_neighbors=1,**kwargs)
        elif method == "knn":
            loo_expression = knn_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "spage":
            loo_expression = spage_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "gimvi":
            loo_expression = gimvi_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "tangram":
            loo_expression = tangram_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "stplus":
            loo_expression = stplus_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        else:
            raise Exception ("method not recognized")
    
        # Update 
        if gi == 0:
            predicted_expression_conf = loo_expression.copy()
        else:
            predicted_expression_conf['index'] = range(predicted_expression_conf.shape[0])
            loo_expression['index'] = range(loo_expression.shape[0])
            predicted_expression_conf.set_index('index')
            loo_expression.set_index('index')
            predicted_expression_conf = pd.concat((predicted_expression_conf,loo_expression)).groupby(by="index").sum().reset_index().drop(columns=['index'])
    
    # Take average of target_genes (later overwritten by "all genes"-predicted)
    predicted_expression_conf[target_genes] = predicted_expression_conf[target_genes]/(len(conf_genes))
    
    # Update spatial_adata
    predicted_expression_target.index = spatial_adata.obs_names
    predicted_expression_conf.index = spatial_adata.obs_names
    #spatial_adata.obsm[method+"_predicted_expression"] = predicted_expression_target  
    #spatial_adata.obsm[method+"_combined_loo_expression"] = predicted_expression_conf[[*conf_genes, *target_genes]]
    # gets predictions for target genes followed by conf genes
    predicted_expression_target[conf_genes] = predicted_expression_conf[conf_genes].copy()
    spatial_adata.obsm[method+"_predicted_expression"] = predicted_expression_target
    
    spatial_adata.uns["conf_genes_used"] = conf_genes
    spatial_adata.uns["target_genes_used"] = target_genes


def knn_impute (spatial_adata, RNAseq_adata, genes_to_predict, n_neighbors, **kwargs):
    '''
    Runs basic kNN imputation using Harmony subspace
    '''
    from scanpy.external.pp import harmony_integrate
    from scipy.spatial.distance import cdist
    
    # combine anndatas
    intersection = np.intersect1d(spatial_adata.var_names, RNAseq_adata.var_names)
    subRNA = RNAseq_adata[:, intersection]
    subspatial = spatial_adata[:, intersection]
    joint_adata = ad.AnnData(X=np.vstack((subRNA.X,subspatial.X)))
    joint_adata.obs_names = np.concatenate((subRNA.obs_names.values,subspatial.obs_names.values))
    joint_adata.var_names = subspatial.var_names.values
    joint_adata.obs["batch"] = ["rna"]*len(subRNA.obs_names.values)+["spatial"]*len(spatial_adata.obs_names.values)
    
    # run Harmony
    sc.tl.pca(joint_adata)
    harmony_integrate(joint_adata, 'batch', verbose=False)
    
    # kNN imputation
    #sc.pp.neighbors(joint_adata, use_rep='X_pca_harmony', knn=True, **kwargs) # should specify n_neighbors and
    #knn_mat = joint_adata.obsp['distances'][np.ix_(joint_adata.obs["batch"]=="spatial",joint_adata.obs["batch"]=="rna")]
    knn_mat = cdist(joint_adata[joint_adata.obs["batch"] == "spatial"].obsm['X_pca_harmony'][:,:np.min([30,joint_adata.obsm['X_pca_harmony'].shape[1]])],
                     joint_adata[joint_adata.obs["batch"] == "rna"].obsm['X_pca_harmony'][:,:np.min([30,joint_adata.obsm['X_pca_harmony'].shape[1]])])
    k_dist_threshold = np.sort(knn_mat)[:, n_neighbors-1]
    knn_mat[knn_mat > k_dist_threshold[:,np.newaxis]] = 0 # sets all dist > thresh to 0
    knn_mat[knn_mat > 0] = 1 # 1 for connection to a nn
    row_sums = knn_mat.sum(axis=1)
    knn_mat = knn_mat / row_sums[:,np.newaxis]
    predicted_expression = knn_mat @ RNAseq_adata.X
    
    predicted_expression = pd.DataFrame(predicted_expression, columns=RNAseq_adata.var_names.values)
    predicted_expression = predicted_expression[genes_to_predict]
    
    return(predicted_expression)
    
    
def spage_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Runs SpaGE gene imputation
    '''
    #sys.path.append("Extenrnal/SpaGE-master/")
    from SpaGE.main import SpaGE
    
    # transform adata in spage input data format
    if isinstance(spatial_adata.X,np.ndarray):
        spatial_data = pd.DataFrame(spatial_adata.X.T)
    else:
        spatial_data = pd.DataFrame(spatial_adata.X.T.toarray())
    spatial_data.index = spatial_adata.var_names.values
    if isinstance(RNAseq_adata.X,np.ndarray): # convert to array if needed
        RNAseq_data = pd.DataFrame(RNAseq_adata.X.T)
    else:
        RNAseq_data = pd.DataFrame(RNAseq_adata.X.T.toarray())
    RNAseq_data.index = RNAseq_adata.var_names.values
    # predict with SpaGE
    predicted_expression = SpaGE(spatial_data.T,RNAseq_data.T,genes_to_predict=genes_to_predict,**kwargs)
    
    return(predicted_expression)


def gimvi_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Run gimVI gene imputation
    '''
    import scvi
    from scvi.external import GIMVI
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    
    spatial_adata = spatial_adata[:, spatial_adata.var_names.isin(RNAseq_adata.var_names)]
    predict_idxs = [list(RNAseq_adata.var_names).index(gene) for gene in genes_to_predict]
    
    spatial_adata = spatial_adata.copy()
    RNAseq_adata = RNAseq_adata.copy()
    
    # setup anndata for scvi
    GIMVI.setup_anndata(spatial_adata)
    GIMVI.setup_anndata(RNAseq_adata)
    
    # train gimVI model
    model = GIMVI(RNAseq_adata, spatial_adata, **kwargs)
    #model.to_device("cuda")
    model.train(10)
    
    # apply trained model for imputation
    _, imputation = model.get_imputed_values(normalized=False)
    imputed = imputation[:, predict_idxs]
    predicted_expression = pd.DataFrame(imputed, columns=genes_to_predict)
    
    return(predicted_expression)


def tangram_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Run Tangram gene imputation (positioning) using the more efficient cluster-level approach with Leiden clustering
    '''
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    import tangram as tg
    
    # clustering and preprocessing
    RNAseq_adata_label = RNAseq_adata.copy()
    sc.pp.highly_variable_genes(RNAseq_adata_label)
    RNAseq_adata_label = RNAseq_adata[:, RNAseq_adata_label.var.highly_variable].copy()
    sc.pp.scale(RNAseq_adata_label, max_value=10)
    sc.tl.pca(RNAseq_adata_label)
    sc.pp.neighbors(RNAseq_adata_label)
    sc.tl.leiden(RNAseq_adata_label, resolution = 0.5)
    RNAseq_adata.obs['leiden'] = RNAseq_adata_label.obs.leiden
    del RNAseq_adata_label
    tg.pp_adatas(RNAseq_adata, spatial_adata) # genes=None default using all genes shared between two data
    
    # gene projection onto spatial
    ad_map = tg.map_cells_to_space(RNAseq_adata, spatial_adata, mode='clusters', cluster_label='leiden', density_prior='rna_count_based', verbose=False)
    ad_ge = tg.project_genes(ad_map, RNAseq_adata, cluster_label='leiden')
    #predicted_expression = pd.DataFrame(ad_ge[:,ad_ge.var_names.isin(genes_to_predict)].X, index=ad_ge[:,ad_ge.var_names.isin(genes_to_predict)].obs_names, columns=ad_ge[:,ad_ge.var_names.isin(genes_to_predict)].var_names)
    predicted_expression = pd.DataFrame(ad_ge[:,genes_to_predict].X, index=ad_ge[:,genes_to_predict].obs_names, columns=ad_ge[:,genes_to_predict].var_names)
    
    return(predicted_expression)
    
    
def conformalize_spatial_uncertainty (adata, predicted, calib_genes, #alpha_level=0.05, symmetric=True, 
                                      weight='uniform', mean_normalized=False, add_one=True,
                                      grouping_method=None, k=4, k2=None,
                                      n_pc=None, n_pc2=None):
    '''
    Builds scores from spatial uncertainties in predicted transcript expression using get_spatial_uncertainty_scores()
    Calibrates scores and computes conformal prediction intervals.
    
    Inputs:
        adata - AnnData object with adata.obsm[predicted] and adata.obsp['spatial_connectivites']
        predicted [str] - string corresponding to key in adata.obsm that contains the predicted transcript expression
        calib_genes [list or np.1darray] - strings corresponding to the genes to use in calibration
        alpha_level [float] - between 0 and 1; determines the alpha level; the CI will span the (1-alpha_level) interval
                              default value is alpha_level = 0.05 corresponding to 95% CI
        symmetric [bool] - whether to report symmetric prediction intervals or non-symmetric intervals; default is True (symmetric)
        weight [str] - weights to use when computing spatial variability (either 'exp_cos' or 'uniform'; default is 'uniform')
        mean_normalized [bool] - whether the standard deviation will be mean-normalized (i.e. coefficient of variation)
        For grouping_method [str], k [int>0], k2 [None or int>0], n_pc [None or int>0], n_pc2 [None or int>0]; refer to get_grouping()
        
    Saves the uncertainty in adata.obsm[predicted+"_uncertainty"]
    Saves the scores in adata.obsm[predicted+"_score"]
    Saves an upper and lower bound in adata.obsm[predicted+"_lo"/"_hi"]
    '''
    scores, residuals, G_stdev, G = get_spatial_uncertainty_scores(adata, predicted, calib_genes,
                                                                   weight=weight,
                                                                   mean_normalized=mean_normalized,
                                                                   add_one=add_one)
    
    adata.obsm[predicted+"_uncertainty"] = pd.DataFrame(G_stdev,
                                                        columns=adata.obsm[predicted].columns,
                                                        index=adata.obsm[predicted].index)
    adata.obsm[predicted+"_score"] = pd.DataFrame(scores,
                                                  columns=calib_genes,
                                                  index=adata.obsm[predicted].index)
    adata.obsm[predicted+"_error"] = pd.DataFrame(residuals,
                                                  columns=calib_genes,
                                                  index=adata.obsm[predicted].index)                                              
    
    # conformalization of uncertainty
    
    # define group
    if grouping_method is None:
        groups = np.zeros(G.shape)
    else:
        groups = get_grouping(G, method=grouping_method, k=k, k2=k2, n_pc=n_pc, n_pc2=n_pc2)
    adata.obsm[predicted+"_groups"] = groups
    
    
def get_spatial_uncertainty_scores (adata, predicted, calib_genes, weight='uniform', mean_normalized=False,
                                    add_one=True):
    '''
    Builds scores from spatial uncertainties in predicted transcript expression.
    
    Inputs:
        adata - AnnData object with adata.obsm[predicted] and adata.obsp['spatial_connectivites']
        predicted [str] - string corresponding to key in adata.obsm that contains the predicted transcript expression
        calib_genes [list or np.1darray] - strings corresponding to the genes to use in calibration
        weight [str] - weights to use when computing spatial variability (either 'exp_cos' or 'uniform'; default is 'uniform')
        mean_normalized [bool] - whether the standard deviation will be mean-normalized (i.e. coefficient of variation)
        add_one [bool] - whether to add one to the uncertainty
        
    Returns:
        scores - spatial uncertainty scores for all calib_genes
        G_stdev - spatial standard deviations measured; same shape as adata.obsm[predicted]
        G - adata.obsm[predicted].values
    '''
    if weight not in ["uniform", "exp_cos"]:
        raise Exception('weight not recognized')
    
    if 'spatial_connectivities' not in adata.obsp.keys():
        raise Exception ("'spatial_connectivities' not found in adata.obsp and is required")
        
    # compute standard deviations
    if isinstance(adata.obsp["spatial_connectivities"],np.ndarray):
        A = adata.obsp["spatial_connectivities"].copy() # has 1 on diagonals
    else:
        A = adata.obsp["spatial_connectivities"].toarray().copy()
    
    A[A == 0] = np.nan # convert zero to nan
    G = adata.obsm[predicted].values.copy()
    G_stdev = np.zeros_like(G)
    
    # compute weights
    if weight == "exp_cos":
        from sklearn.metrics.pairwise import cosine_similarity
        cos_weights = cosine_similarity(G)
        weights = np.exp(cos_weights)
    else:
        weights = np.ones(A.shape)

    for j in range(G.shape[1]):
        # multiply to get neighbors (row-wise element-wise multiplication)
        nA = A*G[:,j]
        
        # if weight == "exp_cos": # exponential cosine similarity of expression vectors for weighting
            # nA_std = []
            # for i in range(nA.shape[0]):
                # nA_std.append(nan_weighted_std(nA[i,:], weights=weights[i,:]))
            # nA_std = np.array(nA_std)
        # else: # uniform
            # # take standard deviation across rows
            # nA_std = np.nanstd(nA, axis=1)
        # if mean_normalized is True: # coefficient of variation (SD/mu) -- masked for SD > 0
            # nA_std[nA_std>0] = nA_std[nA_std>0] / np.nanmean(nA, axis=1)[nA_std>0]
        # G_stdev[:,j] = nA_std
        
        # compute variability
        nA_std = []
        for i in range(nA.shape[0]):
            #nA_std.append(nan_weighted_std(nA[i,:], weights=weights[i,:]))
            nA_std.append(cell_centered_variability(nA[i,:], weights=weights[i,:], c_idx=i))
        nA_std = np.array(nA_std)
        
        # mean normalization if specified
        if mean_normalized is True: # coefficient of variation (SD/mu) -- masked for SD > 0
            nA_std[nA_std>0] = nA_std[nA_std>0] / np.nanmean(nA, axis=1)[nA_std>0]
        
        # add one if specified
        if add_one is True:
            nA_std += 1
        
        # update G_stdev with uncertainties
        G_stdev[:,j] = nA_std
        
    # compute scores based on confidence genes (prediction residuals)
    calib_idxs = [np.where(adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]
    residuals = adata[:, calib_genes].X - adata.obsm[predicted][calib_genes].values # Y-G
    
    warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress RuntimeWarning for division by zero
    scores = np.abs(residuals) / G_stdev[:, calib_idxs] # scores
    warnings.filterwarnings("default", category=RuntimeWarning)
    
    return(scores, residuals, G_stdev, G)


def cell_centered_variability (values, weights, c_idx):
    '''
    Takes in an array an nA and weights to compute cell-centered variability:
            
        values [1d arr] - array with cell's masked neighborhood expression (non-neighbors are nan)
        weights [1d arr] - same dim as values; contains weights for computing CCV_c
        c_idx [int] - index for which element of nA corresponds to center cell
    '''
    values_f = values[np.isfinite(values)]
    weights_f = weights[np.isfinite(values)]
    average = values[c_idx] # "average" is simply the center cell value
    variance = np.average((values_f-average)**2, weights=weights_f)
    ccv = np.sqrt(variance)
    
    return(ccv)


def get_spatial_uncertainty_scores_from_metadata(adata, predicted):
    '''
    Returns scores, residuals, G_stdev, G (outputs of get_spatial_uncertainty_scores) from precomputed entries
    in the AnnData (adata) object. Note, these must have been computed and saved in the same was as in
    conformalize_spatial_uncertainty().
    
    adata [AnnData] - object that has saved results in obsm
    predicted [str] - key for predictions in obsm
    '''
    scores = np.array(adata.obsm[predicted+"_score"]).copy()
    residuals = np.array(adata.obsm[predicted+"_error"]).copy()
    G_stdev = np.array(adata.obsm[predicted+"_uncertainty"]).copy()
    G = np.array(adata.obsm[predicted]).copy()
    groups = np.array(adata.obsm[predicted+"_groups"]).copy()
    
    return(scores, residuals, G_stdev, G, groups)


def get_grouping(G, method, k=1, k2=None, min_samples=5, n_pc=None, n_pc2=None):
    '''
    Given the predicted gene expression matrix G (rows=cells, cols=genes),
    creates a grouping of the different genes (or cells) determined by:
    
        G [numpy matrix/array] - predicted gene expression; columns are genes
        method [str] - 'cv_exp' to separate by quantiles of CV in gene expression
                       'kmeans_gene' to separate genes by k-means clustering
                       'kmeans_cell' to separate cells by k-means clustering
                       'kmeans_gene_cell' to separate by genes and the by cells by k-means clustering
        k [int] - number of groups; only for cv_exp, kmeans_gene, kmeans_cell and kmeans_gene_cell
                  if <=1 then defaults to one group including all values
        k2 [int] - second number of groups for kmeans_gene_cell
                  if <=1 then defaults to one group including all values
        min_samples [int] - min number of samples; only for dbscan clustering
        n_pc and npc2 [None or int] - number of PCs to use before KMeans clustering
                           - NOTE: It is recommended to do this for methods: "kmeans_gene" and "kmeans_gene_cell"
        
    Returns:
        groups [numpy array] - same dimension as G with values corresponding to group number (integer)
    '''
    if k <= 1: # just one group
        groups = np.zeros(G.shape)
    else:
        if method == "cv_exp":
            cv_exp = np.nanstd(G, axis=0)/np.nanmean(G, axis=0)
            grouping = np.array_split(np.argsort(cv_exp), k) # split into ordered groups by index
            groups = np.ones(len(cv_exp))*np.nan # init nan group array
            for gi, g in enumerate(grouping): # assign group labels
                groups[g] = gi
            groups = np.tile(groups, (G.shape[0], 1)) # expand so each row is the same
        
        elif method == "kmeans_gene":
            X = StandardScaler().fit_transform(G.T)
            if n_pc is not None:
                X = PCA(n_components=n_pc).fit_transform(X)
            kmeans = KMeans(n_clusters=k, random_state=444).fit(X)
            groups = kmeans.labels_
            groups = np.tile(groups, (G.shape[0], 1)) # expand so each row is the same
            
        elif method == "kmeans_cell":
            X = StandardScaler().fit_transform(G)
            if n_pc is not None:
                X = PCA(n_components=n_pc).fit_transform(X)
            kmeans = KMeans(n_clusters=k, random_state=444).fit(X)
            groups = kmeans.labels_
            groups = np.tile(groups, (G.shape[1], 1)).T # expand so each col is the same
            
        elif method == "kmeans_gene_cell":
            # gene grouping
            X = StandardScaler().fit_transform(G.T)
            if n_pc is not None:
                X = PCA(n_components=n_pc).fit_transform(X)
            kmeans_genes = KMeans(n_clusters=k, random_state=444).fit(X)
            cluster_genes = kmeans_genes.labels_
            
            groups = np.ones(G.shape)*np.nan # init groups array
            counter = 0 # to index new groupsn with integers
            
            # within each gene group, group cells
            for cg in np.unique(cluster_genes):
                G_group = G[:, cluster_genes==cg]
                X_group = StandardScaler().fit_transform(G_group)
                if n_pc2 is not None:
                    X_group = PCA(n_components=n_pc2).fit_transform(X_group)
                kmeans_cells = KMeans(n_clusters=k2, random_state=444).fit(X_group)
                cluster_cells = kmeans_cells.labels_
                for cc in np.unique(cluster_cells):
                    groups[np.ix_(cluster_cells==cc, cluster_genes==cg)] = counter
                    counter += 1
            
        else:
            raise Exception("method for get_grouping() is not recognized")
    
    return(groups)



def conformalize_prediction_interval (adata, predicted, calib_genes, alpha_level=0.05, symmetric=False, return_scores_dict=False):
    '''
    Builds conformal prediction interval sets for the predicted gene expression
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm that corresponds to predicted gene expression 
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        alpha_level [float] - the alpha for the prediction interval (between 0 and 1 for 1-alpha coverage)
        symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
        return_scores_dict [bool] - whether to return the scores dictionary
    '''
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    ### Building calibration sets for scores
    
    scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric)
    
    ### Building prediction intervals

    prediction_sets = (np.zeros(G.shape), np.zeros(G.shape)) # init prediction sets

    # conformalize independently within groups of genes
    for group in np.unique(groups[~np.isnan(groups)]):
        if symmetric is True:
            scores_flattened = scores_flattened_dict[str(group)]
            n = len(scores_flattened)
            if (n < 100): # if less than 100 samples in either set, then use the full group set
                scores_flattened = scores_flattened_dict[str(np.nan)]
                n = len(scores_flattened)-np.isnan(scores_flattened).sum()
            try:
                qhat = np.nanquantile(scores_flattened, np.ceil((n+1)*(1-alpha_level))/n)
            except:
                qhat = np.nan
            prediction_sets[0][groups==group] = (G-G_stdev*qhat)[groups==group]
            prediction_sets[1][groups==group] = (G+G_stdev*qhat)[groups==group]
        else:
            scores_lo_flattened = scores_flattened_dict[str(group)][0]
            scores_hi_flattened = scores_flattened_dict[str(group)][1]
            n_lo = len(scores_lo_flattened)-np.isnan(scores_lo_flattened).sum()
            n_hi = len(scores_hi_flattened)-np.isnan(scores_hi_flattened).sum()
            if (n_lo < 100) or (n_hi < 100): # if less than 100 samples in either set, then use the full group set
                scores_lo_flattened = scores_flattened_dict[str(np.nan)][0]
                scores_hi_flattened = scores_flattened_dict[str(np.nan)][1]
                n_lo = len(scores_lo_flattened)-np.isnan(scores_lo_flattened).sum()
                n_hi = len(scores_hi_flattened)-np.isnan(scores_hi_flattened).sum()
            try:
                qhat_lo = np.nanquantile(scores_lo_flattened, np.ceil((n_lo+1)*(1-alpha_level))/n_lo)
                qhat_hi = np.nanquantile(scores_hi_flattened, np.ceil((n_hi+1)*(1-alpha_level))/n_hi)
            except:
                qhat_lo = np.nan
                qhat_hi = np.nan
            prediction_sets[0][groups==group] = (G-G_stdev*qhat_lo)[groups==group]
            prediction_sets[1][groups==group] = (G+G_stdev*qhat_hi)[groups==group]
            
    # add prediction intervals to adata
    adata.uns['alpha'] = alpha_level
    adata.obsm[predicted+"_lo"] = pd.DataFrame(prediction_sets[0],
                                               columns=adata.obsm[predicted].columns,
                                               index=adata.obsm[predicted].index)
    adata.obsm[predicted+"_hi"] = pd.DataFrame(prediction_sets[1],
                                               columns=adata.obsm[predicted].columns,
                                               index=adata.obsm[predicted].index)
            
    if return_scores_dict is True:
    
        return(scores_flattened_dict)
        
        
        
def build_calibration_scores (adata, predicted, calib_genes, symmetric=False, include_zero_scores=False,
                              trim_quantiles=[None,None]):
    '''
    Builds calibration score sets
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm with predicted gene expression values
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
        include_zero_scores [bool] - whether to exclude zero scores
        trim_quantiles [list of len 2; None or float between 0 and 1] - specifies what quantile range of scores to trim to; None implies no bounds
    '''
    
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)

    scores_flattened_dict = {}
    
    # get calibration genes
    calib_idxs = [np.where(adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]
    
    # iterate groups and build conformal sets of calibration scores
    for group in np.unique(groups[~np.isnan(groups)]):
        if (np.isnan(group)) or (group not in groups[:, calib_idxs]): # defer to using full calibration set
            scores_group = scores.copy()
            residuals_group = residuals.copy()
        else: # for groups that are found in the calibration set, build group-specific sets
            scores_group = scores.copy()[groups[:, calib_idxs]==group]
            residuals_group = residuals.copy()[groups[:, calib_idxs]==group]
        if symmetric is True: # symmetric calibration set
            if include_zero_scores is False:
                scores_flattened = scores_group[residuals_group != 0].flatten() # exclude zeros
            else:
                scores_flattened = scores_group.flatten()
            scores_flattened_dict[str(group)] = scores_flattened[np.isfinite(scores_flattened)] # add to dict
        else: # separate into hi/lo non-symmetric calibration sets
            if include_zero_scores is False:
                scores_lo_flattened = scores_group[residuals_group < 0].flatten()
                scores_hi_flattened = scores_group[residuals_group > 0].flatten()
            else:
                scores_lo_flattened = scores_group[residuals_group <= 0].flatten()
                scores_hi_flattened = scores_group[residuals_group >= 0].flatten()
            scores_flattened_dict[str(group)] = (scores_lo_flattened[np.isfinite(scores_lo_flattened)],
                                                 scores_hi_flattened[np.isfinite(scores_hi_flattened)]) # add to dict

    # build nan group consisting of all scores
    if symmetric is True: # symmetric calibration set
        if include_zero_scores is False:
            scores_flattened = scores[residuals != 0].flatten() # exclude zeros
        else:
            scores_flattened = scores.flatten()
        scores_flattened_dict[str(np.nan)] = scores_flattened[np.isfinite(scores_flattened)] # add to dict
    else: # separate into hi/lo non-symmetric calibration sets
        if include_zero_scores is False:
            scores_lo_flattened = scores[residuals < 0].flatten()
            scores_hi_flattened = scores[residuals > 0].flatten()
        else:
            scores_lo_flattened = scores[residuals <= 0].flatten()
            scores_hi_flattened = scores[residuals >= 0].flatten()
        scores_flattened_dict[str(np.nan)] = (scores_lo_flattened[np.isfinite(scores_lo_flattened)],
                                             scores_hi_flattened[np.isfinite(scores_hi_flattened)]) # add to dict
    
    # trim all scores if specified
    for key in scores_flattened_dict.keys():
    
        # determine quantiles from original scores
        if symmetric is True:
            if trim_quantiles[0] is not None:
                lower_bound = np.nanquantile(scores_flattened_dict[key], trim_quantiles[0])
            if trim_quantiles[1] is not None:
                upper_bound = np.nanquantile(scores_flattened_dict[key], trim_quantiles[1])
        else:
            if trim_quantiles[0] is not None:
                lower_bound_lo = np.nanquantile(scores_flattened_dict[key][0], trim_quantiles[0])
                lower_bound_hi = np.nanquantile(scores_flattened_dict[key][1], trim_quantiles[0])
            if trim_quantiles[1] is not None:
                upper_bound_lo = np.nanquantile(scores_flattened_dict[key][0], trim_quantiles[1])
                upper_bound_hi = np.nanquantile(scores_flattened_dict[key][1], trim_quantiles[1])
        
        # trim based on quantiles
        if symmetric is True:
            if trim_quantiles[0] is not None:    
                scores_flattened_dict[key] = scores_flattened_dict[key][scores_flattened_dict[key]>lower_bound]
            if trim_quantiles[1] is not None:    
                scores_flattened_dict[key] = scores_flattened_dict[key][scores_flattened_dict[key]<upper_bound]
        else:
            if trim_quantiles[0] is not None:    
                scores_flattened_dict[key] = (scores_flattened_dict[key][0][scores_flattened_dict[key][0]>lower_bound_lo],
                                              scores_flattened_dict[key][1][scores_flattened_dict[key][1]>lower_bound_hi])
            if trim_quantiles[1] is not None:    
                scores_flattened_dict[key] = (scores_flattened_dict[key][0][scores_flattened_dict[key][0]<upper_bound_lo],
                                              scores_flattened_dict[key][1][scores_flattened_dict[key][1]<upper_bound_hi])
       
    return (scores_flattened_dict)