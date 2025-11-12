def infer_meta_cols(df, exclude_cols=None, tol_unique=1):
    """
    Infer metadata columns: those that are constant across the DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        The run-level DataFrame.
    exclude_cols : list, optional
        Columns to ignore (e.g., obvious prediction or target columns).
    tol_unique : int, default=1
        Maximum number of unique non-NaN values to consider 'constant'.
    
    Returns
    -------
    list of str
        Column names likely representing run-level metadata.
    """
    if exclude_cols is None:
        exclude_cols = []

    const_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        n_unique = df[col].nunique(dropna=True)
        if n_unique <= tol_unique:
            const_cols.append(col)
    return const_cols


def count_pars(modality, days, minutes, field, poly_feat=False, time_res=60, horizon=60, diag=False):

    def count_kalman(n_feat, n_out):
        if diag:
            return n_out * (2 * n_feat + 1) # two vectors + noise
        else:
            return n_out * (n_feat**2 + n_feat + 1)

    n_kernel_recent = minutes // time_res    
    n_out = horizon // time_res
        
    if field == 'block':
        n_feat = days * (n_kernel_recent - n_out)
    elif field == 'cross':
        n_feat = ((days - 1) +  (n_kernel_recent - n_out))
    elif field == 'hour':
        n_feat = (n_kernel_recent - n_out)
    elif field == 'day':
        n_feat = (days - 1)
    else:
        raise ValueError(f"Unknown field: {field}")

    if poly_feat:
        n_feat = n_feat*(n_feat-1)/ 2 # no interactions
    
    if modality == 'ah|ah': # augmented space
        n_total = 2*count_kalman(2*n_feat, n_out) # two filters
    elif modality == 'a|ah': # only one augmented space
        n_act = count_kalman(2*n_feat, n_out)
        n_heart = count_kalman(n_feat, n_out)
        n_total = n_heart + n_act
    elif modality == 'a|a': # only one filter
        n_total = count_kalman(n_feat, n_out)
    elif modality == 'a|h': # only one filter
        n_total = count_kalman(n_feat, n_out)
    else:
        raise ValueError(f"Unknown modality: {modality}")
    
    return n_total
