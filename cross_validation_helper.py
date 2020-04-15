#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  14 23:45:24 2020

@author: joeranbosma
"""
import pandas as pd
import numpy as np

def cv_train_val_split(df, cross_val_num=0, cross_val_parts=8, random_state=576, verbose=False):
    """Obtain train and validation sets for k-fold cross validation. 
    Using this methods, the same splits can be performed across multiple trails. 
    This also enables to, for example, perform a test with only the first split 
    between train/test for all hyperparameter settings. When this test is satisfactional, 
    the test can be extended with missing cross validation parts 2..8. 

    Default cross validation parts: 8, with 200.840 * 12.5% = 25.105 validation samples
    Default cross validation split: 0
    """
    # get indices for each split (%%timeit: 1 loop, best of 3: 478 ms per loop)
    kfold_train_indices, kfold_val_indices = cv_train_val_split_indices(
        df, cross_val_parts=cross_val_parts, random_state=random_state, verbose=verbose)
    
    # return corresponding split
    idx_train = kfold_train_indices[cross_val_num]
    idx_val   = kfold_val_indices[cross_val_num]
    return df.loc[idx_train], df.loc[idx_val]

def cv_train_val_split_indices(df, cross_val_parts=8, random_state=576, verbose=True):
    """Note: indices change between pd.Index and np.array several times for performance reasons"""
    
    assert isinstance(df, pd.DataFrame), "Requires pandas DataFrame"
    
    kfold_train_indices, kfold_val_indices = [], []
    np.random.seed(random_state)
    
    for k in range(cross_val_parts):
        # Select all indices contained in the train dataframe
        indices = df.index

        # remove the indices which have already been in any of the previous validation sets
        available_test_indices = indices
        for i in range(k):
            available_test_indices = available_test_indices.difference(kfold_val_indices[i])

        available_test_indices = np.array(available_test_indices)
        if verbose: 
            print("{} indices available".format(available_test_indices.size))

        # Shuffle the indices
        np.random.shuffle(available_test_indices)
        
        # Add a portion of test_size to the test set
        test_indices = pd.Index(available_test_indices[0 : int(len(indices) / cross_val_parts)])
        train_indices = indices.difference(test_indices)

        # Shuffle the indices (again)
        train_indices = train_indices.values
        test_indices = test_indices.values
        np.random.shuffle(train_indices)
        np.random.shuffle(test_indices)

        kfold_val_indices.append(test_indices)
        kfold_train_indices.append(train_indices)
        
        if verbose:
            print("Selected {} rows for training and {} for validation (of total {} rows)".format(
                len(train_indices), len(test_indices), df.index.size))
    
    # small check:
    assert (kfold_val_indices[0] != kfold_val_indices[1]).all(), "Index reappeared in two validation splits!"
    
    return kfold_train_indices, kfold_val_indices

""""
    Realised that the code above could be faster, with the code below. But this would void the current cross validation splits
    cv_parts = 8
    all_idx = list(np.arange(202000))
    np.random.shuffle(all_idx)
    val_size = int(len(all_idx) / cv_parts)
    
    cv_train_idx, cv_val_idx = [], []
    for k in range(cv_parts):
        val_start, val_stop = int(k*val_size), int((k+1)*val_size)
        cv_val_idx.append(  all_idx[val_start:val_stop]  )
        
        new_train_idx = all_idx[0: val_start]
        new_train_idx.extend( all_idx[val_stop:] )
        cv_train_idx.append(  new_train_idx  )
"""
