def precision(ground_truth, pred):
    '''
    Root function to calculate the precision.
    
    Inputs:
        ground_truth: A (H x W x ...) array of ground truth values
        pred: A (H x W x ...) array of predicted bounding box values
    '''
    assert isinstance(ground_truth, np.ndarray), 'Expected NumPy array.'
    assert isinstance(pred, np.ndarray), 'Expected NumPy array.'
    assert ground_truth.shape == pred.shape, \
    'Normalized bounding box coordinates for ground truth and predictions do not agree.'
    
    return (ground_truth[..., 0] * pred[..., 0]).sum() / ground_truth[..., 0].sum()