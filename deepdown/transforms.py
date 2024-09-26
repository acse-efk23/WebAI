import torch
import numpy as np


class ToTensor:
    def __call__(self, np_array):
        return torch.from_numpy(np_array).float()
    
    
class Crop:
    def __call__(self, np_array):
        return np_array[:, :, 2:114, :, 63:95]
    

class MinMaxNormalise:
    def __call__(self, np_array):
        num_features = np_array.shape[1]  # Get the number of features from the second dimension

        feature_mins = np.zeros((num_features, 1, 1, 1))
        feature_maxs = np.zeros((num_features, 1, 1, 1))

        for i in range(num_features):
            feature_mins[i] = np.nanmin(np_array[:, i, :, :, :])
            feature_maxs[i] = np.nanmax(np_array[:, i, :, :, :])

        normalised_data = np.zeros_like(np_array)
        for i in range(num_features):
            if feature_maxs[i] - feature_mins[i] == 0:
                normalised_data[:, i, :, :, :] = np_array[:, i, :, :, :]
            else:
                normalised_data[:, i, :, :, :] = (np_array[:, i, :, :, :] - feature_mins[i]) / (feature_maxs[i] - feature_mins[i])

        return normalised_data
    

class LocalisedImpute:
    def __call__(self, np_array, kernel_size=5):   
        half_size = kernel_size // 2
        shape = np_array.shape
        output_array = np.copy(np_array)
        
        for case in range(shape[0]):
            for i in range(shape[1]):
                for j in range(shape[2]):
                    for k in range(shape[3]):
                        for l in range(shape[4]):
                            if np.isnan(np_array[case, i, j, k, l]):
                                cube = np_array[case, 
                                                i,
                                                max(0, j-half_size):min(shape[2], j+half_size+1),
                                                max(0, k-half_size):min(shape[3], k+half_size+1),
                                                max(0, l-half_size):min(shape[4], l+half_size+1)]
                                if np.isnan(cube).all():
                                    output_array[case, i, j, k, l] = np.nanmean(np_array[case, i, :, :, :])
                                else:
                                    output_array[case, i, j, k, l] = np.nanmean(cube)
        return output_array