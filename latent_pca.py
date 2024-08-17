import numpy
import torch
import numpy as np
from sklearn.decomposition import PCA
def latent_pca(z: np.array, k = 100):
    """
    Args:
        z: (N, V, C) array of N training pose latents   
        k: num of principal components
    Returns:
        pca: sklearn pca object
        newX: k principal matrix
    """ 
    N, V, C = z.shape
    X = np.reshape(z, [N, (V * C)])
    z_mean = np.mean(X, axis = 0)
    pca = PCA(n_components = 0.95) 
    pca = pca.fit(X)
    newX = pca.components_.T
    std = np.std(newX, axis=0)
    return newX, z_mean, std

def pca_projection(newX, z_mean, std, z):
    """
    Args: 
        z: (B, V, C) array of GRU output latents
    Returns:
        z_recon: (B, V, C) array of reconstructed z
    """
    B, V, C = z.shape
    z = np.reshape(z, [B, (V * C)])
    beta = (z - z_mean) @ newX
    beta = np.clip(beta, - 2 * std, 2 * std)
    z_recon = beta @ newX.T + z_mean
    return np.reshape(z_recon, [B, V, C])


if __name__ == "__main__":
    z = np.random.rand(1000, 160, 128)
    newX, z_mean, std = latent_pca(z)
    new_z = z[0][None]
    z_recon = pca_projection(newX, z_mean, std, new_z)
    print(z_recon.shape)
    
