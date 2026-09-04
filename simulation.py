import numpy as np
import scipy.spatial.transform as st
from scipy.spatial.transform import Rotation as R

class Config:
    data_num=10000
    n_samples=100
    alpha_base = [0,90,-90]  
    beta_base = [0,45,-45]  
    tx_min=0
    tx_max=1
    ty_min=0
    ty_max=1
    tz_min=0
    tz_max=1
    rmin_min=0.6
    rmin_max=1
    rmaj_min=0
    rmaj_max=0.2
config=Config()

def matrix_to_quaternion(matrix):
    rotation = st.Rotation.from_matrix(matrix)
    return rotation.as_quat()  

def gram_schmidt(vectors):
    rotation = R.from_matrix(vectors.T)
    vectors = rotation.as_matrix().T
    return vectors


data = [] 
label = [] 

for _ in range(config.data_num):

    tx = np.random.uniform(config.tx_min, config.tx_max)
    ty = np.random.uniform(config.ty_min, config.ty_max)
    tz = np.random.uniform(config.tz_min, config.tz_max)
    rmin = np.random.uniform(config.rmin_min, config.rmin_max)
    rmaj = np.random.uniform(config.rmaj_min, config.rmaj_max)

    def g(alpha, beta):
        A = np.array([[1, 0, 0],
                      [0, np.cos(alpha), -np.sin(alpha)],
                      [0, np.sin(alpha), np.cos(alpha)]], dtype=np.float64)
        B = np.array([[np.cos(beta), 0, np.sin(beta)],
                      [0, 1, 0],
                      [-np.sin(beta), 0, np.cos(beta)]], dtype=np.float64)
        p = np.array([tx, ty, rmin + tz], dtype=np.float64)
        q = np.array([0, 0, rmaj], dtype=np.float64)
        R2 = A @ B
        t = R2 @ p + A @ q
        return R2, t


    R1 = np.random.randn(3, 3)
    R1 = gram_schmidt(R1).T.astype(np.float64)

    def generate_circles():
        alpha_circles_R = []
        alpha_circles_t = []
        beta_circles_R = []
        beta_circles_t = []
        alphas = np.linspace(0, 2 * np.pi, config.n_samples, dtype=np.float64)
        betas = np.linspace(0, 2 * np.pi, config.n_samples, dtype=np.float64)

        for alpha in config.alpha_base:
            alpha_circle_R = []
            alpha_circle_t = []
            for beta in betas:
                R2, t = g(alpha, beta)
                alpha_circle_R.append(R2 @ R1)
                alpha_circle_t.append(t)
            alpha_circles_R.append(alpha_circle_R)
            alpha_circles_t.append(alpha_circle_t)

        for beta in config.beta_base:
            beta_circle_R = []
            beta_circle_t = []
            for alpha in alphas:
                R2, t = g(alpha, beta)
                beta_circle_R.append(R2 @ R1)
                beta_circle_t.append(t)
            beta_circles_R.append(beta_circle_R)
            beta_circles_t.append(beta_circle_t)

        return np.array(alpha_circles_R, dtype=np.float64), np.array(alpha_circles_t, dtype=np.float64), np.array(
            beta_circles_R, dtype=np.float64), np.array(beta_circles_t, dtype=np.float64)
    alpha_circles_R, alpha_circles_t, beta_circles_R, beta_circles_t = generate_circles()


    data_i = []
    for alpha_index in range(len(config.alpha_base)):
        for i in range(config.n_samples):
            rotation = matrix_to_quaternion(alpha_circles_R[alpha_index, i, :, :]) 
            translation = alpha_circles_t[alpha_index, i, :].reshape(-1)
            data_alpha = np.concatenate([rotation, translation]) 
            data_i.append(data_alpha)
    for beta_index in range(len(config.beta_base)):
        for i in range(config.n_samples):
            rotation = matrix_to_quaternion(beta_circles_R[beta_index, i, :, :])
            translation = beta_circles_t[beta_index, i, :].reshape(-1)
            data_beta = np.concatenate([rotation, translation]) 
            data_i.append(data_beta)
    data.append(data_i)

    T = np.eye(4, 4, dtype=np.float64)
    T[:3, :3] = alpha_circles_R[0, 0, :, :]
    T[:3, 3] = alpha_circles_t[0, 0, :]
    T = np.linalg.inv(T).astype(np.float64)
    t = T @ np.array([0, 0, rmaj, 1], dtype=np.float64)
    label.append(t[:-1])

data = np.array(data, dtype=np.float64)
label = np.array(label, dtype=np.float64)

np.savez('dataset.npz', alpha_R=data[:,:300,:4], alpha_t=data[:,:300,4:], beta_R=data[:,300:,:4], beta_t=data[:,300:,4:], t=label)
dataset=np.load('dataset.npz')
print(dataset['alpha_R'].shape)
print(dataset['alpha_t'].shape)
print(dataset['beta_R'].shape)
print(dataset['beta_t'].shape)
print(dataset['t'].shape)