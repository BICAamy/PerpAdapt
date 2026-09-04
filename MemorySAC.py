import numpy as np
import matplotlib.pyplot as plt
import torch
from torchdiffeq import odeint
import torch.nn.functional as F
from scipy.optimize import minimize


class Config:
    device=torch.device("cpu")
    t0=0
    tbar=2
    tpoints=10
    n_samples=100
    inputsize=3
    outputsize=2
    batchsize=32
    lr=0.01
    num_workers=2
    epoch=100
config=Config()

def variance_of_distances(center, points):
    distances = np.linalg.norm(points - center, axis=1)
    return np.var(distances)

def circumcenter(points):
    def objective(center):
        return variance_of_distances(center, points)

    initial_guess = np.mean(points, axis=0)

    result = minimize(objective, initial_guess, method='BFGS')

    return result.x

def model_func(vertices):
    center = circumcenter(vertices)
    v1 = vertices[1] - vertices[0]
    v2 = vertices[2] - vertices[0]
    normal = np.cross(v1, v2)
    radius = np.linalg.norm(center - vertices[0])
    return center, normal, radius

def distance_func(model, point):
    center, normal, radius = model
    
    normal = normal / np.linalg.norm(normal)
    
    point_to_plane_dist = np.abs(np.dot(point - center, normal))
    
    point_proj = point - np.dot(point - center, normal) * normal
    
    point_proj_to_center_dist = np.linalg.norm(point_proj - center)
    
    point_proj_to_circle_dist = np.abs(point_proj_to_center_dist - radius)
    
    distance = np.sqrt(point_to_plane_dist**2 + point_proj_to_circle_dist**2)
    
    return distance


class WeightedAveragePooling(torch.nn.Module):
    def __init__(self, dim):
        super(WeightedAveragePooling, self).__init__()
        self.weights = torch.nn.Parameter(torch.ones(dim))

    def forward(self, inputs):
        normalized_weights = F.softmax(self.weights, dim=0).view(-1,1,1,1)
        output = torch.sum(normalized_weights * inputs, dim=0)
        return output


class neuralODE(torch.nn.Module):
    def __init__(self):
        super(neuralODE, self).__init__()
    def fresh(self, gamma):
        self.gamma = gamma
    def forward(self, t, p):
        dpdt = -p + torch.pow(torch.sin(p + self.gamma), 2)
        return dpdt


class nmODE(torch.nn.Module):
    def __init__(self):
        super(nmODE,self).__init__()
        self.neuralODE = neuralODE()
        self.t = torch.tensor(np.linspace(config.t0, config.tbar, config.tpoints), device=config.device)
        self.w1 = torch.nn.Linear(config.inputsize,256)
        self.w2 = torch.nn.Linear(256,1024)
        self.pool = torch.nn.MaxPool1d(kernel_size=config.n_samples)
        self.w3 = torch.nn.Linear(1024+256,128)
        self.out =  torch.nn.Linear(128,config.outputsize)
        self.wpool = WeightedAveragePooling(config.tpoints)
        
    def forward(self, x):
        feature = F.relu(self.w1(x))

        self.neuralODE.fresh(feature)
        y0 = torch.zeros(x.shape[0],x.shape[1],256,device=config.device)
        yt = odeint(self.neuralODE, y0, self.t, method="rk4", options=dict(step_size=0.1))
        yt = self.wpool(yt)

        x = F.relu(self.w2(feature))
        x, _ = torch.max(x, dim=1, keepdim=True)
        x = x.repeat(1, feature.shape[1], 1)
        x = torch.cat((x,yt),dim=2)
        x = F.relu(self.w3(x))
        x = self.out(x)
        return x


class MemorySAC:
    def __init__(self, num_iterations, sample_size, distance_threshold, update_rate, pth):
        self.num_iterations = num_iterations
        self.sample_size = sample_size
        self.distance_threshold = distance_threshold
        self.update_rate = update_rate
        self.net = nmODE()
        self.net.load_state_dict(torch.load(pth, map_location=torch.device('cpu')))
        self.net.eval()

    def fit(self, data, model_func=model_func, distance_func=distance_func):
        n_points = len(data)
        memory_scores = self.net(torch.tensor(data.astype(np.float32).reshape(-1,data.shape[0],data.shape[1])))
        memory_scores = F.softmax(memory_scores,dim=2)
        memory_scores=memory_scores.squeeze()[:,1].squeeze().detach().numpy()
        
        best_model = None
        best_inliers = []
        distance_threshold = self.distance_threshold
        for iteration in range(self.num_iterations):
            sample_indices = np.random.choice(n_points, self.sample_size, p=memory_scores/np.sum(memory_scores), replace=False)
            sample = data[sample_indices]
            
            candidate_model = model_func(sample)
            
            distances = np.array([distance_func(candidate_model, point) for point in data])
            inliers = np.where(distances < distance_threshold)[0]
            
            for idx in range(n_points):
                memory_scores[idx] -= self.update_rate * distances[idx]
                if memory_scores[idx]<0: 
                    memory_scores[idx]=0
            
            if len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_model = candidate_model

        return best_model, best_inliers