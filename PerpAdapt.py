import numpy as np
from PPC import PPC
from PC import PC
from MemorySAC import MemorySAC
from RANSAC import RANSAC
from util import Loader

class Config:
    calibration="PPC" #PPC, PC
    sac="MemorySAC" #no, MemorySAC, RANSAC

    num_iterations=10
    sample_size=3
    distance_threshold=0.05
    update_rate=0.001
    memory_scorer="nmODE"
    pth="net.pth"

    dataset="dataset.npz"
    pc_circle_num=4 #4 6
    seg_begin=0
    seg_end=1
    point_num=100 #100 50 20 10
    noise_type="no" #no, gaussian, uniform
    noise_direction="all" #alpha, beta, all
    noise_point=10
    mean=0
    std=1
    min=-1
    max=1
config=Config()



if __name__ == '__main__':
    loader=Loader()
    dataset=loader.load(config.dataset,config.calibration,config.pc_circle_num,
                        config.seg_begin,config.seg_end,config.point_num,
                        config.noise_type,config.noise_direction,config.noise_point,
                        config.mean,config.std,config.min,config.max)

    if config.sac=="MemorySAC":
        sac=MemorySAC(config.num_iterations,config.sample_size,config.distance_threshold,config.update_rate, config.pth)
    elif config.sac=="RANSAC":
        sac=RANSAC(config.num_iterations,config.sample_size,config.distance_threshold)
    elif config.sac=="no":
        sac=None

    if config.calibration=="PPC":
        calibration=PPC(sac)
    elif config.calibration=="PC":
        calibration=PC(sac,config.pc_circle_num)
    calibration.calibrate(dataset)