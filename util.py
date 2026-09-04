import numpy as np

class Loader:
    def __init__(self):
        pass
    def load(self,dataset,calibration,pc_circle_num,
             seg_begin,seg_end,point_num,
             noise_type,noise_direction,noise_point,
             mean,std,min,max):
        dataset=np.load(dataset)
        if calibration=="PPC":
            alpha_R=dataset['alpha_R'][seg_begin:seg_end,:100:int(100/point_num),:]
            alpha_t=dataset['alpha_t'][seg_begin:seg_end,:100:int(100/point_num),:]
            beta_R=dataset['beta_R'][seg_begin:seg_end,:100:int(100/point_num),:]
            beta_t=dataset['beta_t'][seg_begin:seg_end,:100:int(100/point_num),:]
            t=dataset['t'][seg_begin:seg_end,:]
            alpha_point_num=point_num
            beta_point_num=point_num
            alpha_noise_point=noise_point
            beta_noise_point=noise_point

        elif calibration=="PC":
            alpha_R=dataset['alpha_R'][seg_begin:seg_end,::int(100/point_num),:]
            alpha_t=dataset['alpha_t'][seg_begin:seg_end,::int(100/point_num),:]
            alpha_point_num=point_num*3
            alpha_noise_point=noise_point*3
            if pc_circle_num==4:
                beta_R=dataset['beta_R'][seg_begin:seg_end,:100:int(100/point_num),:]
                beta_t=dataset['beta_t'][seg_begin:seg_end,:100:int(100/point_num),:]
                beta_point_num=point_num
                beta_noise_point=noise_point
            elif pc_circle_num==6:
                beta_R=dataset['beta_R'][seg_begin:seg_end,::int(100/point_num),:]
                beta_t=dataset['beta_t'][seg_begin:seg_end,::int(100/point_num),:]
                beta_point_num=point_num*3
                beta_noise_point=noise_point*3
            t=dataset['t'][seg_begin:seg_end,:]
            
        alpha_indices = np.random.choice(alpha_point_num, size=(seg_end-seg_begin, alpha_noise_point))
        beta_indices = np.random.choice(beta_point_num, size=(seg_end-seg_begin, beta_noise_point))


        if noise_type=="no":
            pass
        elif noise_type=="gaussian":
            if noise_direction=="alpha":
                noise = np.random.normal(mean, std, (alpha_t.shape[0],alpha_noise_point,alpha_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(alpha_noise_point):
                        alpha_t[i,alpha_indices[i,j]] += noise[i,j]
            elif noise_direction=="beta":
                noise = np.random.normal(mean, std, (beta_t.shape[0],beta_noise_point,beta_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(beta_noise_point):
                        beta_t[i,beta_indices[i,j]] += noise[i,j]
            elif noise_direction=="all":
                noise = np.random.normal(mean, std, (alpha_t.shape[0],alpha_noise_point,alpha_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(alpha_noise_point):
                        alpha_t[i,alpha_indices[i,j]] += noise[i,j]
                noise = np.random.normal(mean, std, (beta_t.shape[0],beta_noise_point,beta_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(beta_noise_point):
                        beta_t[i,beta_indices[i,j]] += noise[i,j]
        elif noise_type=="uniform":
            if noise_direction=="alpha":
                noise = np.random.uniform(min, max, (alpha_t.shape[0],alpha_noise_point,alpha_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(alpha_noise_point):
                        alpha_t[i,alpha_indices[i,j]] += noise[i,j]
            elif noise_direction=="beta":
                noise = np.random.uniform(min, max, (beta_t.shape[0],beta_noise_point,beta_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(beta_noise_point):
                        beta_t[i,beta_indices[i,j]] += noise[i,j]
            elif noise_direction=="all":
                noise = np.random.uniform(min, max, (alpha_t.shape[0],alpha_noise_point,alpha_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(alpha_noise_point):
                        alpha_t[i,alpha_indices[i,j]] += noise[i,j]
                noise = np.random.uniform(min, max, (beta_t.shape[0],beta_noise_point,beta_t.shape[2]))
                for i in range(seg_end-seg_begin):
                    for j in range(beta_noise_point):
                        beta_t[i,beta_indices[i,j]] += noise[i,j]
        return {"alpha_R":alpha_R,
                "alpha_t":alpha_t,
                "beta_R":beta_R,
                "beta_t":beta_t,
                "t":t}