import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import time


def find_foot(point1, direction_vector1, point2, direction_vector2):
    def line_parametric_equation(point, direction_vector):
        def equation(t):
            return point + t * direction_vector
        return equation

    line1 = line_parametric_equation(point1, direction_vector1)
    line2 = line_parametric_equation(point2, direction_vector2)

    A = np.array([[np.dot(direction_vector1, direction_vector1), -np.dot(direction_vector1, direction_vector2)],
                  [-np.dot(direction_vector1, direction_vector2), np.dot(direction_vector2, direction_vector2)]])
    B = np.array([np.dot(direction_vector1, point2 - point1), np.dot(direction_vector2, point2 - point1)])
    t1, t2 = np.linalg.solve(A, B)

    foot1 = line1(t1)
    foot2 = line2(t2)

    return foot1, foot2


def quaternion_to_rotation_matrix(q):
    x, y, z, w = q
    rotation_matrix = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return rotation_matrix


def getT(rotation, position):
    temp = np.concatenate((np.array(rotation), np.array(position).reshape(-1, 1)), axis=1)
    Tmat = np.concatenate((temp, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)
    return Tmat


def normal(cc):
    x, y, z = cc
    A = 2 * (x - 0)
    B = 2 * (y - 0)
    C = 2 * (z - 0)

    length = np.sqrt(A ** 2 + B ** 2 + C ** 2)

    normal_vector = np.array([A, B, C]) / length
    return normal_vector


def circle_fit_error(params, data):
    x0, y0, z0, r = params
    error = 0
    for point in data:
        x, y, z = point
        error += np.abs((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2)
    return error


def step1(alpha_circles, beta_circles):
    for circle in alpha_circles:
        data = np.column_stack((circle[:, 0],
                                circle[:, 1],
                                circle[:, 2]))
        result = minimize(circle_fit_error, x0=(0, 1, 0, 0.5), args=(data,), method='BFGS')
        x0, y0, z0, r = result.x
        c_alpha = [x0, y0, z0]
        n_alpha = normal(c_alpha)
        # print(f"c_alpha:{c_alpha}")
        # print(f"n_alpha:{n_alpha}")
    for circle in beta_circles:
        data = np.column_stack((circle[:, 0],
                                circle[:, 1],
                                circle[:, 2]))
        result = minimize(circle_fit_error, x0=(0, 1, 0, 0.5), args=(data,), method='BFGS')
        x0, y0, z0, r = result.x
        c_beta = [x0, y0, z0]
        n_beta = normal(c_beta)
        # print(f"c_beta:{c_beta}")
        # print(f"n_beta:{n_beta}")
    return c_alpha, n_alpha, c_beta, n_beta


def step2(c_alpha, n_alpha, c_beta, n_beta):
    c_alpha=np.array(c_alpha)
    n_alpha=np.array(n_alpha)
    c_beta=np.array(c_beta)
    n_beta=np.array(n_beta)
    cc, _ = find_foot(c_alpha, n_alpha, c_beta, n_beta)
    return cc


def step3(T, cc):
    cc=np.append(cc,1)
    t = []
    for T_eachcircle in zip(T):
        temp = []
        for T in T_eachcircle:
            T = np.linalg.inv(T)
            temp.append(np.matmul(T, cc))
        t.append(temp)
    t = np.mean(t[0][0], axis=0)
    # print(f"t:{t}")
    return t


class PPC:
    def __init__(self,sac):
        self.sac=sac
    def calibrate(self,dataset):
        alpha_R=dataset['alpha_R']
        alpha_t=dataset['alpha_t']
        beta_R=dataset['beta_R']
        beta_t=dataset['beta_t']
        t=dataset['t']
        t_pred=[]
        sac_time_cost=[]
        calibration_time_cost=[]
        for i in range(t.shape[0]):

            if self.sac is None:
                beta_circles=beta_t[i,:,:].reshape(1,-1,3)
                alpha_circles=alpha_t[i,:,:].reshape(1,-1,3)
                rotations=alpha_R[i,:,:].reshape(1,-1,4)
            else:
                beta_circles=beta_t[i,:,:].reshape(1,-1,3).tolist()
                alpha_circles=alpha_t[i,:,:].reshape(1,-1,3).tolist()
                rotations=alpha_R[i,:,:].reshape(1,-1,4).tolist()

                start_time = time.time()
                for j in range(len(beta_circles)):
                    beta_circle=np.array(beta_circles[j])
                    _,inliers=self.sac.fit(beta_circle)
                    beta_circles[j]=beta_circle[inliers]
                
                for j in range(len(alpha_circles)):
                    alpha_circle=np.array(alpha_circles[j])
                    _,inliers=self.sac.fit(alpha_circle)
                    alpha_circles[j]=alpha_circle[inliers]

                    rotations[j]=np.array(rotations[j])[inliers]
                end_time = time.time()
                sac_time_cost.append(end_time-start_time)

                beta_circles=np.array(beta_circles)
                alpha_circles=np.array(alpha_circles)
                rotations=np.array(rotations)

            T=[]
            for j in range(rotations.shape[0]):
                Tcircle=[]
                for k in range(rotations.shape[1]):
                    rotation=np.array(quaternion_to_rotation_matrix(rotations[j,k,:]))
                    position=alpha_circles[j,k,:]
                    Tcircle.append(getT(rotation, position))
                T.append(Tcircle)

            start_time = time.time()
            c_alpha, n_alpha, c_beta, n_beta = step1(alpha_circles, beta_circles)
            cc = step2(c_alpha, n_alpha, c_beta, n_beta)
            t_pred.append(step3(T, cc)[:-1])
            end_time = time.time()
            calibration_time_cost.append(end_time-start_time)

        t_pred=np.array(t_pred)
        # print(f"MAE:{mean_absolute_error(t,t_pred)}")
        print(f"MSE:{mean_squared_error(t,t_pred)}")
        # print(f"R2:{r2_score(t,t_pred)}")
        # print(f"sac_time_cost:{np.mean(sac_time_cost)}")
        # print(f"calibration_time_cost:{np.mean(calibration_time_cost)}")
        if self.sac is None:
            print(f"calibration_time_cost:{np.mean(calibration_time_cost)}")
        else:
            print(f"all time:{np.mean(sac_time_cost)+np.mean(calibration_time_cost)}")