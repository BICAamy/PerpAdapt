import numpy as np
from scipy.optimize import minimize
import sympy as sp
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import time
def project_to_plane(point, plane_point, plane_normal):
    plane_normal = plane_normal / np.linalg.norm(plane_normal)
    projected_point = point - (point - plane_point * plane_normal) * plane_normal
    return projected_point

def point_to_line_distance(point, line_point, line_direction):
    # line_direction = line_direction / np.linalg.norm(line_direction)
    projection = np.dot(point - line_point, np.array(line_direction)) * np.array(line_direction)
    distance_vector = point - line_point - projection
    distance = np.linalg.norm(distance_vector)
    return distance

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


def computecc(nc, cc, C, R):
    x, y, z, x0, y0, z0, a, b, c, t, Cx, Cy, Cz, r = sp.symbols('x y z x0 y0 z0 a b c t Cx Cy Cz r')

    x0, y0, z0 = cc  
    a, b, c = nc  
    Cx, Cy, Cz = C  
    r = R  

    x_expr = x0 + a * t
    y_expr = y0 + b * t
    z_expr = z0 + c * t

    circle_equation = (x - Cx) ** 2 + (y - Cy) ** 2 + (z - Cz) ** 2 - r ** 2

    equation = circle_equation.subs({x: x_expr, y: y_expr, z: z_expr})

    solutions = sp.solve(equation, t)

    intersection_points = [(x_expr.subs(t, sol), y_expr.subs(t, sol), z_expr.subs(t, sol)) for sol in solutions]
    intersection_points = np.array(intersection_points)
    point = np.mean(intersection_points, axis=0)

    return np.append(point, 1)


def circle_fit_error(params, data):
    x0, y0, z0, r = params
    error = 0
    for point in data:
        x, y, z = point
        error += np.abs((x - x0) ** 2 + (y - y0) ** 2 + (z - z0) ** 2 - r ** 2)
    return error


def step1(beta_circles):
    cx_list = []
    n_list = []
    for circle in beta_circles:
        data = np.column_stack((circle[:, 0],
                                circle[:, 1],
                                circle[:, 2]))
        result = minimize(circle_fit_error, x0=(0, 1, 0, 0.5), args=(data,), method='BFGS')
        x0, y0, z0, r = result.x
        cx = [x0, y0, z0]
        cx_list.append(cx)
        n = normal(cx)
        n_list.append(n)
    cx = np.mean(cx_list, axis=0)
    # print(f"cx:{cx}")
    n = np.mean(n_list, axis=0)
    # print(f"n:{n}")
    return cx, n


def step2(cx, n, alpha_circles):
    cc_list = []
    nc_list = []
    rmin_list = []
    for circle in alpha_circles:
        data = np.column_stack((circle[:, 0],
                                circle[:, 1],
                                circle[:, 2]))
        result = minimize(circle_fit_error, x0=(0, 1, 0, 0.5), args=(data,), method='BFGS')
        x0, y0, z0, r = result.x
        cc = [x0, y0, z0]
        cc_list.append(cc)
        rmin_list.append(r)
        nc = normal(cc)
        nc_list.append(nc)
    cc = np.mean(cc_list, axis=0)
    rmin = np.mean(rmin_list, axis=0)
    # print(rmin_list)
    # print(f"cc:{cc}")
    # print(f"rmin:{rmin}")
    projected_cx = project_to_plane(cx, cc, n)
    # print(f"projected_cx:{projected_cx}")
    c = (cc + projected_cx) / 2
    # print(f"c:{c}")
    return cc_list, nc_list, c


def step3(cc_list, nc_list, c, n):
    r_list = []
    for (cc, nc) in zip(cc_list, nc_list):
        r_list.append(point_to_line_distance(c, cc, nc))
    r = np.mean(r_list)
    # print(f"r_list:{r_list}")
    # print(f"rmaj:{r}")

    theta = np.linspace(0, 2 * np.pi, 100)
    n = n / np.linalg.norm(n)
    if n[2] == 0:
        v1 = np.array([0, 0, 1.])
    else:
        v1 = np.array([0., 0, -n[0] / n[2]])
    v1 -= np.dot(v1, n) * n
    v1 /= np.linalg.norm(v1)
    v2 = np.cross(n, v1)
    locus = c + r * (np.outer(np.cos(theta), v1) + np.outer(np.sin(theta), v2))
    return r


def step4(T, nc_list, cc_list, c, r):
    t = []
    for T_eachcircle, nc, cc in zip(T, nc_list, cc_list):
        cc_point = computecc(nc, cc, c, r)
        temp = []
        for T in T_eachcircle:
            T = np.linalg.inv(T)
            temp.append(np.matmul(T, cc_point))
        t.append(temp)
    t_alpha_0 = np.mean(t[0], axis=0)
    t_alpha_90 = np.mean(t[1], axis=0)
    t_alpha_090 = np.mean(t[2], axis=0)
    t_avg = np.mean((t_alpha_0, t_alpha_90, t_alpha_090), axis=0)
    # print(f"t_alpha_0:{t_alpha_0}")
    # print(f"t_alpha_90:{t_alpha_90}")
    # print(f"t_alpha_090:{t_alpha_090}")
    # print(f"t_avg:{t_avg}")
    return [t_alpha_0, t_alpha_90, t_alpha_090, t_avg]

class PC:
    def __init__(self,sac,pc_circle_num):
        self.sac=sac
        self.pc_circle_num=pc_circle_num
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
                beta_circles=beta_t[i,:,:].reshape(self.pc_circle_num-3,-1,3)
                alpha_circles=alpha_t[i,:,:].reshape(3,-1,3)
                rotations=alpha_R[i,:,:].reshape(3,-1,4)
            else:
                beta_circles=beta_t[i,:,:].reshape(self.pc_circle_num-3,-1,3).tolist()
                alpha_circles=alpha_t[i,:,:].reshape(3,-1,3).tolist()
                rotations=alpha_R[i,:,:].reshape(3,-1,4).tolist()

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

            T=[]
            for j in range(len(rotations)):
                Tcircle=[]
                for k in range(len(rotations[j])):
                    rotation=np.array(quaternion_to_rotation_matrix(rotations[j][k]))
                    position=alpha_circles[j][k]
                    Tcircle.append(getT(rotation, position))
                T.append(Tcircle)
            
            start_time = time.time()
            cx, n = step1(beta_circles)
            cc_list, nc_list, c = step2(cx, n, alpha_circles)
            r = step3(cc_list, nc_list, c, n)
            t_pred.append(step4(T, nc_list, cc_list, c, r)[-1][:-1])
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