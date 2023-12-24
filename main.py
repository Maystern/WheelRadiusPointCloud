import csv
from itertools import islice
import numpy as np
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN
from collections import Counter
import torch
import torch.optim as optim


# 设置区域 ---------- begin ----------

"""
corner_example_name = "clear_cloud_for_test" # csv文件名称, 需要放在./data文件夹下
pic_store_name = "pic" # 存储输出图片文件夹，需要放在项目根目录./下
result_store_name = "results" # 存储输出结果文件夹，需要放在根目录./下
expected_interval_count = 50 # 对 y 汇聚因数, 邻近多少个y计算一次R
downsampling_factor = 5 # 下采样因数
line_fitting_window_size = 5 # 滑动窗口点数
random_seed = 42 # 随机种子
dbscan_eps = 0.05 # dbscan 邻域的半径大小
dbscan_min_sample = 5 # dbscan 簇最小成员数
learning_rate = 1e-2 # 梯度下降的学习率
num_iterations = 200 # 梯度下降执行次数
prior_circle_dis = 1 # 先验的圆半径
"""


corner_example_name = "200r2"
pic_store_name = "pic"
result_store_name = "results"
expected_interval_count = 50
downsampling_factor = 5
line_fitting_window_size = 5
random_seed = 42
dbscan_eps = 0.01
dbscan_min_sample = 5
learning_rate = 0.01
num_iterations = 200
prior_circle_dis = 1



# 设置区域 ---------- end ----------

store_pic_folder_name = os.path.join( "./" + pic_store_name, corner_example_name)
store_result_folder_name = os.path.join("./" + result_store_name, corner_example_name)

if not os.path.exists(store_pic_folder_name):
    os.makedirs(store_pic_folder_name)

if not os.path.exists(store_result_folder_name):
    os.makedirs(store_result_folder_name)
    
with open(os.path.join(store_result_folder_name, "settings.txt"), 'w') as file:
    file.write(f"corner_example_name = \"{corner_example_name}\"\n")
    file.write(f"pic_store_name = \"{pic_store_name}\"\n")
    file.write(f"result_store_name = \"{result_store_name}\"\n")
    file.write(f"expected_interval_count = {expected_interval_count}\n")
    file.write(f"downsampling_factor = {downsampling_factor}\n")
    file.write(f"line_fitting_window_size = {line_fitting_window_size}\n")
    file.write(f"random_seed = {random_seed}\n")
    file.write(f"dbscan_eps = {dbscan_eps}\n")
    file.write(f"dbscan_min_sample = {dbscan_min_sample}\n")
    file.write(f"learning_rate = {learning_rate}\n")
    file.write(f"num_iterations = {num_iterations}\n")
    file.write(f"prior_circle_dis = {prior_circle_dis}\n")
    
def intersection_point(m1, b1, m2, b2):
    coefficients_matrix = np.array([[m1, -1], [m2, -1]])
    constants_matrix = np.array([-b1, -b2])
    solution = np.linalg.solve(coefficients_matrix, constants_matrix)
    return solution[0], solution[1]

def calculate_angle_bisector_slope(m1, m2):
    # bisector_slope = (m1 * m2 + np.sqrt(m1 * m1 * m2 * m2 + m1 * m1 + m2 * m2 + 1) - 1) / (m1 + m2)
    bisector_slope = (m1 * m2 - np.sqrt(m1 * m1 * m2 * m2 + m1 * m1 + m2 * m2 + 1) - 1) / (m1 + m2)
    return bisector_slope

def calculate_direction_vector_from_slope(slope):
    delta_x = 1
    delta_y = slope
    len = np.sqrt(delta_x * delta_x + delta_y * delta_y)
    return delta_x / len, delta_y / len

def calculate_normal_vector(id, data):
    id_return = (2 * id + line_fitting_window_size + 1) // 2
    coefficients = np.polyfit(data[:, 0], data[:, 1], 1)
    lope, intercept = coefficients
    return_x = -lope
    return_y = 1
    len = np.sqrt(return_x * return_x +  return_y * return_y)
    return np.array([id_return, return_x / len, return_y / len], dtype=float)

def calculate_point_to_line_perpendicular(x0, y0, k, b):
    x = (x0 + k * y0 - k * b) / (k * k + 1)
    y = (k * k * y0 + k * x0 + b) / (k * k + 1)
    return x, y

def calculate_distance(x1, y1, x2, y2, device = "numpy"):
    if device == "numpy":
        return np.sqrt((x1 - x2) ** 2 +  (y1 - y2) ** 2)
    elif device == "torch":
        return torch.sqrt((x1 - x2) ** 2 +  (y1 - y2) ** 2) 

def solve(interval_id, solve_data):
    store_this_interval_path = os.path.join(store_pic_folder_name, 'batch_' + str(interval_id))
    if not os.path.exists(store_this_interval_path):
        os.makedirs(store_this_interval_path)
    unique_x_values = np.unique(solve_data[:, 0])
    unique_y_values = np.unique(solve_data[:, 1])
    num = np.array([0] * len(unique_x_values))
    sum = np.array([0] * len(unique_x_values), dtype=float)
    cnt = 0
    for y_value in unique_y_values:
        cnt = cnt + 1
        sample = solve_data[np.where(solve_data[:, 1] == y_value)]
        indice = np.where(np.in1d(sample[:, 0], unique_x_values))
        num[indice] += 1
        sum[indice] += sample[:, 2]
    
    not_zero_indice = np.where(num[:] != 0)
    unique_x_values = unique_x_values[not_zero_indice]
    num = num[not_zero_indice]
    sum = sum[not_zero_indice]
    
    result = np.column_stack((unique_x_values, sum / num))
    downsampled_data = result[::downsampling_factor]
    sorted_indices = np.argsort(downsampled_data[:, 0])
    downsampled_data = downsampled_data[sorted_indices]
    
    plt.scatter(downsampled_data[:, 0], downsampled_data[:, 1], color='blue', marker='o', label='Scatter Plot', s=10)
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.savefig(os.path.join(store_this_interval_path, 'scatter_plot.png'))
    plt.clf()
    
    nor_vecs = []
    for i in range(len(downsampled_data) - line_fitting_window_size + 1):
        select_points = downsampled_data[i: i + line_fitting_window_size]
        nor_vecs.append(calculate_normal_vector(i, select_points))
    nor_vecs = np.array(nor_vecs)
    
    plt.scatter(nor_vecs[:, 1], nor_vecs[:, 2], color='blue', marker='o', label='Scatter Plot', s=10)
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.legend()
    plt.savefig(os.path.join(store_this_interval_path, 'nor_vecs_plot.png'))
    plt.clf()
    
    X = nor_vecs[:, 1:]
    dbscan = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_sample)
    labels = dbscan.fit_predict(X)
    unique_labels = np.unique(labels)
    for label in unique_labels:
        cluster_points = X[labels == label]
        if label == -1:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], marker='x', label=f'Noise')
        else:
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f'Cluster {label + 1}')
    
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.savefig(os.path.join(store_this_interval_path, 'DBSCAN_clustering_result.png'))
    plt.clf()
    
    filtered_labels = [label for label in labels if label != -1]
    label_counts = Counter(filtered_labels)
    most_common_labels = label_counts.most_common(2)
    assert len(most_common_labels) == 2, "Please adjust hyperparameters to prevent the clustering result from being too bad."
    # 无法分出两个请调整超参数，如调高 expected_interval_count
    
    plt.scatter(result[:, 0], result[:, 1], color='blue', marker='o', label='Noise', s=10)
    plt.axis('equal')
    plt.title('Scatter Plot')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    colors = ["green", "red"]
    line_colors = ['brown', 'magenta']
    ms = []
    bs = []
    
    for i, most_common_label in enumerate(most_common_labels):
        cluster_points_id = nor_vecs[labels == most_common_label[0]][:, 0].tolist()
        cluster_points_idx = [int(x) for x in cluster_points_id]
        cluster_points = downsampled_data[cluster_points_idx]
        plt.scatter(cluster_points[:, 0], cluster_points[:, 1], color = colors[i], marker='o', label = 'Class#' + str(i), s=10)
        coefficients = np.polyfit(cluster_points[:, 0], cluster_points[:, 1], 1)
        slope, intercept = coefficients
        ms.append(slope)
        bs.append(intercept)
    
    cross_point = intersection_point(ms[0], bs[0], ms[1], bs[1])
    formatted_point = f'({cross_point[0]:.2f}, {cross_point[1]:.2f})'
    plt.scatter(cross_point[0], cross_point[1], color='black', marker='o', label='CrossPoint', s=20)
    plt.text(cross_point[0], cross_point[1], formatted_point, ha='right', va='bottom')
    
    for i, most_common_label in enumerate(most_common_labels):
        cluster_points_id = nor_vecs[labels == most_common_label[0]][:, 0].tolist()
        cluster_points_idx = [int(x) for x in cluster_points_id]
        cluster_points = downsampled_data[cluster_points_idx]
        x1 = cross_point[0]
        x2 = min(cluster_points[:, 0])
        x3 = max(cluster_points[:, 0])
        if abs(x2 - x1) < abs(x3 - x1):
            x2 = x3
        y1 = ms[i] * x1 + bs[i]
        y2 = ms[i] * x2 + bs[i]
        plt.plot([x1, x2], [y1, y2], label="LineFit#" + str(i), linestyle = '--', color=line_colors[i])
    
    angle_bisector_slope = calculate_angle_bisector_slope(ms[0], ms[1])
    angle_bisector_intercept = cross_point[1] - angle_bisector_slope * cross_point[0]
    
    x1 = cross_point[0]
    x2 = 9
    y1 = angle_bisector_slope * x1 + angle_bisector_intercept
    y2 = angle_bisector_slope * x2 + angle_bisector_intercept
    # plt.plot([x1, x2], [y1, y2], label="AngleBisector", linestyle = '--', color='black')
    
    dv_x, dv_y = calculate_direction_vector_from_slope(angle_bisector_slope)    

    sgd_dis = torch.tensor([prior_circle_dis], requires_grad=True, dtype=float)
        
    optimizer = optim.Adam([sgd_dis], lr=learning_rate)
    num_iter = num_iterations
    losses = []
    best_sed_dis = None
    best_loss = None
    for _ in range(num_iter):
        optimizer.zero_grad()
        circle_center_x = cross_point[0] + sgd_dis * dv_x
        circle_center_y = cross_point[1] + sgd_dis * dv_y
        xp1, yp1 = calculate_point_to_line_perpendicular(circle_center_x, circle_center_y, ms[0], bs[0])
        xp2, yp2 = calculate_point_to_line_perpendicular(circle_center_x, circle_center_y, ms[1], bs[1])
        dis = calculate_distance(circle_center_x, circle_center_y, xp1, yp1, "torch")
        loss = torch.tensor(0.0, requires_grad=True)
        cnt = 0
        # tmp = []
        for data in result:
            if data[0] >= min(xp1, xp2) and data[0] <= max(xp1, xp2):
                loss1 = calculate_distance(data[0], data[1], circle_center_x, circle_center_y, "torch") - dis
                loss = loss + loss1 * loss1
                # tmp.append(float(loss1))
                cnt = cnt + 1
        loss = loss / cnt
        detached_loss = float(loss.detach())
        losses.append(detached_loss)
        if best_loss is None or detached_loss < best_loss:
            best_loss = detached_loss
            best_sed_dis = float(sgd_dis.detach())
        loss.backward()
        optimizer.step()
        # print("loss: ", float(loss.detach()), "sgd_dis: ", float(sgd_dis.detach().float()), "cnt: ", cnt)
        # print(tmp, "\n")
    
    # sgd_dis = torch.tensor([best_sed_dis], requires_grad=True)
    
    circle_center_x = cross_point[0] + best_sed_dis * dv_x
    circle_center_y = cross_point[1] + best_sed_dis * dv_y
    xp1, yp1 = calculate_point_to_line_perpendicular(circle_center_x, circle_center_y, ms[0], bs[0])
    xp2, yp2 = calculate_point_to_line_perpendicular(circle_center_x, circle_center_y, ms[1], bs[1])
    distance = calculate_distance(circle_center_x, circle_center_y, xp1, yp1, "numpy")
    plt.scatter(circle_center_x, circle_center_y, color='black', marker='o', label='CircleCenter', s=20)
    plt.scatter(xp1, yp1, color='black', marker='o', label='PerpendicularPoint1', s=20)
    plt.scatter(xp2, yp2, color='black', marker='o', label='PerpendicularPoint2', s=20)
    circle = plt.Circle((circle_center_x, circle_center_y), distance , color='black', fill=False)
    plt.gcf().gca().add_artist(circle)
    plt_lightgreen_x = []
    plt_lightgreen_y = []
    for data in result:
        if data[0] >= min(xp1, xp2) and data[0] <= max(xp1, xp2):
            plt_lightgreen_x.append(data[0])
            plt_lightgreen_y.append(data[1])
    plt.scatter(plt_lightgreen_x, plt_lightgreen_y,  color='lightgreen', marker='o', s=10, label = "TrainPoints")
            
    plt.title(f"r = {float(distance)} loss = {float(best_loss):5e}")
    plt.scatter(xp1, yp1, color='black', marker='o', label='CutPoint1', s=20)
    plt.scatter(xp2, yp2, color='black', marker='o', label='CutPoint2', s=20)
    # plt.legend()
    plt.savefig(os.path.join(store_this_interval_path, 'new_scatter_plot.png'))
    plt.clf()
    
    return distance

example_csv_path = os.path.join("./data", corner_example_name + ".csv")
example_npy_path = os.path.join("./data", corner_example_name + ".npy")

try:
    data_array = np.load(example_npy_path)
    print("the " + corner_example_name + ".npy file already exists!")
except:
    print("the " + corner_example_name + ".npy file does not exist! Creating...")
    with open(example_csv_path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=' ')
        next(reader)
        total_rows = int(next(reader)[0])
        data_array = np.array(list(tqdm(reader, total=total_rows, desc="Converting to NumPy array")), dtype=float)
        np.save(example_npy_path, data_array)
        print("the " + corner_example_name + ".npy file has been created successfully!")

unique_values, counts = np.unique(data_array[:, 1], return_counts=True)
interval_id = 0
estimated_R = []
# max_test_count = 1
for i in tqdm(range(0, len(unique_values), expected_interval_count), desc="Processing"):
    interval_id += 1
    # print("------------[the " + str(interval_id) + "th interval]------------")
    idx = np.where(np.isin(data_array[:, 1], unique_values[i:i + expected_interval_count]))
    solve_data = data_array[idx][:, [0, 1, 2]]
    estimated_R.append(solve(interval_id, solve_data))
    # if interval_id >= max_test_count:
    #     break

csv_file_path = os.path.join(store_result_folder_name, "results.csv")
with open(csv_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['Block', 'Values'])  # 写入列名
    for epoch, value in enumerate(estimated_R, start=1):
        csv_writer.writerow([epoch, value])
print(f"results has been saved in the CSV file '{csv_file_path}' successfully!")

plt.hist(estimated_R, bins=20, density=True, alpha=0.7, color='blue')
plt.title('Histogram of Partial Normal Distribution')
plt.xlabel('Values')
plt.ylabel('Frequency')
plt.savefig(os.path.join(store_result_folder_name, "results_box.png"))
plt.show()
plt.clf()