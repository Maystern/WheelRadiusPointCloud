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

# 设置区域 ---------- begin ----------
corner_example_name = "round_corner_example" # csv文件名称，需要放在./data文件夹下
pic_store_name = "pic" # 存储输出图片文件夹，需要放在项目根目录下
expected_interval_count = 10 # 对 y 汇聚因数（邻近多少个y计算一次R）
downsampling_factor = 15 # 下采样因数
line_fitting_window_size = 15 # 滑动窗口点数
random_seed = 42 # 随机种子
dbscan_eps = 0.1
dbscan_min_sample = 10
# 设置区域 ---------- end ----------

store_pic_folder_name = "./" + pic_store_name

if not os.path.exists(store_pic_folder_name):
    os.makedirs(store_pic_folder_name)
    
def intersection_point(m1, b1, m2, b2):
    coefficients_matrix = np.array([[m1, -1], [m2, -1]])
    constants_matrix = np.array([-b1, -b2])
    solution = np.linalg.solve(coefficients_matrix, constants_matrix)
    return solution[0], solution[1]

def calculate_angle_bisector_slope(a, b):
    
    # bisector_slope = (a * b + np.sqrt(a * a * b * b + a * a + b * b + 1) - 1) / (a + b)
    bisector_slope = (a * b - np.sqrt(a * a * b * b + a * a + b * b + 1) - 1) / (a + b)

    return bisector_slope

def calculate_normal_vector(id, data):
    id_return = (2 * id + line_fitting_window_size + 1) // 2
    coefficients = np.polyfit(data[:, 0], data[:, 1], 1)
    lope, intercept = coefficients
    return_x = -lope
    return_y = 1
    len = np.sqrt(return_x * return_x +  return_y * return_y)
    return np.array([id_return, return_x / len, return_y / len], dtype=float)
    

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
    assert len(most_common_labels) == 2
    
    plt.scatter(downsampled_data[:, 0], downsampled_data[:, 1], color='blue', marker='o', label='Noise', s=10)
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
        if  min(cluster_points[:, 0]) <= 8.5: 
            x1 = 7
            x2 = 9
        else:
            x1 = 8
            x2 = 13
        y1 = slope * x1 + intercept
        y2 = slope * x2 + intercept
        plt.plot([x1, x2], [y1, y2], label="LineFit#" + str(i), linestyle = '--', color=line_colors[i])
    
    cross_point = intersection_point(ms[0], bs[0], ms[1], bs[1])
    formatted_point = f'({cross_point[0]:.2f}, {cross_point[1]:.2f})'
    plt.scatter(cross_point[0], cross_point[1], color='black', marker='o', label='CrossPoint', s=20)
    plt.text(cross_point[0], cross_point[1], formatted_point, ha='right', va='bottom')
    
    angle_bisector_slope = calculate_angle_bisector_slope(ms[0], ms[1])
    angle_bisector_intercept = cross_point[1] - angle_bisector_slope * cross_point[0]
    
    x1 = cross_point[0]
    x2 = 9
    y1 = angle_bisector_slope * x1 + angle_bisector_intercept
    y2 = angle_bisector_slope * x2 + angle_bisector_intercept
    plt.plot([x1, x2], [y1, y2], label="AngleBisector", linestyle = '--', color='black')
    plt.legend()
    plt.savefig(os.path.join(store_this_interval_path, 'new_scatter_plot.png'))
    plt.clf()
    return 0

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
max_test_count = 100
for i in tqdm(range(0, len(unique_values), expected_interval_count), desc="Processing"):
    interval_id += 1
    # print("------------[the " + str(interval_id) + "th interval]------------")
    idx = np.where(np.isin(data_array[:, 1], unique_values[i:i + expected_interval_count]))
    solve_data = data_array[idx][:, [0, 1, 2]]
    estimated_R.append(solve(interval_id, solve_data))
    if interval_id >= max_test_count:
        break