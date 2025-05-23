from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark import AccumulatorParam
import numpy as np
from scipy.spatial import KDTree
from collections import deque, defaultdict
import itertools
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
from pyspark import TaskContext

# ================== 初始化 Spark 会话 ==================
spark = SparkSession.builder \
    .appName("Distributed-DBSCAN-KDE") \
    .master("spark://192.168.192.163:7077") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext

# ================== 数据加载与清洗 ==================
df = spark.read.option("header", "true") \
    .option("mode", "DROPMALFORMED") \
    .csv("file:///root/data/Aggregation.csv")

df = df.withColumn("x", split(df["`x y label`"], " ").getItem(0).cast("double")) \
       .withColumn("y", split(df["`x y label`"], " ").getItem(1).cast("double")) \
       .withColumn("label", split(df["`x y label`"], " ").getItem(2).cast("int")) \
       .drop("`x y label`")

# ================== QuadTree 划分类 ==================
class QuadTreePartitioner:
    def __init__(self, max_points=100, max_depth=10):
        self.max_points = max_points
        self.max_depth = max_depth
        self.partitions = []

    def build(self, points, depth=0, bounds=None):
        if bounds is None:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            bounds = (min(xs), max(xs), min(ys), max(ys))  # (x_min, x_max, y_min, y_max)
        x_min, x_max, y_min, y_max = bounds

        if len(points) <= self.max_points or depth >= self.max_depth:
            self.partitions.append(points)
            return

        x_mid = (x_min + x_max) / 2
        y_mid = (y_min + y_max) / 2

        q1 = [p for p in points if p[0] <= x_mid and p[1] <= y_mid]
        q2 = [p for p in points if p[0] > x_mid and p[1] <= y_mid]
        q3 = [p for p in points if p[0] <= x_mid and p[1] > y_mid]
        q4 = [p for p in points if p[0] > x_mid and p[1] > y_mid]

        for q, new_bounds in zip([q1, q2, q3, q4], [
            (x_min, x_mid, y_min, y_mid),
            (x_mid, x_max, y_min, y_mid),
            (x_min, x_mid, y_mid, y_max),
            (x_mid, x_max, y_mid, y_max)
        ]):
            if q:
                self.build(q, depth + 1, new_bounds)

    def get_partitions(self):
        return self.partitions

# ================== 构建 QuadTree 分区 ==================
data_rdd = df.rdd.map(lambda row: (float(row.x), float(row.y), int(row.label)))
collected_points = data_rdd.collect()

qt = QuadTreePartitioner(max_points=300, max_depth=6)
qt.build(collected_points)
partitioned_data = qt.get_partitions()

# 并行化为自定义分区的 RDD
data_rdd = sc.parallelize(partitioned_data, len(partitioned_data)).flatMap(lambda p: p).cache()

# ================== KDE 估计 ==================
def compute_kde_density(coords):
    coords = np.array(coords)
    grid = GridSearchCV(KernelDensity(kernel='gaussian'),
                        {'bandwidth': np.linspace(0.05, 1.0, 20)},
                        cv=3)
    grid.fit(coords)
    kde = grid.best_estimator_
    log_densities = kde.score_samples(coords)
    densities = np.exp(log_densities)  # 从 log density 转为真实密度值
    return densities

# ================== 局部聚类（KDE + KDTree） ==================
def local_dbscan_with_kde(index, partition):
    points = list(partition)
    if not points:
        return []

    # --- 计算密度 ---
    coords = [(p[0], p[1]) for p in points]
    densities = compute_kde_density(coords)
    
    # --- 用密度的 75% 分位值反推 eps ---
    threshold_density = np.percentile(densities, 75)
    eps = 0.2 / (np.sqrt(threshold_density) + 1e-5)  # 防止除以零

    # --- minPts 通过 log(n) 来设置 ---
    minPts = int(np.log(len(points))+1)

    # --- 构建 KDTree 并进行聚类 ---
    coords = np.array(coords)
    kdtree = KDTree(coords)

    visited = [False] * len(points)
    cluster_labels = [-1] * len(points)
    cluster_id = 1

    # --- 通过高密度点开始聚类 ---
    for i in range(len(points)):
        if visited[i]:
            continue
        visited[i] = True
        neighbors = kdtree.query_ball_point(coords[i], r=eps)
        if len(neighbors) < minPts:
            cluster_labels[i] = -1
        else:
            cluster_labels[i] = cluster_id
            queue = deque(neighbors)
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = kdtree.query_ball_point(coords[j], r=eps)
                    if len(j_neighbors) >= minPts:
                        queue.extend(n for n in j_neighbors if not visited[n])
                if cluster_labels[j] == -1:
                    cluster_labels[j] = cluster_id
            cluster_id += 1

    # 通过分区索引为每个簇加上一个全局偏移量
    global_offset = index * 10000
    return [(float(p[0]), float(p[1]), int(p[2]), global_offset + int(cluster_labels[i])) for i, p in enumerate(points)]

# ================== 局部聚类执行 ==================
# 使用mapPartitionsWithIndex将分区索引传递给每个分区的聚类函数
local_clusters = data_rdd.mapPartitionsWithIndex(local_dbscan_with_kde).cache()

# ================== 输出为键值对形式 ((x, y), cluster_id) ==================
pair_rdd = local_clusters.map(lambda row: ((row[0], row[1]), row[3]))

# ===== 检查并收集冲突点 =====
conflicts = pair_rdd.groupByKey() \
    .mapValues(lambda ids: list(set(ids))) \
    .filter(lambda kv: len(kv[1]) > 1)

# ===== 检查并收集冲突点 =====
conflicts = pair_rdd.groupByKey() \
    .mapValues(lambda ids: list(set(ids))) \
    .filter(lambda kv: len(kv[1]) > 1)

# ===== 并查集结构用于全局合并 =====
class UnionFind:
    def __init__(self):
        self.parent = {}
    def find(self, u):
        if u not in self.parent:
            self.parent[u] = u
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]
    def union(self, u, v):
        pu, pv = self.find(u), self.find(v)
        if pu != pv:
            self.parent[pu] = pv

# 构建合并映射
def build_union_find(pairs):
    uf = UnionFind()
    for _, cluster_ids in pairs:
        for u, v in itertools.combinations(cluster_ids, 2):
            uf.union(u, v)
    return uf.parent

merge_relations = conflicts.collect()
merged_map = build_union_find(merge_relations)
bc_merged_map = sc.broadcast(merged_map)

# 应用合并结果
def assign_global_cluster(row):
    x, y, label, local_cluster = row
    global_cluster = bc_merged_map.value.get(local_cluster, local_cluster)
    return (x, y, label, global_cluster)

final_result = local_clusters.map(assign_global_cluster)

# ===== 计算 Pairwise F-score =====
def calculate_pairwise_f_score(true_labels, predicted_labels):
    TP = FP = FN = TN = 0
    n = len(true_labels)
    for i in range(n):
        for j in range(i + 1, n):
            same_true = (true_labels[i] == true_labels[j])
            same_pred = (predicted_labels[i] == predicted_labels[j])
            if same_true and same_pred:
                TP += 1
            elif not same_true and same_pred:
                FP += 1
            elif same_true and not same_pred:
                FN += 1
            else:
                TN += 1
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall    = TP / (TP + FN) if (TP + FN) > 0 else 0
    f_score   = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
    return f_score, precision, recall

local = final_result.collect()
true_labels = [row[2] for row in local]
pred_labels = [row[3] for row in local]

f, p, r = calculate_pairwise_f_score(true_labels, pred_labels)

print("Precision:", p)
print("Recall   :", r)
print("F-Score  :", f)

# ===== 保存结果到本地 =====
results = final_result.collect()
output_file = "/root/output_labels.txt"
with open(output_file, "w") as f:
    for x, y, true_label, pred_label in results:
        f.write(f"{x},{y},{pred_label}\n")

print(f"最终聚类结果已保存到本地文件：{output_file}")