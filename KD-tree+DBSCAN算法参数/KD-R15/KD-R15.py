from pyspark.sql import SparkSession
from pyspark.sql.functions import split
from pyspark import AccumulatorParam
import numpy as np
from scipy.spatial import KDTree
from collections import deque, defaultdict
import itertools

# 初始化 Spark 会话
spark = SparkSession.builder \
    .appName("Distributed-DBSCAN") \
    .master("spark://192.168.192.163:7077") \
    .getOrCreate()
spark.sparkContext.setLogLevel("WARN")
sc = spark.sparkContext

# ===== 数据加载与清洗 =====
df = spark.read.option("header", "true") \
    .option("mode", "DROPMALFORMED") \
    .csv("file:///root/data/R15.csv")

df = df.withColumn("x", split(df["`x y label`"], " ").getItem(0).cast("double")) \
       .withColumn("y", split(df["`x y label`"], " ").getItem(1).cast("double")) \
       .withColumn("label", split(df["`x y label`"], " ").getItem(2).cast("int")) \
       .drop("`x y label`")

# 转换为 RDD 格式并缓存
data_rdd = df.rdd.map(lambda row: (float(row.x), float(row.y), int(row.label)))
data_rdd = data_rdd.repartition(2).cache()  # 分区数

# ===== 广播参数 =====
eps = 0.5
minPts = 10
bc_eps = sc.broadcast(eps)
bc_minPts = sc.broadcast(minPts)

# ===== 分区内部局部聚类函数（带 KD-Tree） =====
def local_dbscan(partition):
    points = list(partition)
    if not points:
        return []

    coords = np.array([(p[0], p[1]) for p in points])
    labels_true = [p[2] for p in points]
    kdtree = KDTree(coords)

    visited = [False] * len(points)
    cluster_labels = [-1] * len(points)
    cluster_id = 1

    for i in range(len(points)):
        if visited[i]:
            continue

        visited[i] = True
        neighbors = kdtree.query_ball_point(coords[i], r=bc_eps.value)

        if len(neighbors) < bc_minPts.value:
            cluster_labels[i] = -1  # 噪声
        else:
            # 新建簇
            cluster_labels[i] = cluster_id
            queue = deque(neighbors)
            while queue:
                j = queue.popleft()
                if not visited[j]:
                    visited[j] = True
                    j_neighbors = kdtree.query_ball_point(coords[j], r=bc_eps.value)
                    if len(j_neighbors) >= bc_minPts.value:
                        queue.extend(j for j in j_neighbors if not visited[j])

                if cluster_labels[j] == -1:
                    cluster_labels[j] = cluster_id
            cluster_id += 1

    # 返回格式：(x, y, true_label, local_cluster_label)
    return [(float(p[0]), float(p[1]), int(p[2]), int(cluster_labels[i])) for i, p in enumerate(points)]

# 局部聚类（每个分区处理）
local_clusters = data_rdd.mapPartitions(local_dbscan).cache()

# 转换为 Pair RDD：((x, y), local_cluster_id)
pair_rdd = local_clusters.map(lambda row: ((row[0], row[1]), row[3]))

# Group by 坐标点，找出冲突标签
conflicts = pair_rdd.groupByKey() \
    .mapValues(lambda ids: list(set(ids))) \
    .filter(lambda kv: len(kv[1]) > 1)

# 构建并查集
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

# 构建全局合并关系
def build_union_find(pairs):
    uf = UnionFind()
    for _, cluster_ids in pairs:
        for u, v in itertools.combinations(cluster_ids, 2):
            uf.union(u, v)
    return uf.parent

# 收集合并关系（小数据量可以 driver 完成）
merge_relations = conflicts.collect()
merged_map = build_union_find(merge_relations)

# 广播映射关系
bc_merged_map = sc.broadcast(merged_map)

# 应用全局簇 ID
def assign_global_cluster(row):
    x, y, label, local_cluster = row
    global_cluster = bc_merged_map.value.get(local_cluster, local_cluster)
    return (x, y, label, global_cluster)

final_result = local_clusters.map(assign_global_cluster)

from sklearn.metrics import f1_score

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

# 从 final_result 中提取真实标签和预测标签
local = final_result.collect()
true_labels = [row[2] for row in local]
pred_labels = [row[3] for row in local]

# 调用 pairwise 计算函数
f, p, r = calculate_pairwise_f_score(true_labels, pred_labels)

print("Precision:", p)
print("Recall   :", r)
print("F-Score  :", f)

results = final_result.collect()

output_file = "/root/output_labels.txt"
with open(output_file, "w") as f:
    for x, y, true_label, pred_label in results:
        f.write(f"{x},{y},{pred_label}\n")

print(f"最终聚类结果已保存到本地文件：{output_file}")