import org.apache.spark.sql.SparkSession
import org.apache.spark.rdd.RDD
import scala.collection.mutable
import com.github.davidmoten.rtree._
import org.apache.log4j.{Level, Logger}
import java.io.{File, PrintWriter}

case class Point(id: Long, x: Double, y: Double)
case class LabeledPoint(id: Long, x: Double, y: Double, label: Int, pointType: String)

object PDR_DBSCAN_Spark {

  def main(args: Array[String]): Unit = {

    val rootLogger = Logger.getRootLogger
    rootLogger.setLevel(Level.WARN)

    val spark = SparkSession.builder.appName("PDR-DBSCAN").getOrCreate()
    val sc = spark.sparkContext

    // 一次性读取数据并提取字段
    val raw = sc.textFile("file:///root/data/Aggregation.csv")
    val header = raw.first()
    val dataWithoutHeader = raw.filter(row => row != header)

    // 处理数据，提取坐标和标签
    val parsedData = dataWithoutHeader.flatMap { line =>
      val parts = line.trim.split("\\s+")
      if (parts.length >= 3) {
        try {
          val x = parts(0).toDouble
          val y = parts(1).toDouble
          val label = parts(2).toInt
          Some((x, y, label))
        } catch {
          case _: NumberFormatException => None
        }
      } else None
    }

    // 添加唯一 ID
    val dataWithId = parsedData.zipWithIndex().map {
      case ((x, y, label), idx) =>
        val point = Point(idx.toLong, x, y)
        val labeledPoint = LabeledPoint(idx.toLong, x, y, label, "default")
        (point, labeledPoint)
    }.cache() // 缓存结果，避免重复计算

    // === 聚类输入 ===
    val points = dataWithId.map(_._1)

    val minPts = 3

    // === 阶段1：二次区域划分 ===
    val subAreas: Array[RDD[Point]] = SplitUtil.partition(points, minRect = 10)

    // === 阶段2：局部聚类 ===
    val localLabeled: RDD[LabeledPoint] = sc.union(subAreas.map(area => DRDBSCAN.localClustering(area, minPts)))

    // === 阶段3：融合标签 ===
    val globalLabels = LabelFusion.mergeLabels(localLabeled)

    // === 阶段3.5：融合标签写回 txt（模仿 fscore.txt 写法）===

    // 构造 (id, LabeledPoint) 方便按 ID join
    val originalById = dataWithId.map { case (_, p) => (p.id, p) }
    val fusedById = globalLabels.map(p => (p.id, p))

    // join 原始和融合结果，提取 x, y, 原始 label, 融合 label
    val joined = originalById.join(fusedById).map {
      case (id, (original, fused)) =>
        val x = original.x
        val y = original.y
        val trueLabel = original.label
        val fusedLabel = fused.label
        s"$x $y $trueLabel $fusedLabel"
    }

    // 保存到 TXT 文件（用 PrintWriter）
    val outputPath = "/root/data/fused_labels.txt"
    val pf = new PrintWriter(new File(outputPath))
    joined.collect().foreach(pf.println)
    pf.close()


    // === 阶段4：计算 F 值 ===
    val trueLabels: RDD[LabeledPoint] = dataWithId.map(_._2)

    val fScore = FScoreCalculator.calculatePairwiseFScore(trueLabels, globalLabels)
    println(s"F-Score: $fScore")

    // 将 F 值写入文件
    val outputPath2 = "/root/data/fscore.txt" //
    val pw = new PrintWriter(new File(outputPath2))
    pw.write(s"F-Score: $fScore\n")
    pw.close()

    spark.stop()
  }
}

object SplitUtil {

  val step = 0.4 // 二次划分步长
  val ptDiffThresholdLow = 0.4
  val ptDiffThresholdHigh = 0.6

  def partition(points: RDD[Point], minRect: Double): Array[RDD[Point]] = {
    // 坐标偏移
    val (xmin, xmax, ymin, ymax) = points.aggregate((Double.MaxValue, Double.MinValue, Double.MaxValue, Double.MinValue))(
      (acc, p) => (
        math.min(acc._1, p.x), math.max(acc._2, p.x),
        math.min(acc._3, p.y), math.max(acc._4, p.y)
      ),
      (a, b) => (
        math.min(a._1, b._1), math.max(a._2, b._2),
        math.min(a._3, b._3), math.max(a._4, b._4)
      )
    )

    val offsetX = -xmin
    val offsetY = -ymin
    val shiftedPoints = points.map(p => Point(p.id, p.x + offsetX, p.y + offsetY)).cache()

    val totalCount = shiftedPoints.count()
    val avgCount = totalCount / ((xmax - xmin) * (ymax - ymin) / (minRect * minRect))

    // 主递归函数（自顶向下二分法）
    def splitRegion(data: RDD[Point], x0: Double, x1: Double, y0: Double, y1: Double): Array[RDD[Point]] = {
      val width = x1 - x0
      val height = y1 - y0

      // 停止划分条件
      if (width <= minRect || height <= minRect || data.count() <= 2 * avgCount) {
        return Array(data)
      }

      // 向四周扩展 minRect
      val extendedX0 = x0 - minRect
      val extendedX1 = x1 + minRect
      val extendedY0 = y0 - minRect
      val extendedY1 = y1 + minRect

      // 尝试在x方向做二分
      val xMid = (x0 + x1) / 2

      val leftRegion = data.filter(p => p.x >= extendedX0 && p.x <= xMid)
      val rightRegion = data.filter(p => p.x > xMid && p.x <= extendedX1)

      val countLeft = leftRegion.count()
      val countRight = rightRegion.count()
      val ptDiff = math.abs(countLeft - countRight).toDouble / (countLeft + countRight)

      if (ptDiff >= ptDiffThresholdLow && ptDiff <= ptDiffThresholdHigh) {
        // ptDiff 满足，继续递归
        splitRegion(leftRegion, extendedX0, xMid, extendedY0, extendedY1) ++
          splitRegion(rightRegion, xMid, extendedX1, extendedY0, extendedY1)
      } else {
        // ptDiff 不满足，进入算法3的试探式拆分
        splitBalanced(data, extendedX0, extendedX1, extendedY0, extendedY1, minRect)
      }
    }

    // 修改后的 splitBalanced
    def splitBalanced(data: RDD[Point], x0: Double, x1: Double, y0: Double, y1: Double, minRect: Double): Array[RDD[Point]] = {
      var splitLen = 0.0
      val maxTry = 20
      var i = 0
      while (i < maxTry) {
        val splitX = x0 + (x1 - x0) / 2 + splitLen
        val left = data.filter(p => p.x >= x0 && p.x <= splitX)
        val right = data.filter(p => p.x > splitX && p.x <= x1)
        val countLeft = left.count()
        val countRight = right.count()
        val ptDiff = math.abs(countLeft - countRight).toDouble / (countLeft + countRight)

        if (ptDiff >= ptDiffThresholdLow && ptDiff <= ptDiffThresholdHigh) {
          // 扩展区域并继续递归
          return splitRegion(left, x0 - minRect, splitX + minRect, y0 - minRect, y1 + minRect) ++
            splitRegion(right, splitX - minRect, x1 + minRect, y0 - minRect, y1 + minRect)
        }
        splitLen += step * minRect
        i += 1
      }
      // 如果找不到合适的拆分方式，保留原区域
      Array(data)
    }

    // 启动初始拆分
    splitRegion(shiftedPoints, 0.0, xmax - xmin, 0.0, ymax - ymin)
  }
}

object DRDBSCAN {
  def localClustering(points: RDD[Point], minPts: Int): RDD[LabeledPoint] = {
    points.mapPartitions { partition =>
      val partitionPoints = partition.toArray

      // 1. 计算自适应 eps（仅对该分区的点）
      val eps = EpsCalculator.computeEps(partitionPoints, minPts)

      // 2. 构建局部 R 树索引（使用自定义 RTree）
      var rtree = RTreeBuilder.create[Long]()
      for (p <- partitionPoints) {
        val geom = geometry.Point(p.x, p.y) // 使用你的 Point 实现
        rtree = rtree.add(p.id, geom)
      }

      // 3. 查询邻居（返回 Map[点ID -> 邻居ID数组]）
      val neighborsMap: Map[Long, Array[Long]] = partitionPoints.map { p =>
        val queryGeom = geometry.Point(p.x, p.y)
        val neighbors = rtree.search(queryGeom, eps)
          .filter(_.value != p.id)
          .map(_.value)
          .toArray
        (p.id, neighbors)
      }.toMap

      // 4. 局部 DBSCAN 聚类
      val visited = mutable.Set[Long]()
      val labels = mutable.Map[Long, Int]()
      var clusterId = 1

      for (p <- partitionPoints) {
        if (!visited.contains(p.id)) {
          visited.add(p.id)
          val neighbors = neighborsMap(p.id)
          if (neighbors.length < minPts) {
            labels(p.id) = 0 // 噪声
          } else {
            expandCluster(p.id, neighbors.toList, clusterId, labels, visited, neighborsMap, minPts)
            clusterId += 1
          }
        }
      }

      // 5. 输出带标签的结果
      partitionPoints.iterator.map { p =>
        val label = labels.getOrElse(p.id, -1)
        val pointType =
          if (label == -1) "noise"
          else if (neighborsMap(p.id).length >= minPts) "core"
          else "border"
        LabeledPoint(p.id, p.x, p.y, label, pointType)
      }
    }
  }


  def expandCluster(pId: Long, neighbors: List[Long], clusterId: Int,
                    labels: mutable.Map[Long, Int], visited: mutable.Set[Long],
                    neighborMap: Map[Long, Array[Long]], minPts: Int): Unit = {
    val queue = mutable.Queue(neighbors: _*)
    labels(pId) = clusterId

    while (queue.nonEmpty) {
      val qId = queue.dequeue()
      if (!visited.contains(qId)) {
        visited.add(qId)
        val qNeighbors = neighborMap.getOrElse(qId, Array.empty)
        if (qNeighbors.length >= minPts) {
          queue ++= qNeighbors.filterNot(visited.contains)
        }
      }
      if (!labels.contains(qId)) {
        labels(qId) = clusterId
      }
    }
  }

  def euclideanDistance(p1: Point, p2: Point): Double = {
    val dx = p1.x - p2.x
    val dy = p1.y - p2.y
    math.sqrt(dx * dx + dy * dy)
  }
}

object EpsCalculator {
  def computeEps(points: Array[Point], minPts: Int): Double = {
    val distances = points.map { p =>
      val dists = points.filter(_ != p).map(q => DRDBSCAN.euclideanDistance(p, q)).sorted
      if (dists.length >= minPts) Some(dists(minPts - 1)) else None
    }.flatten

    if (distances.isEmpty) {
      println("[WARN] 无有效距离用于计算 eps，返回默认值")
      return Double.PositiveInfinity
    }

    // 采用基于密度的 eps 计算
    val sortedDistances = distances.sorted
    sortedDistances(sortedDistances.length / 2) // 使用中位数来稳定eps
  }
}

object LabelFusion {

  // Merge labels across overlapping regions
  def mergeLabels(points: RDD[LabeledPoint]): RDD[LabeledPoint] = {
    // === Step 1: Group points by their id ===
    val groupedPoints: RDD[(Long, Iterable[LabeledPoint])] = points.groupBy(_.id)

    // === Step 2: Process each group (i.e., each data point and its labels across multiple regions) ===
    groupedPoints.mapValues { points =>
        // Extract points of different types (core, border, noise)
        val corePoints = points.filter(_.pointType == "core").toList
        val borderPoints = points.filter(_.pointType == "border").toList
        val noisePoints = points.filter(_.pointType == "noise").toList

        // Initialize the final label and type
        var finalLabel = -1
        var finalPointType = "noise" // Default to "noise"

        // Case 1: If there are multiple core points, merge them into one cluster
        if (corePoints.nonEmpty) {
          finalLabel = corePoints.head.label
          finalPointType = "core"
        }
        // Case 2: If there are no core points but border points, check if any border point can be merged
        else if (borderPoints.nonEmpty) {
          finalPointType = "border"

          // Check if any border point's neighbors can merge (look for core points in neighboring regions)
          val borderLabels = borderPoints.map(_.label).toSet
          if (borderLabels.size == 1) {
            finalLabel = borderLabels.head
          }
        }
        // Case 3: If no core or border points, it remains noise
        else if (noisePoints.nonEmpty) {
          finalPointType = "noise"
          finalLabel = -1
        }

        // Return the final labeled point
        LabeledPoint(points.head.id, points.head.x, points.head.y, finalLabel, finalPointType)
      }
      .values
  }
}
object FScoreCalculator {
  def calculatePairwiseFScore(trueLabels: RDD[LabeledPoint], predictedLabels: RDD[LabeledPoint]): Double = {
    // 将两个 RDD 以 ID 对齐后收集成 (id, trueLabel, predictedLabel)
    val trueMap = trueLabels.map(p => (p.id, p.label)).collectAsMap()
    val predMap = predictedLabels.map(p => (p.id, p.label)).collectAsMap()

    val ids = trueMap.keys.toArray

    var TP = 0
    var FP = 0
    var FN = 0
    var TN = 0

    for (id <- ids) {
      val trueLabel = trueMap(id)
      val predictedLabel = predMap.getOrElse(id, -1)

      // 根据真实标签和预测标签判断 TP, FP, FN, TN
      if (trueLabel == predictedLabel && trueLabel != 0) {
        TP += 1
      } else if (trueLabel != predictedLabel && trueLabel != 0 && predictedLabel != 0) {
        FP += 1
      } else if (trueLabel != predictedLabel && trueLabel != 0) {
        FN += 1
      } else if (trueLabel == predictedLabel && trueLabel == 0) {
        TN += 1
      }
    }

    val precision = if (TP + FP > 0) TP.toDouble / (TP + FP) else 0.0
    val recall    = if (TP + FN > 0) TP.toDouble / (TP + FN) else 0.0
    val fScore    = if (precision + recall > 0) 2 * precision * recall / (precision + recall) else 0.0

    println(s"TP=$TP, FP=$FP, FN=$FN, TN=$TN")
    println(f"Precision: $precision%.4f, Recall: $recall%.4f, F1 Score: $fScore%.4f")

    fScore
  }
}
