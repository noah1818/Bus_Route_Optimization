# 🚍 Robust Route Optimization for Chicago’s Bus Network
**Authors:** Julian Vignisson, Pascal Stefani, Noah Boss, Sascha Schrempf  
**Project Seminar – Final Paper**

---

## 📘 Overview
This project presents a robust optimization algorithm designed to find the most efficient and reliable bus routes in **Chicago’s complex street network**.  
By combining **graph theory**, **heuristic search techniques**, and **historical traffic data**, the algorithm balances three key goals:

- **Speed:** minimize travel time using estimated edge costs.  
- **Similarity:** maintain consistency with historical routes for familiarity.  
- **Robustness:** account for unpredictable events like congestion or closures.

The model was implemented in **Python** and evaluated on real-world traffic data from Chicago, aiming to enhance urban mobility by producing routes that are both efficient and reliable.

---

## 🧮 Mathematical Model
The routing problem is modeled as a **directed graph**:

\[
G = (V, A)
\]

- \( V \): set of nodes (bus stops, intersections)  
- \( A \): set of directed edges (roads)  
- \( c^0 \): estimated edge costs = edge length / estimated travel speed  

The optimization problem seeks the path \( x \) from a start node \( a \) to an end node \( b \) minimizing:

\[
\min_{x \in X} \alpha c^T x - (1 - \alpha) \frac{1}{|S_0|} \sum_{x_s \in S_0} x^T x_s
\]

where  
- \( \alpha \) balances travel cost and similarity,  
- \( S_0 \) contains the \( k \)-most similar historical routes,  
- and uncertainty in edge costs is modeled using **robust optimization** following Bertsimas and Sim (2003).

---

## ⚙️ Implementation
The algorithm was implemented in **Python 3** using:

| Library | Purpose |
|----------|----------|
| `networkx` | Graph creation & manipulation |
| `numpy` | Numerical operations |
| `pandas` | Data management |
| `heapq` | Priority queue for A* algorithm |

The **A\*** search algorithm was adapted to solve multiple nominal optimization problems, corresponding to different uncertainty levels \( \Gamma \).

---

## 🔍 Similarity Measures
To identify the most relevant historical routes, three similarity metrics were implemented:

1. **Euclidean Distance** – measures vector similarity between routes.  
2. **Manhattan Distance** – captures differences suited to grid-like networks.  
3. **RBF Kernel Method** – identifies nonlinear relationships in historical route data.

These metrics determine which \( k \) scenarios from historical data are included in the optimization.

---

## 🧠 Heuristic Approaches
Two heuristic functions were used within A\*:

1. **Heuristic 1:**  
   Euclidean distance between current node and goal, divided by estimated travel speed.  
   Admissible and non-negative.

2. **Heuristic 2:**  
   Combines Euclidean distance and historical route alignment via a weighting parameter \( \beta \):  
   \[
   h(n) = \frac{\beta \cdot \text{distance}(n, g)}{v} - (1 - \beta) \frac{1}{|S_0|} \sum_{x_s \in S_0} s^T x_s
   \]
   This integrates both geometry and learned historical behavior.

---

## 🧹 Data Preparation
The dataset contained:
- Historical bus routes  
- Traffic updates (March 28 – May 12, 2017, every 15 minutes)  
- Geographic coordinates of the network

**Cleaning steps:**
- Set missing speeds to 24 mph (default)
- Enforced a 3 mph minimum threshold
- Removed duplicate edges and averaged coordinates
- Filtered data to instances with available historical routes

---

## 🧩 Runtime & Optimization
Graph deep-copy operations initially caused performance issues, consuming ~50% of runtime.  
This was optimized by creating only one deep copy per **discrete time group**.

Parameter settings for analysis:
