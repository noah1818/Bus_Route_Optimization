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

$$
G = (V, A)
$$

- \( V \): set of nodes (bus stops, intersections)  
- \( A \): set of directed edges (roads)  
- \( c^0 \): estimated edge costs = edge length / estimated travel speed  

The optimization problem seeks the path \( x \) from a start node \( a \) to an end node \( b \) minimizing:

$$
\min_{x \in X} \; \alpha \, c^T x - (1 - \alpha) \frac{1}{|S_0|} \sum_{x_s \in S_0} x^T x_s
$$

where  
- \( \alpha \) balances travel cost and similarity,  
- \( S_0 \) contains the \( k \)-most similar historical routes,  
- and uncertainty in edge costs is modeled using **robust optimization** following *Bertsimas & Sim (2003)*.

The robust counterpart introduces an uncertainty set with the parameter \( \Gamma \), limiting how many edge costs may deviate from their nominal values.

---

## ⚙️ Implementation
The algorithm was implemented in **Python 3** using:

| Library | Purpose |
|----------|----------|
| `networkx` | Graph creation & manipulation |
| `numpy` | Numerical operations |
| `pandas` | Data management |
| `heapq` | Priority queue for A* algorithm |

The **A\*** search algorithm was adapted to solve multiple nominal optimization problems corresponding to different uncertainty levels \( \Gamma \).

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

### 1. Heuristic 1
Euclidean distance between the current node and goal, divided by estimated travel speed:

$$
h(n) = \frac{\sqrt{(q_1 - p_1)^2 + (q_2 - p_2)^2}}{v}
$$

This heuristic is **admissible** and **non-negative**, providing a consistent lower bound on travel cost.

---

### 2. Heuristic 2
Combines Euclidean distance and historical route alignment via a weighting parameter \( \beta \):

$$
h(n) = \frac{\beta \cdot \text{distance}(n, g)}{v} - (1 - \beta) \frac{1}{|S_0|} \sum_{x_s \in S_0} s^T x_s
$$

This integrates both geometric distance and learned historical behavior, maintaining admissibility while improving contextual accuracy.

---

## 🧹 Data Preparation
The dataset contained:
- Historical bus routes  
- Traffic updates (March 28 – May 12, 2017, every 15 minutes)  
- Geographic coordinates of the network

**Cleaning steps:**
- Set missing speeds to 24 mph (default)
- Applied a minimum threshold of 3 mph
- Removed duplicate edges and averaged coordinates
- Filtered to include only instances with available historical routes

---

## 🧩 Runtime & Optimization
Graph deep-copy operations initially caused performance issues, consuming ~50% of total runtime.  
This was optimized by creating only **one deep copy per discrete time group**.

### ⚙️ Parameter Settings for Analysis
| Parameter | Symbol | Value | Description |
|------------|---------|--------|--------------|
| Alpha | \( \alpha \) | 0.5 | Weight between speed and similarity |
| Beta | \( \beta \) | 0.5 | Weight for historical influence in heuristic 2 |
| Gamma | \( \Gamma \) | \( N / 2 \) | Number of edges allowed to vary under uncertainty |
| k | \( k \) | 10 | Number of most similar scenarios used |
| Similarity Method | – | Euclidean Distance | Method used to determine historical similarity |

The **second heuristic** proved more robust but computationally heavier due to additional dot-product calculations at each A\* iteration.

---

## 📊 Results
- The algorithm consistently avoided high-traffic zones (e.g., downtown Chicago during rush hours).  
- During low-traffic periods, it sometimes produced more conservative routes due to robustness.  
- Different similarity metrics (Euclidean, Manhattan, RBF Kernel) yielded nearly identical route structures.

Visual comparisons showed strong alignment with historical routes and adaptive behavior during traffic peaks.

---

## 🧾 Conclusion
The developed algorithm successfully integrates:
- **Robust optimization** for uncertainty handling,  
- **Heuristic search** for computational efficiency, and  
- **Historical data** for practical route familiarity.

### Key Outcomes
- Improved route stability under variable traffic conditions  
- Adaptive performance for different time periods  
- Scalable approach for other cities and networks  

### Future Work
- Parameter tuning (\( \alpha, \beta, \Gamma, k \)) using automated training  
- Improved runtime performance for heuristic 2  
- Integration with real-time, large-scale datasets  

---

## 📚 References
- Bertsimas, D., & Sim, M. (2003). *The Price of Robustness*.  
- Russell, S. J., & Norvig, P. (2016). *Artificial Intelligence: A Modern Approach*.  
- Additional sources cited in the full paper.
