# -----------------------------------------------------------------------------
# Project:     Bus Route Optimization
# File:        model.py
# Company:     Stratton Oakmont
# Date:        September 13, 2024
# Version:     2.2.3
# Description: This script contains the main algorithm to optimize bus routes
#              in the city of Chicago, using graph theory and heuristic search.
# -----------------------------------------------------------------------------
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import networkx as nx
import copy
import heapq


class Model(object):
    def __init__(self, data_all: pd.DataFrame, route_all: pd.DataFrame, graph_all: pd.DataFrame, no_rh_night_df: pd.DataFrame, no_rh_day_df: pd.DataFrame, morning_rh_df: pd.DataFrame, evening_rh_df: pd.DataFrame, weekend_df: pd.DataFrame, time_columns: int, n_similar: int, start: int, end: int) -> None:
        """
        Initializes the Model class

        Parameters
        ----------
            instances_data : The data_all data frame, containing the scenarios.
            routes_data : The route_all data frame, containing the routes.
            graph_data : The graph data frame, containing the coordinates of the nodes.
            no_rh_night_df : The DataFrame masked on the scenarios not in the nightly rush hour.
            no_rh_day_df : The DataFrame masked on the scenarios not in the daily rush hour.
            morning_rh_df : The DataFrame masked on scenarios in the morning rush hour.
            evening_rh_df : The DataFrame masked on scenarios in the evening rush hour.
            weekend_df : The DataFrame masked on scenarios on the weekend.
            time_columns : The columns of the data_all DataFrame containinf time values, like year, day or month.
            heuristic_a_star : The heuristic used in the A* algorithm. (See our paper, for the exmplanation of heuristic 1 or 2).
            start : The start node.
            end : The end node.

        Returns
        -------
        """

        self.instances_data = data_all
        self.routes_data = route_all
        self.graph_data = graph_all
        self.no_rh_night_df = no_rh_night_df
        self.no_rh_day_df = no_rh_day_df
        self.morning_rh_df = morning_rh_df
        self.evening_rh_df = evening_rh_df
        self.weekend_df = weekend_df
        self.time_columns = time_columns
        self.n_similar = n_similar
        self.start = start
        self.end = end

        # ignore, these are just some params being initalized later
        self.c = None
        self.alpha = None
        self.beta = None
        self.S = None
        self.paths = None

        # scenario method and heuristic are None but initialized with the start_algo method
        self.heuristic = None
        self.scenario_method = None

    def cost_weighted_data_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates the costs assigned to each value in the masked data_all frame.

        Parameters
        ----------
            df : The DataFrame that was alreads masked with seasonal similaritys.

        Returns
        -------
            pd.DataFrame : The DataFrame with cost assigned values.
        """

        M = df.shape[0]
        N = df.shape[1] - self.time_columns
        edge_lengths = np.array(
            [self.euclidian_distance_coordinates(j + 1) for j in range(N)])
        travel_speeds = df.iloc[:, :N].values
        result = np.divide(edge_lengths[np.newaxis, :], travel_speeds, out=np.zeros_like(
            travel_speeds), where=travel_speeds != 0)
        df.iloc[:, :N] = result
        return df

    def euclidean_distance(self, u: np.array, v: np.array) -> float:
        """
        Calculates the euclidian distance for two vectors.

        Parameters
        ----------
            u : First vector.
            v : Second vector.

        Returns
        -------
            float : The euclidan distance from the first to the second vector.
        """

        return ((u - v) ** 2).sum(axis=1) ** 0.5

    def rbf_kernel(self, df: pd.DataFrame, scenario: int, k: int, gamma=0.0001) -> pd.DataFrame:
        """
        This method implements the rbf kernel.

        Parameters
        ----------
            df: Data Frame of our pre masked scenarios
            scenario : The index of the scenario in data_all.
            k : Number of most similar scenarios we want
            gamma : Spread of the kernel => influence on decision range. 
                Example: 
                    1. High gamma.
                    2. Low gamma.
                Effect: 
                    1. Even points that are relatively far from each other will have a non-negligible similarity score.
                    2. Only points that are very close to each other have a significant similarity score.
        Returns
        -------
            np.array : DataFrame of the most similar instances based on the masked DataFrame and the rbf kernel.
        """

        scaler = StandardScaler()
        df_normalized = scaler.fit_transform(
            df.iloc[:, :-self.time_columns].fillna(0))
        rbf_kernel_matrix = rbf_kernel(df_normalized, gamma=gamma)

        df_ = pd.DataFrame(index=df.index, columns=df.index)
        df_.loc[:] = rbf_kernel_matrix
        similarity_scores = df_[scenario]
        most_similar_indices = np.argsort(similarity_scores)[::-1][:k]
        return most_similar_indices

    def manhattan_distance(self, u: np.array, v: np.array) -> float:
        """
        Calculate the Manhattan distance between two vectors.

        Parameters
        ----------
            u : First vector.
            v : Second vector.

        Returns
        -------
            float: Manhattan distance between the first and second vector.
        """

        return np.sum(np.abs(u - v))

    def k_most_similar_helper(self, c0: np.array, scenario: int, df: pd.DataFrame, k: int) -> pd.Series:
        """
        Calculates k most similar scenarios based of the scenario method, that is pre determined in the innit method of the Model class.

        Parameters
        ----------
            c0 : Reference vector for which the most similar scenarios are searched.
            scenario : The index of the scenario in data_all.
            df : DataFrame containing the scenarios as rows. Each row represents a scenario 
                with numeric values that will be compared against the reference vector.
            k : Number of similar scenarios to return.

        Returns
        -------
            pd.Series : The k most similar scenarios compared to c0.
        """

        df_masked = None

        if self.scenario_method == 'euclidean_distance':
            distances = self.euclidean_distance(
                df.iloc[:, :-self.time_columns], c0)
            best_k_indices = distances.nsmallest(k).index
            df_masked = df.loc[best_k_indices]

        elif self.scenario_method == 'rbf_kernel':
            best_k_indices = self.rbf_kernel(df, scenario, k)
            df_masked = df.iloc[best_k_indices]

        elif self.scenario_method == 'manhattan_distance':
            distances = df.iloc[:, :-self.time_columns].apply(
                lambda row: self.manhattan_distance(row.values, c0), axis=1)
            best_k_indices = distances.nsmallest(k).index
            df_masked = df.loc[best_k_indices]

        else:
            raise ValueError(
                f"we didnt implement this yet, maybe you meant one of the following scenario methods: [{'euclidean_distance', 'rbf_kernel', 'manhattan_distance'}]")

        return df_masked

    def get_df_seasonal_similarities(self, weekday: int, hour: int) -> pd.DataFrame:
        """
        We return the DataFrame belonging to the seasonal similarity.

        Parameters
        ----------
            weekday : Weekday of start scenario.
            hour : Hour of start scenario.

        Returns
        -------
            pd.DataFrame : The k most similar scenarios compared to c0 as a Data Frame, based in weekday and hour.
        """

        df = None
        if weekday in [5, 6]:
            df = self.weekend_df
        else:
            if hour in [8, 9]:
                df = self.morning_rh_df
            elif hour in [16, 17]:
                df = self.evening_rh_df
            elif hour in [10, 15]:
                df = self.no_rh_day_df
            else:
                df = self.no_rh_night_df
        return df

    def get_k_most_similar(self, df_seasonal_similarities: pd.DataFrame, k: int, c0: np.array, scenario: int) -> pd.DataFrame:
        """
        We filter the DataFrame on seasonal similarities and the scenario method.

        Parameters
        ----------
            df_seasonal_similarities : The DataFrame containing the seasonal similarities.
            k :  Number of similar scenarios to return.
            c0 : Reference vector for which the most similar scenarios are searched.
            scenario : The index of the scenario in data_all.

        Returns
        -------
            pd.DataFrame : The k most similar scenarios compared to c0 as a Data Frame.
        """

        df = self.cost_weighted_data_all(df_seasonal_similarities.copy())
        df_seasonal_similarities_weight = df
        df = df[df['timestamp'].isin(self.routes_data['timestamp'])]
        if len(df) > k:
            print(
                f"There are more then k (k = {k-1}) to filter with the specified scenario method")
            best_k = self.k_most_similar_helper(
                c0, scenario, df, k)
            df = best_k

        elif len(df) < k:
            print(
                f"There are not more then k (k = {k-1}) to filter with the specified scenario method, we just use the DataFrame filtered on the time")

        else:
            raise ValueError("There are not enough data points")

        return df, df_seasonal_similarities_weight

    def euclidian_distance_coordinates(self, n: int) -> float:
        """
        Calcualtes the euclidian distance for the start and end node of an edge.

        Parameters
        ----------
            n : The edge name.

        Returns
        -------
            float : The euclidian distance from the start to the end node of an edge.
        """

        try:
            index = np.where(self.graph_data['edge_name'] == n)[0][0]
        except:
            return 0
        a_x = self.graph_data['x_coordinate_A'][index]
        a_y = self.graph_data['y_coordinate_A'][index]
        b_x = self.graph_data['x_coordinate_B'][index]
        b_y = self.graph_data['y_coordinate_B'][index]
        return np.sqrt((a_x - b_x)**2 + (a_y - b_y)**2)

    def get_t0(self, scenario: int) -> Tuple[int, int]:
        """
        Calcualtes the weekday and hour of the scenario.

        Parameters
        ----------
            scenario : The index of the scenario in data_all.

        Returns
        -------
            Tuple(int,int) : The weekday and hour as a tuple.
        """

        return self.instances_data['weekday'][scenario], self.instances_data['hours'][scenario]

    def get_c0(self, scenario: int) -> pd.Series:
        """
        Calcualtes the vector c0, where c0_i = edge_length / estimate_travel_speed.

        Parameters
        ----------
        scenario : The index of the scenario in data_all.

        Returns
        -------
        pd.Series : c0 as a pandas.Series object.
        """
        c0 = np.zeros(self.instances_data.shape[1]-self.time_columns)
        for i in range(0, c0.shape[0]):
            edge_length = self.euclidian_distance_coordinates(i+1)
            estimate_travel_speed = self.instances_data.iloc[scenario].values[i]
            c0[i] = edge_length / estimate_travel_speed
        return c0

    def calc_delta(self, c: pd.Series) -> pd.Series:
        """
        Calculates the uncertainty interval.

        Parameters
        ----------
            c : Cost vector c.

        Returns
        -------
            pd.Series : PandasSeries of the updatet weights of an edge.
        """

        N = c.shape[0]
        delta = np.zeros(N)

        for j in range(N):
            delta[j] = np.max(c.values) - c.values[j]

        return delta

    def get_alpha(self) -> float:
        """
        Calcualtes alpha, is set to 0.5, see paper for explanation.

        Parameters
        ----------

        Returns
        -------
            float : The value of alpha.
        """

        return .5

    def get_beta(self) -> float:
        """
        Calcualtes beta, is set to 0.5, see paper for explanation.

        Parameters
        ----------

        Returns
        -------
            float : The value of beta.
        """

        return .5

    def get_gamma(self, N) -> float:
        """
        Calcualtes gamma, is set to N/2, see paper for explanation.

        Parameters
        ----------
        N : The length of df_similar.

        Returns
        -------
            float : The value of gamma.
        """

        return int(N / 2)

    def get_gamma_graphs(self, G: nx.DiGraph) -> float:
        """
        Calcualtes the gamma graphs by changing gamma of the rows of our DataFrame.

        Parameters
        ----------
            G : A directional graph.

        Returns
        -------
            float : The value of gamma.
        """

        N = len(G.edges())
        gamma = self.get_gamma(N)

        eges_with_data = G.edges()
        edges_sorted = sorted(eges_with_data)

        uncertainty_graphs = []

        # first of all calc_weights
        updates = {}
        updates['gamma_c'] = []
        updates['index'] = []

        edges = G.edges()
        for edge in edges:
            if not (G.edges[edge]['weights'] == G.edges[edge]['weights'].values[0]).all():
                G.edges[edge]['weights'] = (
                    G.edges[edge]['weights']).sort_values(ascending=False)

        for i in range(N):
            delta_c = self.calc_delta(G.edges[edges_sorted[i]]['weights'])
            updates['gamma_c'].append(delta_c)
            updates['index'].append(edges_sorted[i])

        array_sum_index_tuples = [(np.sum(arr), arr, idx) for arr, idx in zip(
            updates['gamma_c'], updates['index'])]
        sorted_tuples = sorted(array_sum_index_tuples,
                               key=lambda x: x[0], reverse=True)
        sorted_arrays = [tup[1] for tup in sorted_tuples]
        sorted_indices = [tup[2] for tup in sorted_tuples]
        updates['gamma_c'] = sorted_arrays
        updates['index'] = sorted_indices
        copys = []

        for i in range(N + 1):
            if i > 0 and i < N-1 and (updates['gamma_c'][i] == updates['gamma_c'][i+1]).all():
                uncertainty_graphs.append(uncertainty_graphs[-1])
                copys.append(1)

            else:
                H = copy.deepcopy(G)
                for l in range(0, i):
                    edge = updates['index'][l]
                    delta = updates['gamma_c'][l]
                    delta[gamma:] = 0
                    H.edges[edge]['weights'] += delta

                copys.append(0)
                uncertainty_graphs.append(H)
        return uncertainty_graphs, copys

    def heuristic_1(self, G: nx.DiGraph, current: float, neighbor: float, target: float, scenario: int) -> float:
        """
        This is the first heuristic for the A* search algorithm. As proposed in our paper.

        Parameters
        ----------
            G : A directional graph.
            current: The current node from wich we calcaulte the estimated travel speed to the neighbor.
            neighbor : The neighbor node from wich we calcaulte the euclidian distance to the target node.
            target : The target node, where we calculated the distance to, from the first neighbor node.
            scenario : The index of the scenario in data_all.

        Returns
        -------
            float : The value of the first custom heurisitc at state n.
        """

        x1, y1 = G.nodes[target]['pos']
        x2, y2 = G.nodes[neighbor]['pos']
        v = None

        v = G.edges[current, neighbor]['weights'][scenario]

        if v == None:
            v = 1
        return np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / v

    def heuristic_2(self, G: nx.DiGraph, current: int, neighbor: int, target: int, scenario: int, beta: float, S0: np.ndarray, xs: np.array) -> float:
        """
        This is the second heuristic for the A* search algorithm. As proposed in our paper.

        Parameters
        ----------
            G : A directional graph.
            current: The current node from wich we calcaulte the estimated travel speed to the neighbor.
            neighbor : The neighbor node from wich we calcaulte the euclidian distance to the target node.
            target : The target node, where we calculated the distance to, from the first neighbor node.
            scenario : The index of the scenario in data_all.
            beta : A weighting factor.
            S0 : The historic routes of the k most similar scenarios.
            xs : The path we took from the start node to the current node.

        Returns
        -------
            float : The value of the second custom heurisitc at state n.
        """

        h_n = None
        v = None
        if neighbor == target:
            h_n = 0

        else:
            x1, y1 = G.nodes[target]['pos']
            x2, y2 = G.nodes[neighbor]['pos']

            v = G.edges[current, neighbor]['weights'][scenario]

            if v == None:
                v = 1
            eulcidian_distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2) / v
            h_n = beta * (eulcidian_distance / v) - \
                (1 - beta) * (1 / S0.shape[0]) * np.sum(np.dot(S0, xs))
        return h_n

    def reconstruct_path(self, came_from: dict, current: int) -> list:
        """
        Reconstruct the path from the start node to the current node using the came_from dictionary.

        Parameters
        ----------
            came_from : A dictionary where each key is a node and the value is the node from which the current node was reached.
            current : The goal node or the end node from which the path reconstruction begins.

        Returns
        -------
            list : A list of nodes representing the path from the start node to the current node.
        """

        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]

    def a_star(self, G: nx.DiGraph, source: int, target: int, scenario: int, heuristic: int, beta: float, S0: np.ndarray, df_similar: pd.DataFrame) -> Optional[list]:
        """
        Implementation of the A* algorithm with a custom heuristic (1 or 2), as seen in our paper.

        Parameters
        ----------
            G : A directional graph.
            source: The source node.
            target : The target node.
            scenario : The index of the scenario in data_all.
            heuristic : One of our custom heuristics, as seen in our paper (binary encoded => 0 for the first one, 1 for the second one).
            S0 : The historic routes of the k most similar scenarios.
            df_similar : The DataFrame containing the n most similar scenarios.

        Returns
        -------
            list : The path from source to target.
        """

        open_set = []
        counter = 0  # Counter to ensure stable heap insertion
        heapq.heappush(open_set, (0, counter, source))
        came_from = {}
        g_score = {node: float('inf') for node in G.nodes}
        g_score[source] = 0
        f_score = {node: float('inf') for node in G.nodes}
        f_score[source] = 0
        heuristic_vals = []

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == target:
                return self.reconstruct_path(came_from, current)

            for neighbor in G.neighbors(current):
                tentative_g_score = g_score[current] + \
                    G[current][neighbor]['weights'][scenario]
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score

                    # calc own heuristic
                    route_now = self.reconstruct_path(came_from, current)
                    route_now = self.get_paths_0_1(route_now, df_similar)
                    # heuristic 1
                    if heuristic == 1:
                        heuristic_val = self.heuristic_1(
                            G, current, neighbor, target, scenario)
                    else:
                        # heuristic 2
                        heuristic_val = self.heuristic_2(
                            G, current, neighbor, target, scenario, beta, S0, route_now)

                    f_score[neighbor] = tentative_g_score + \
                        heuristic_val

                    heuristic_vals.append(heuristic_val)

                    if neighbor not in [item[2] for item in open_set]:
                        counter += 1
                        heapq.heappush(
                            open_set, (f_score[neighbor], counter, neighbor))

        return None

    def create_graph(self, df: pd.DataFrame) -> nx.DiGraph:
        """
        Creates a graph, based on the forecasted weights of our seasonal similarities DataFrame.

        Parameters
        ----------
            df : The DataFrame containing the seasonal similarities.

        Returns
        -------
            nx.DiGraph : A directional graph, based on the given weights of the DataFrame.
        """

        G = nx.DiGraph()
        for _, row in self.graph_data.iterrows():
            G.add_node(row['node_A'], pos=(
                row['x_coordinate_A'], row['y_coordinate_A']))
            G.add_node(row['node_B'], pos=(
                row['x_coordinate_B'], row['y_coordinate_B']))
            G.add_edge(row['node_A'], row['node_B'], name=row['edge_name'],
                       weights=df.get(row['edge_name']))

        return G

    def get_historic_routes(self, df_similar: pd.DataFrame) -> np.array:
        """
        Creates the historic routes, based on the DataFrame of the seasonal similarities and the scenario method.

        Parameters
        ----------
            df_similar : The DataFrame containing the n most similar scenarios.

        Returns
        -------
            np.array : An array, with values in {0,1} where it is 1 if the route is used and 0 otherwise.
        """

        timestamps = df_similar.timestamp
        routes = self.routes_data.loc[self.routes_data['timestamp'].isin(
            timestamps)]

        N = df_similar.shape[1] - self.time_columns
        M = routes.shape[0]

        routes_0_1 = np.zeros(shape=(M, N))

        for i in range(M):
            for j in range(N):
                if j in routes.values[i][:-self.time_columns]:
                    routes_0_1[i, j] = 1
        return routes_0_1

    def get_paths_0_1(self, path: list, df_similar: pd.DataFrame) -> np.array:
        """
        Creates an array containing 0 or 1, 1 if the edge in df_similar is used in the path, 0 else.

        Parameters
        ----------
            path : The path we want to map to values in {0,1}.
            df_similar : The DataFrame containing the n most similar scenarios.

        Returns
        -------
            np.array : An array, with each element in {0,1} where it is 1 if the route is used and 0 otherwise.
        """

        N = df_similar.shape[1] - self.time_columns
        path_normed = np.zeros(N)
        for i in range(0, N):
            if i in path:
                path_normed[i] = 1
        return path_normed

    def get_c(self, G: nx.DiGraph, scenario: int, df_similar: pd.DataFrame) -> np.array:
        """
        Creates the cost vector with the weighs being in the graph G and the given scenario, df_similar is only used for getting the shape right here.

        Parameters
        ----------
            G : A directional graph.
            scenario : The index of the scenario in data_all.
            df_similar : The DataFrame containing the n most similar scenarios.

        Returns
        -------
            np.array : An array, with the weights of an instance as an entry.
        """

        edge_names = np.asarray([data['name']
                                for u, v, data in G.edges(data=True)])
        edge_weight = np.asarray([data
                                  for u, v, data in G.edges(data=True)])

        N = df_similar.shape[1] - self.time_columns
        c = np.zeros(N)

        for i in range(1, N+1):
            if i in edge_names:
                k = np.where(edge_names == i)[0]
                if k.size > 0:
                    c[i-1] = edge_weight[k][0]['weights'][scenario]
        return c

    def get_paths_for_uncertainty_graphs(self, A: int, B: int, uncertainty_graphs: list, scenario: int, df_similar: pd.DataFrame, copys: list, S0: np.array, heuristic: int, beta: float) -> Tuple[list, list, list]:
        """
        Gets the different paths for the uncertainty graphs.

        Parameters
        ----------
            A : The start node, used in a start.
            B : The goal node, used in a start.
            uncertainty_graphs : A list of the differenct nx.DiGraph objects, as the uncertainty graphs.
            scenario : The index of the scenario in data_all.
            df_similar : The DataFrame containing the n most similar scenarios.
            copys : Containing either 0 or 1, depending on if we changed the weights or not.
            S0 : The historic routes of the k most similar scenarios.
            heuristic : One of our custom heuristics, as seen in our paper (binary encoded => 0 for the first one, 1 for the second one).
            beta : A weighting factor.

        Returns
        -------
            Tuple[list, list, list] : The different lists of the normal path, the path ony with values in {0,1} and the path with its costs.
        """

        paths = []
        paths_0_1 = []
        C = []
        for i, uncertainty_graph in enumerate(uncertainty_graphs):
            if i > 0 and copys[i] == 1:
                paths.append(paths[-1])
                paths_0_1.append(paths_0_1[-1])
                C.append(C[-1])
            else:
                path = self.a_star(uncertainty_graph, A, B,
                                   scenario, heuristic, beta, S0, df_similar)
                paths.append(path)
                path_0_1 = self.get_paths_0_1(
                    path, df_similar)
                paths_0_1.append(path_0_1)
                c = self.get_c(uncertainty_graph, scenario, df_similar)
                C.append(c)

        return paths, paths_0_1, C

    def objective(self, x: np.array, c: np.array) -> np.array:
        """
        This is the objective function as seen in the paper.

        Parameters
        ----------
            x : The parameter we want to minimize over, with x element X := {paths from A to B in G, solved with A*}
            c : The cost vector, with the gamma weights as entrys.

        Returns
        -------
            np.array : The minimum x.
        """

        term1 = self.alpha * np.dot(c, x)
        term2 = -(1 - self.alpha) * np.mean(
            [np.dot(xs, x) for xs in self.S])
        return term1 + term2

    def find_min(self, x0: np.array) -> Tuple[list, list, list]:
        """
        Finds the minimum given an initale guess.

        Parameters
        ----------
            x0 : The initial guess

        Returns
        -------
            Tuple[list, list, list] : A tuple consisting of three lists, being the path with 0 and 1, the minimum weights used and the actual minimum path.
        """

        global_minimum = float('inf')
        min_x = x0
        min_c = self.c[0]
        min_path = self.paths[0]
        for x, path, c in zip(self.paths_0_1, self.paths, self.c):
            local_minimum = self.objective(x, c)
            if local_minimum < global_minimum:
                global_minimum = local_minimum
                # define new minimums
                min_x = x
                min_c = c
                min_path = path
        return min_x, min_c, min_path

    def get_original_weights(self, path: list, G: nx.DiGraph, scenario: int) -> list:
        """
        Finds the original weights of a graph with estimated travel speed.

        Parameters
        ----------
            path : A list of nodes, representing the path.
            scenario : The index of the scenario in data_all.

        Returns
        -------
            Tuple[list, list, list] : A tuple consisting of three lists, being the path with 0 and 1, the minimum weights used and the actual minimum path.
        """

        weights = []
        for i in range(len(path)-1):
            # this try catch is just incease of the edge not existing in the graph
            try:
                weight = G.edges[path[i], path[i+1]]['weights'][scenario]
            except:
                weight = weights[-1]
            weights.append(weight)

        return weights

    def get_S0(self, k_most_similar: pd.DataFrame, scenario: int) -> np.ndarray:
        """
        Calculates the historic routes of the k most similar scenarios.

        Parameters
        ----------
            k_most_similar : The k most similar scenarios as a DataFrame.
            scenario : The index of the scenario in data_all.

        Returns
        -------
            np.ndarray : The historic routes, with values in {0,1}.
        """

        if scenario in k_most_similar.index:
            k_most_similar = k_most_similar.drop(scenario)
        else:
            k_most_similar = k_most_similar[:-1]
        S0 = self.get_historic_routes(k_most_similar)
        return S0

    def get_sum_path(self, path: list, scenario: int) -> float:
        """
        Calculates the sum of mph from the edges unsed in the path.

        Parameters
        ----------
            path : The optimal path calculated by our algorithm.
            scenario : The index of the scenario in data_all.

        Returns
        -------
            float : The sum of mph from the edges unsed in the path.
        """

        sum_ = 0
        for i in range(len(path)-1):
            index = np.where((self.graph_data.node_A == path[i]) & (
                self.graph_data.node_B == path[i+1]))[0][0]
            sum_ += self.instances_data[index][scenario]

        return sum_

    def start_algo(self, scenario: int, scenario_method: str, heuristic: int) -> list:
        """
        This method starts our algorithm.

        Parameters
        ----------
            scenario : The index of the scenario in data_all.
            scenario_method : The scenario method (rbf_kernel, euclidean_distance, manhattan_distance).
            heuristic : The heuristic used in A*, binary decoded as 1 or 2.

        Returns
        -------
            np.array : The actual path of the shortest route.
        """

        if type(scenario) != int:
            raise ValueError("please choose an integer as a scenario")

        if scenario >= self.instances_data.shape[0]:
            raise ValueError(
                "please choose a scenario within the DataFrame data_all.csv")

        if type(scenario_method) != str:
            raise ValueError(
                "please choose a scenario_method within [{'euclidean_distance', 'rbf_kernel', 'manhattan_distance'}] and use strings")

        if scenario_method not in ["euclidean_distance", "rbf_kernel", "manhattan_distance"]:
            raise ValueError(
                "please choose a scenario_method within [{'euclidean_distance', 'rbf_kernel', 'manhattan_distance'}]")

        if type(heuristic) != int:
            raise ValueError(
                "please choose an integer as a heuristic, the two different methods from our paper are binary encoded as 1 or 2")

        if heuristic not in [1, 2]:
            raise ValueError(
                "please choose the integer 1 or 2 as a heuristic, see our paper for the difference")

        self.scenario_method = scenario_method
        self.heuristic = heuristic

        c0 = self.get_c0(scenario)
        t0_weekday, t0_hour = self.get_t0(scenario)

        df_seasonal_similarities = self.get_df_seasonal_similarities(
            t0_weekday, t0_hour)
        k_most_similar, df_seasonal_similarities = self.get_k_most_similar(df_seasonal_similarities[:self.routes_data.index[-1]],
                                                                           self.n_similar+1, c0, scenario)

        S0 = self.get_S0(k_most_similar, scenario)

        G = self.create_graph(df_seasonal_similarities)

        gamma_graphs, copys = self.get_gamma_graphs(G)

        self.beta = self.get_beta()
        self.alpha = self.get_alpha()

        self.paths, self.paths_0_1, c = self.get_paths_for_uncertainty_graphs(
            self.start, self.end, gamma_graphs, scenario, df_seasonal_similarities, copys, S0, self.heuristic, self.beta)

        self.c = c
        self.S = S0

        x0 = np.zeros(self.c[0].shape[0])
        min_x, min_c, min_path = self.find_min(x0)

        # we return the sum of the values of the edges used in the path, e.g. the sum of mph
        # if one wants to return the actual cost, like seen in the plots in our paper, just return original_costs and outcomment the calculation of the original_costs
        # original_costs = edge_length / estimated_travel_speed
        sum_path = self.get_sum_path(min_path, scenario)

        # original_costs = self.get_original_weights(min_path, G, scenario)
        return min_path, sum_path  # , original_costs
