import numpy as np
import pandas as pd
import networkx as nx
from tqdm import tqdm
import geopandas as gpd
from shapely.geometry import Point

pd.set_option('future.no_silent_downcasting', True)

class Model:
    def __init__(self, grid, G, pop_col='m', crs_out=4326):
        self.grid = grid.copy()
        if 'node_coordinate' in self.grid.columns:
            self.grid = self.grid.rename(columns={'node_coordinate': 'geometry'})
        if not isinstance(self.grid.iloc[0]['geometry'], Point):
            self.grid['geometry'] = self.grid['geometry'].apply(lambda x: Point(x) if not isinstance(x, Point) else x)
        self.grid = gpd.GeoDataFrame(self.grid, geometry='geometry', crs=crs_out)
        self.grid['node'] = self.grid['node_id']
        self.pop_col = pop_col
        self.G = G

    def _compute_distance_matrix(self):
        nodes = self.grid['node'].tolist()
        dist_matrix = pd.DataFrame(index=nodes, columns=nodes)
        for i in tqdm(nodes, desc="Distance matrix", unit="node"):
            lengths = nx.single_source_dijkstra_path_length(self.G, i, weight='length')
            for j in nodes:
                dist_matrix.at[i, j] = lengths.get(j, np.inf)
        return dist_matrix.astype(float)

    def compute_gravity_od(self, epsilon=1e-6):
        nodes = self.grid['node'].tolist()
        masses = self.grid.set_index('node')[self.pop_col].to_dict()
        dist_matrix = self._compute_distance_matrix()
        od_matrix = pd.DataFrame(index=nodes, columns=nodes)
        for i in nodes:
            for j in nodes:
                if i == j:
                    od_matrix.at[i, j] = 0
                else:
                    d = dist_matrix.at[i, j]
                    od_matrix.at[i, j] = (masses[i] * masses[j]) / (d ** 2 + epsilon)
        return od_matrix.fillna(0).infer_objects(copy=False).astype(float)


    def compute_radiation_od(self):
        nodes = self.grid['node'].tolist()
        masses = self.grid.set_index('node')[self.pop_col].to_dict()
        dist_matrix = self._compute_distance_matrix()
        od_matrix = pd.DataFrame(index=nodes, columns=nodes)
        for i in nodes:
            for j in nodes:
                if i == j:
                    od_matrix.at[i, j] = 0
                    continue
                dij = dist_matrix.at[i, j]
                if np.isinf(dij):
                    od_matrix.at[i, j] = 0
                    continue
                s_ij = 0
                for k in nodes:
                    if k != i and k != j and dist_matrix.at[i, k] < dij:
                        s_ij += masses[k]
                m_i, m_j = masses[i], masses[j]
                od_matrix.at[i, j] = m_i * m_j / ((m_i + s_ij) * (m_i + m_j + s_ij))
        return od_matrix.fillna(0).infer_objects(copy=False).astype(float)
        

    def compute_logit_od(self, beta=-0.01):
        nodes = self.grid['node'].tolist()
        masses = self.grid.set_index('node')[self.pop_col].to_dict()
        dist_matrix = self._compute_distance_matrix()
        od_matrix = pd.DataFrame(index=nodes, columns=nodes)
        for i in nodes:
            utilities = []
            for j in nodes:
                if i == j or np.isinf(dist_matrix.at[i, j]):
                    utilities.append(-np.inf)
                else:
                    utilities.append(beta * dist_matrix.at[i, j])
            exp_utilities = np.exp(utilities - np.max(utilities))
            probs = exp_utilities / np.sum(exp_utilities)
            for idx, j in enumerate(nodes):
                od_matrix.at[i, j] = masses[i] * probs[idx] if i != j else 0
        return od_matrix.fillna(0).infer_objects(copy=False).astype(float)
        

class CostModel:
    def __init__(self, grid, G, pop_col='m', crs_out=4326):
        self.grid = grid.copy()
        if 'node_coordinate' in self.grid.columns:
            self.grid = self.grid.rename(columns={'node_coordinate': 'geometry'})
        if not isinstance(self.grid.iloc[0]['geometry'], Point):
            self.grid['geometry'] = self.grid['geometry'].apply(lambda x: Point(x) if not isinstance(x, Point) else x)
        self.grid = gpd.GeoDataFrame(self.grid, geometry='geometry', crs=crs_out)
        self.grid['node'] = self.grid['node_id']
        self.pop_col = pop_col
        self.G = G

    def _compute_distance_matrix(self):
        nodes = self.grid['node'].tolist()
        dist_matrix = pd.DataFrame(index=nodes, columns=nodes)
        for i in tqdm(nodes, desc="Distance matrix", unit="node"):
            lengths = nx.single_source_dijkstra_path_length(self.G, i, weight='length')
            for j in nodes:
                dist_matrix.at[i, j] = lengths.get(j, np.inf)
        return dist_matrix.astype(float)

    def _normalize_and_transform(self, od_matrix):
        """
        Normalize each row to get probabilities and then compute costs as -log(probability).
        Diagonal entries (i==j) are set to 0.
        """
        # Normalize each row so that it sums to 1
        row_sums = od_matrix.sum(axis=1)
        prob_matrix = od_matrix.div(row_sums, axis=0).fillna(0)
        # Avoid log(0) by replacing zeros with a small number (except on the diagonal)
        small_val = 1e-12
        prob_matrix_safe = prob_matrix.replace(0, small_val)
        cost_matrix = -np.log(prob_matrix_safe)
        # Set diagonal back to 0 (i.e. cost from node to itself)
        for node in cost_matrix.index:
            cost_matrix.at[node, node] = 0
        return cost_matrix.astype(float)

    def compute_gravity_cost(self, epsilon=1e-6):
        nodes = self.grid['node'].tolist()
        masses = self.grid.set_index('node')[self.pop_col].to_dict()
        dist_matrix = self._compute_distance_matrix()
        od_matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=float)
        for i in nodes:
            for j in nodes:
                if i == j:
                    od_matrix.at[i, j] = 0
                else:
                    d = dist_matrix.at[i, j]
                    od_matrix.at[i, j] = (masses[i] * masses[j]) / (d ** 2 + epsilon)
        od_matrix = od_matrix.fillna(0)
        # Normalize and convert probabilities to cost
        cost_matrix = self._normalize_and_transform(od_matrix)
        return cost_matrix

    def compute_radiation_cost(self):
        nodes = self.grid['node'].tolist()
        masses = self.grid.set_index('node')[self.pop_col].to_dict()
        dist_matrix = self._compute_distance_matrix()
        od_matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=float)
        for i in nodes:
            for j in nodes:
                if i == j:
                    od_matrix.at[i, j] = 0
                    continue
                dij = dist_matrix.at[i, j]
                if np.isinf(dij):
                    od_matrix.at[i, j] = 0
                    continue
                s_ij = 0
                for k in nodes:
                    if k != i and k != j and dist_matrix.at[i, k] < dij:
                        s_ij += masses[k]
                m_i, m_j = masses[i], masses[j]
                od_matrix.at[i, j] = m_i * m_j / ((m_i + s_ij) * (m_i + m_j + s_ij))
        od_matrix = od_matrix.fillna(0)
        cost_matrix = self._normalize_and_transform(od_matrix)
        return cost_matrix

    def compute_logit_cost(self, beta=-0.01):
        nodes = self.grid['node'].tolist()
        masses = self.grid.set_index('node')[self.pop_col].to_dict()
        dist_matrix = self._compute_distance_matrix()
        od_matrix = pd.DataFrame(index=nodes, columns=nodes, dtype=float)
        for i in nodes:
            utilities = []
            for j in nodes:
                if i == j or np.isinf(dist_matrix.at[i, j]):
                    utilities.append(-np.inf)
                else:
                    utilities.append(beta * dist_matrix.at[i, j])
            # Compute softmax probabilities for destination choices from i
            exp_utilities = np.exp(np.array(utilities) - np.max(utilities))
            probs = exp_utilities / np.sum(exp_utilities)
            for idx, j in enumerate(nodes):
                od_matrix.at[i, j] = masses[i] * probs[idx] if i != j else 0
        od_matrix = od_matrix.fillna(0)
        cost_matrix = self._normalize_and_transform(od_matrix)
        return cost_matrix
