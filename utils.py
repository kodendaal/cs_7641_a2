from mlrose_hiive import FourPeaks, DiscreteOpt
from mlrose_hiive.algorithms.crossovers import UniformCrossOver
from mlrose_hiive.algorithms.mutators import ChangeOneMutator
from mlrose_hiive.fitness import MaxKColor
from mlrose_hiive.opt_probs.discrete_opt import DiscreteOpt
import networkx as nx
import os
import pickle as pk

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score, log_loss
import time

# maximize FourPeaks Generator - NOT UPDATED IN NEWEST MLROSE-HIIVE PACKAGE
class FourPeaksGenerator:
    """Generator class for Four Peaks."""
    @staticmethod
    def generate(size=20, t_pct=0.2, seed=None):
        # np.random.seed(seed)
        fitness = FourPeaks(t_pct=t_pct)
        problem = DiscreteOpt(length=size, fitness_fn=fitness, maximize=True, max_val=2) 
        return problem
    

# maximize FourPeaks Generator - NOT UPDATED IN NEWEST MLROSE-HIIVE PACKAGE
class MaxKColorGenerator:
    @staticmethod
    def generate(number_of_nodes=20, max_connections_per_node=4, max_colors=None, maximize=False, seed=None):

        """
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_hiive.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        """
        # np.random.seed(seed)
        # all nodes have to be connected, somehow.
        node_connection_counts = 1 + np.random.randint(max_connections_per_node, size=number_of_nodes)

        node_connections = {}
        nodes = range(number_of_nodes)
        for n in nodes:
            all_other_valid_nodes = [o for o in nodes if (o != n and (o not in node_connections or
                                                                      n not in node_connections[o]))]
            count = min(node_connection_counts[n], len(all_other_valid_nodes))
            other_nodes = sorted(np.random.choice(all_other_valid_nodes, count, replace=False))
            node_connections[n] = [(n, o) for o in other_nodes]

        # check connectivity
        g = nx.Graph()
        g.add_edges_from([x for y in node_connections.values() for x in y])

        for n in nodes:
            cannot_reach = [(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()]
            for s, f in cannot_reach:
                g.add_edge(s, f)
                check_reach = len([(n, o) if n < o else (o, n) for o in nodes if o not in nx.bfs_tree(g, n).nodes()])
                if check_reach == 0:
                    break

        edges = [(s, f) for (s, f) in g.edges()]
        problem = MaxKColorOpt(edges=edges, length=number_of_nodes, maximize=maximize, max_colors=max_colors, source_graph=g)
        return problem
    
import numpy as np


class MaxKColorOpt(DiscreteOpt):
    def __init__(self, edges=None, length=None, fitness_fn=None, maximize=False,
                 max_colors=None, crossover=None, mutator=None, source_graph=None):

        if (fitness_fn is None) and (edges is None):
            raise Exception("fitness_fn or edges must be specified.")

        if length is None:
            if fitness_fn is None:
                length = len(edges)
            else:
                length = len(fitness_fn.weights)

        self.length = length

        if fitness_fn is None:
            fitness_fn = MaxKColor(edges, maximize)

        # set up initial state (everything painted one color)
        if source_graph is None:
            g = nx.Graph()
            g.add_edges_from(edges)
            self.source_graph = g
        else:
            self.source_graph = source_graph

        self.stop_fitness = self.source_graph.number_of_edges() if maximize else 0

        fitness_fn.set_graph(self.source_graph)
        # if none is provided, make a reasonable starting guess.
        # the max val is going to be the one plus the maximum number of neighbors of any one node.
        if max_colors is None:
            total_neighbor_count = [len([*self.source_graph.neighbors(n)]) for n in range(length)]
            max_colors = 1 + max(total_neighbor_count)
        self.max_val = max_colors

        crossover = UniformCrossOver(self) if crossover is None else crossover
        mutator = ChangeOneMutator(self) if mutator is None else mutator
        super().__init__(length, fitness_fn, maximize, max_colors, crossover, mutator)

        # state = [len([*g.neighbors(n)]) for n in range(length)]
        state = np.random.randint(max_colors, size=self.length)
        np.random.shuffle(state)
        # state = [0] * length
        self.set_state(state)

    def can_stop(self):
        return int(self.get_fitness()) == self.stop_fitness
    
class MaxKColor:
    """Fitness function for Max-k color optimization problem. Evaluates the
    fitness of an n-dimensional state vector
    :math:`x = [x_{0}, x_{1}, \\ldots, x_{n-1}]`, where :math:`x_{i}`
    represents the color of node i, as the number of pairs of adjacent nodes
    of the same color.

    Parameters
    ----------
    edges: list of pairs
        List of all pairs of connected nodes. Order does not matter, so (a, b)
        and (b, a) are considered to be the same.

    Example
    -------
    .. highlight:: python
    .. code-block:: python

        >>> import mlrose_hiive
        >>> import numpy as np
        >>> edges = [(0, 1), (0, 2), (0, 4), (1, 3), (2, 0), (2, 3), (3, 4)]
        >>> fitness = mlrose_hiive.MaxKColor(edges)
        >>> state = np.array([0, 1, 0, 1, 1])
        >>> fitness.evaluate(state)
        3

    Note
    ----
    The MaxKColor fitness function is suitable for use in discrete-state
    optimization problems *only*.

    If this is a cost minimization problem: lower scores are better than
    higher scores. That is, for a given graph, and a given number of colors,
    the challenge is to assign a color to each node in the graph such that
    the number of pairs of adjacent nodes of the same color is minimized.

    If this is a cost maximization problem: higher scores are better than
    lower scores. That is, for a given graph, and a given number of colors,
    the challenge is to assign a color to each node in the graph such that
    the number of pairs of adjacent nodes of different colors are maximized.
    """

    def __init__(self, edges, maximize=False):

        # Remove any duplicates from list
        edges = list({tuple(sorted(edge)) for edge in edges})

        self.graph_edges = None
        self.edges = edges
        self.prob_type = 'discrete'
        self.maximize = maximize

    def evaluate(self, state):
        """Evaluate the fitness of a state vector.

        Parameters
        ----------
        state: array
            State array for evaluation.

        Returns
        -------
        fitness: float
            Value of fitness function.
        """

        fitness = 0

        # this is the count of neigbor nodes with the same state value.
        # Therefore state value represents color.
        # This is NOT what the docs above say.

        edges = self.edges if self.graph_edges is None else self.graph_edges

        if self.maximize:
            # Maximise the number of adjacent nodes not of the same colour.
            fitness = sum(int(state[n1] != state[n2]) for (n1, n2) in edges)
        else:
            # Minimise the number of adjacent nodes of the same colour.
            fitness = sum(int(state[n1] == state[n2]) for (n1, n2) in edges)
        return fitness

    def get_prob_type(self):
        """ Return the problem type.

        Returns
        -------
        self.prob_type: string
            Specifies problem type as 'discrete', 'continuous', 'tsp'
            or 'either'.
        """
        return self.prob_type

    def set_graph(self, graph):
        self.graph_edges = [e for e in graph.edges()]


def data_store(df, file_name, directory_root='./outputdir'):

    # extract parameters
    df_run_stats, df_run_curves = df

    # Define the new directory path for each iteration
    new_directory = os.path.join(directory_root, str(file_name[:9])) 
    
    # Create the directory if it does not exist
    if not os.path.exists(new_directory):
        os.makedirs(new_directory)
    
    # Define the file paths
    csv_file_path_df1 = os.path.join(new_directory, str(file_name) + '_stats.csv')
    csv_file_path_df2 = os.path.join(new_directory, str(file_name) + '_curves.csv')
    pickle_file_path_df1 = os.path.join(new_directory, str(file_name) + '_stats.pickle')
    pickle_file_path_df2 = os.path.join(new_directory, str(file_name) + '_curves.pickle')
    
    # Save the DataFrames as CSV files
    df_run_stats.to_csv(csv_file_path_df1, index=False)
    df_run_curves.to_csv(csv_file_path_df2, index=False)
    
    # Save the DataFrames as pickle files
    with open(pickle_file_path_df1, 'wb') as f:
        pk.dump(df_run_stats, f)
    
    with open(pickle_file_path_df2, 'wb') as f:
        pk.dump(df_run_curves, f)
    
    print(f'Saved files in {new_directory}')


# source: Metrics sklearn: https://scikit-learn.org/stable/modules/model_evaluation.html
def final_classifier_evaluation(clf, X_train, X_test, y_train, y_test):
    
    # calculate the model fitting time
    start = time.time()
    clf.fit(X_train, y_train)
    end = time.time()
    training_time = end - start
    print("Model Training Time (s)", training_time)
    
    # calcualte the model prediction time
    start = time.time()  
    y_pred = clf.predict(X_test)
    end = time.time()
    pred_time = end - start
    print("Model Prediction Time (s):",pred_time)

    # evaluate classification metrics (weighted for multi-class evaluation)
    accuracy = accuracy_score(y_test,y_pred) 
    print("Accuracy:",accuracy)
    precision = precision_score(y_test,y_pred, average='weighted')
    print("Precision:",precision)
    recall = recall_score(y_test,y_pred, average='weighted')
    print("Recall:",recall)
    f1 = f1_score(y_test,y_pred, average='weighted') # harmonic mean between precision/recall
    print("F1-Score:",f1)
    f1_manual = 2*precision*recall/(precision+recall)   
    print("Manual F1 Score:",f1_manual)
