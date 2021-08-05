import numpy as np
import pandas as pd
import templates
import plotly.graph_objects as go


class Kmeans():
    """A class for computing K-means from scratch
    
    Attributes
    ----------
    X : numpy array
        The input values for computing K-means
    K : int
        Number of clusters
    max_iterations : int
        Max iterations (recomputing of centroids)
    num_individuals : int
        Number of observations in the dataset
    num_features : int
        Number of variables in the dataset
    _centroids : list
        List of all centroids calculated in each step
    centroids : numpy array
        Array with final centroids
    mode : str
        Whether or not the centroids should be initialized randomly or with the kmeans++ algorithm
    _labels : list
        List of all labels calculated by each step
    labels: numpy array
        Array with final labels (clusters)
    
    Methods
    -------
    initialize_random_centroids()
        Initialize centroids randomly
    initialize_improved_centroids()
        Initialize centroids using the kmeans++ algorithm
    create_clusters(centroids)
        Create K clusters by taking the minimal euclidean distance between the centroids and data points
    calculate_new_centroids(clusters)
        Realocate centroids to the center of each cluster
    predict_cluster()
        Label each observation to its cluster
    fit
        Initialize centroids and repeat the centroid-clustering functions
        up to reach the max iterations or getting no difference in the means of each centroid
    """

    def __init__(self, X, K, mode='random', seed=None):

        try:
            self.X = X.values
        except:
            self.X = X
        
        self.K = K
        self.mode = mode
        self.seed = seed
        self.max_iterations = 100
        self.num_individuals, self.num_features = self.X.shape


    def initialize_random_centroids(self):
        """Initialize K centroids at random"""
        if self.seed is not None:
            rng = np.random.RandomState(self.seed)
            centroids_idx = rng.choice(range(self.num_individuals), size=self.K, replace=False)
        else:
            centroids_idx = np.random.choice(range(self.num_individuals), size=self.K, replace=False)
        centroids = self.X[centroids_idx]
        self._centroids = [centroids]
        return centroids


    def initialize_improved_centroids(self):
        """Initialize centroids using the kmeans++ algorithm"""
        dist = []
        dist.append(self.X[np.random.randint(0, self.num_individuals)])
        while len(dist)<self.K:
            d2 = np.array([min([np.square(np.linalg.norm(i-c, None)) for c in dist]) for i in self.X])
            prob = d2/d2.sum()
            cum_prob = prob.cumsum()
            r = np.random.random()
            ind = np.where(cum_prob >= r)[0][0]
            dist.append(self.X[ind])
        centroids = np.array(dist)
        self._centroids = [centroids]
        return centroids


    def create_clusters(self, centroids):
        """Create K clusters by taking the minimal Euclidean distance between the centroids and data points

        Parameters
        ----------
        centroids : numpy.array
            Numpy array if K centroids to calculate K clusters using Euclidean distance
        """
        clusters = [[] for _ in range(self.K)]
        
        for point_idx, point in enumerate(self.X):
            closest_centroid = np.argmin(np.sqrt(np.sum((point - centroids)**2, axis=1)))
            clusters[closest_centroid].append(point_idx)
            
        return clusters


    def calculate_new_centroids(self, clusters):
        """Realocate centroids to the center of each cluster

        Parameters
        ----------
        clusters : list
            List of points belonging to a cluster for calculaging the mean by cluster,
            and therefore getting the new centroid position
        """
        centroids = np.zeros((self.K, self.num_features))
        
        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(self.X[cluster], axis=0)
            centroids[idx] = new_centroid
        
        self._centroids.append(centroids)
        return centroids


    def predict_cluster(self, clusters):
        """Label each observation to its cluster
        
        Parameters
        ----------

        clusters : list
            List of clusters with point indexes
        """
        y_pred = np.zeros(self.num_individuals, dtype=int)
        
        for cluster_idx, cluster in enumerate(clusters):
            y_pred[cluster] = cluster_idx
        
        return y_pred

    
    def fit(self):
        """Initialize centroids and repeat the centroid-clustering functions
        up to reach the max iterations or getting no difference in the means of each centroid
        """""
        centroids = self.initialize_random_centroids() if self.mode=="random" else self.initialize_improved_centroids()
        self._labels = []
        
        for it in range(self.max_iterations):
            clusters = self.create_clusters(centroids)
            self._labels.append(self.predict_cluster(clusters))
            
            previous_centroids = centroids
            centroids = self.calculate_new_centroids(clusters)
            
            diff = centroids - previous_centroids
            
            if not diff.any():
                self._centroids = self._centroids[:-1]
                break
        
        self.centroids = self._centroids[-1]
        self.labels = self._labels[-1]
        return self


def calculate_WSS(data, target_k, max_k, mode="random"):
    """Calculate Elbow
        
    Parameters
    ----------

    target_k : int
        Target k to return model
    max_k : int
        Numbers of k to be analyzed
    mode : str
        Whether or not the centroids should be initialized randomly or with the kmeans++ algorithm
    """
    sse = []
    for k in range(1, max_k+1):
        kmeans_model = Kmeans(data, k, mode).fit()
        curr_sse = 0
        
        if k == target_k:
            model = kmeans_model

        for i in range(len(data)):
            centroid = kmeans_model.centroids[kmeans_model.labels[i]]
            curr_sse += np.sum((data[i] - centroid) ** 2)

        sse.append(curr_sse)
    return model, sse


def plot(model, velocidade=2):
    """Plot animated K-means' steps

    Parameters
    ----------

    model: Kmeans class
        A Kmeans class after applying the `.fit()` method.
    velocidade : int
        Velocidade em segundos para transição entre frames
    """
    
    # Convert velocity from seconds to miliseconds
    velocidade*=1000 

    # Set default colors
    tab10_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    color_dict = {i: color for i,color in enumerate(tab10_colors)}

    # Get random K colors
    if model.K <= 10:
        palette = np.random.choice(tab10_colors, model.K, replace=False)
    else:
        palette = np.random.choice(tab10_colors, model.K, replace=True)
        
    # Set Figure Layout
    layout = templates.plotly_dark
    layout["showlegend"] = False
    layout["updatemenus"] = [
        {
            "buttons": [
                {
                    "args": [None, {"frame": {"duration": velocidade, "redraw": False},
                                    "fromcurrent": True, "font": {"color":"red"},
                                    "transition": {"duration": velocidade, "easing": "quadratic-in-out"}}
                             ],
                    "label": "Play",
                    "method": "animate"
                },
                {
                    "args": [[None], {"frame": {"duration": 0, "redraw": False},
                                      "mode": "immediate",
                                      "transition": {"duration": 0}}],
                    "label": "Pause",
                    "method": "animate"
                }
            ],
            "font": dict(color="black"),
            "bgcolor":"orange",
            "direction": "left",
            "pad": {"r": 10, "t": 87},
            "showactive": True,
            "type": "buttons",
            "x": 0.1,
            "xanchor": "right",
            "y": 0,
            "yanchor": "top"
        }
    ]
    
    # Set slider for animation
    sliders_dict = {
        "active": 0,
        "yanchor": "top",
        "xanchor": "left",
        "currentvalue": {
            "font": {"size": 20},
            # "prefix": "Step:",
            "visible": True,
            "xanchor": "right"
        },
        "transition": {"duration": velocidade/2, "easing": "cubic-in-out"},
        "pad": {"b": 10, "t": 50},
        "len": 0.9,
        "x": 0.1,
        "y": 0,
        "steps": []
    }
    
    # Set function to add slider steps
    def make_slider_step(i, label, value=None, redraw=False):
        slider_step = dict(
            args = [
                [i],
                {"frame": {"duration": velocidade/2, "redraw": redraw},
                 "transition": {"duration": velocidade/2},
                 "mode": "immediate"}
            ],
            label = label,
            value = value if value is not None else label,
            method = "animate"
        )
        return slider_step
    
    # Initialize Frames
    frames = []
    
    points_init = go.Scatter(x=model.X[:,0].tolist(), y=model.X[:,1].tolist(), mode="markers", marker=dict(color="gray"), name='points')
    centroids_init = go.Scatter(x=[], y=[], mode="markers", marker=dict(color=list(color_dict.values()), symbol='x', size=30), name='centroids')

    points = go.Scatter(x=model.X[:,0].tolist(), y=model.X[:,1].tolist(), mode="markers", marker=dict(color="gray", size=10, opacity=0.8), name='points')
    centroids = go.Scatter(x=[], y=[], mode="markers", marker=dict(color=list(color_dict.values()), symbol='x', size=30), name='centroids')
    
    # Add frames in loop
    c = 0
    for i in range(len(model._centroids)):

        centroids.update(dict(x=model._centroids[i][:,0], y=model._centroids[i][:,1]))

        c += 1
        frame1 = go.Frame(data=[
            points,
            centroids
        ], name=c)

        frames.append(frame1)
        sliders_dict['steps'].append(make_slider_step(c, i+1))

        colors = pd.Series(model._labels[i]).map(color_dict).values
        points.update(dict(marker=dict(color=colors)))

        if i != len(model._centroids)-1:
            c += 1
            frame2 = go.Frame(data=[
                points,
                centroids
            ], name=c)

            frames.append(frame2)
            sliders_dict['steps'].append(make_slider_step(c, ''))
    
    # Build figure
    layout['sliders'] = [sliders_dict]
    data = [points_init, centroids_init]

    fig_data = {"data": data, "layout": go.Layout(layout), "frames": frames}
    fig = go.Figure(fig_data)
    return fig
    
