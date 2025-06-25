import networkx as nx
import matplotlib.pyplot as plt
from pgmpy.models import DiscreteBayesianNetwork
import pandas as pd
from pgmpy.estimators import MaximumLikelihoodEstimator

df = pd.read_csv("heart_disease.csv")

model = DiscreteBayesianNetwork([
    ('age', 'fbs'),
    ('fbs', 'target'),
    ('target', 'chol'),
    ('target', 'thalach')
])
model.fit(df, estimator=MaximumLikelihoodEstimator)

# Use a plain NetworkX DiGraph
G = nx.DiGraph(model.edges())

pos = nx.spring_layout(G, seed=42)
nx.draw(G, pos=pos, with_labels=True, node_size=2000, node_color='lightblue',
        font_size=10, arrows=True, edge_color='gray')
plt.title("Bayesian Network Structure")
plt.tight_layout()
plt.show()
