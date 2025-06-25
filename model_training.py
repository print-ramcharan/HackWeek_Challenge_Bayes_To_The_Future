from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator
import pandas as pd
from pgmpy.inference import VariableElimination


df = pd.read_csv("heart_disease.csv")
# given structure
model = DiscreteBayesianNetwork([
    ('age', 'fbs'),
    ('fbs', 'target'),
    ('target', 'chol'),
    ('target', 'thalach')
])

model.fit(df, estimator=MaximumLikelihoodEstimator)

inference = VariableElimination(model)

# Probability of heart disease (target) given a normalized age value
result1 = inference.query(variables=["target"], evidence={"age": 70})
print("\nProbability of heart disease given age = 70:\n", result1)

# Cholesterol distribution given heart disease is present
result2 = inference.query(variables=["chol"], evidence={"target": 1})
print("\nCholesterol distribution given target=1:\n", result2)

# Heart disease prediction given age and fbs
result3 = inference.query(variables=["target"], evidence={"age": 69, "fbs": 1})
print("\nHeart disease probability given age=69 and fbs=1:\n", result3)