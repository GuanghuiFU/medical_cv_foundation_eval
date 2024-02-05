import pandas as pd
from scipy.stats import ttest_rel

model1_path = 'your/model1/performance/eval_model1.csv'
model2_path = 'your/model2/performance/eval_model2.csv'

dice1 = pd.read_csv(model1_path)
dice2 = pd.read_csv(model2_path)

dice_scores1 = dice1['dice']
dice_scores2 = dice2['dice']

t_stat, p_value = ttest_rel(dice_scores1, dice_scores2)

print("T-statistic:", t_stat)
print("P-value:", p_value)