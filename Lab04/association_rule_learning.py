import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Read the groceries.csv file (assuming each row is a transaction)
df = pd.read_csv('groceries.csv', header=None)
transactions = df.stack().groupby(level=0).apply(list).tolist()

#Building the Frequent Item dataframe using Apriori algorithm
te = TransactionEncoder()
te_ary = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_ary, columns=te.columns_)

# Apply the Apriori algorithm to find frequent itemsets with a minimum support > 8%
frequent_itemsets = apriori(df_encoded, min_support=0.08, use_colnames=True)
print(frequent_itemsets)
print()

rules = association_rules(frequent_itemsets, metric="lift", min_threshold=1)
print(rules)
print()
print()

high_quality_rules = rules[(rules['lift'] > 4) & (rules['confidence'] > 0.8)]
print("Number of rules with lift > 4 and confidence > 0.8:", len(high_quality_rules))

