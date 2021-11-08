from credit_project import load_model,predict_model
from sklearn.model_selection import train_test_split

# In a real situation, this file will be the pipeline file that reads in data from database and produces a batch output automatically.

df_test = pd.read_csv('../data/cs-test.csv', index_col = 0).drop(columns=['SeriousDlqin2yrs'])
df_test_results = pd.DataFrame(grid.predict_proba(df_test)[:,0])
df_test_results.columns = ['Probability']
df_test_results.index = df_test.index
df_test_results.index.rename('Id', inplace=True)
df_test_results.to_csv('cs-test-results.csv')