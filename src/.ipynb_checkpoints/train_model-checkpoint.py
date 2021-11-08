from main import *
from sklearn.model_selection import train_test_split

# In a real situation, this file will be the pipeline file that reads in data from database and retrains itself based on new batches of data.

#importing the data
df = pd.read_csv('../data/cs-training.csv', index_col = 0)

df_feature = df.drop(columns = ['SeriousDlqin2yrs'])
df_target = df['SeriousDlqin2yrs']

X_train, X_cvtest, y_train, y_cvtest = train_test_split(df_feature,
    df_target, train_size=0.75, test_size=0.25, random_state=42)

pipe = trainAdaBoostPipeline(X_train, y_train, optimize = True)

exportPipeline(pipe, version = '1')
