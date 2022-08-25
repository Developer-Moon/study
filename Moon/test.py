import pandas as pd
import pandas_profiling

path = './_data/kaggle_titanic/'
train_set = pd.read_csv(path + 'train.csv')

profile = train_set.profile_report()
print(profile)
profile.to_file(output_file="test.html")