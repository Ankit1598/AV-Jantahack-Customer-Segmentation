import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder


train = pd.read_csv('./Train/Train.csv')
test = pd.read_csv('Test.csv')


train.Segmentation.value_counts()


train_copy  = train.copy()
test_copy = test.copy()
train_copy['train'] = 1
test_copy['train'] = 0

df = pd.concat([train_copy, test_copy], axis = 0)

cat_cols = ['Gender', 'Ever_Married', 'Graduated', 'Profession', 'Spending_Score', 'Var_1']
label_enc = {}

for col in cat_cols:
    df[col] = df[col].astype(str)
    enc = LabelEncoder().fit(df[col])
    df[col] = enc.transform(df[col])
    label_enc[col] = enc

cats = ['Gender', 'Ever_Married','Graduated','Profession','Spending_Score','Var_1']
df = pd.get_dummies(df, columns = cats)


def id_features(data):
    df = data.copy()
    df['week'] = df['ID']%7
    df['month'] = df['ID']%30
    df['year'] = df['ID']%365
    df['quarter'] = df['ID']%90
    return df


df = id_features(df)

train_copy = df.loc[df['train'] == 1]
test_copy = df.loc[df['train'] == 0]
Xcols = df.drop(['Segmentation', 'train'], axis = 1).columns
ycol = 'Segmentation'

X = train_copy[Xcols]
y = train_copy[ycol]

Xtest = test_copy[Xcols]


model = lgb.LGBMClassifier(n_estimators=300, max_features = .85, max_depth = 15, learning_rate = 1.1).fit(X, y)


pred = pd.DataFrame()
pred['ID'] = test['ID'].values
pred['Segmentation'] = pd.Series(model.predict(Xtest))
pred.to_csv('final.csv', index = None)
