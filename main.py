import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as stats
from sklearn import svm 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, auc, roc_curve
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline
from pandas.plotting import table
from TSVM import TSVM

df = pd.read_csv('datasets/water_potability.csv')
imputer = KNNImputer(n_neighbors = 2)
scaler = StandardScaler()
rus = RandomUnderSampler(random_state = 42)
sw_test = pd.DataFrame(columns = df.drop('Potability', axis = 1).columns, index = ['p-value(W)'])

print(df)

def missing_data_percent(df):
    percent = 100 * df.isnull().sum() / len(df)
    percent = percent[percent > 0].sort_values()
    return percent

def describe(df, stats):
    d = df.describe()
    return d._append(df.reindex(d.columns, axis = 1).agg(stats))


def shapiro_wilk_test(df: pd.DataFrame, cols: list, alpha = 0.05):
    for col in cols:
        _, p_shapiro = stats.shapiro(df[col])
        sw_test[col] = [p_shapiro]
        print(sw_test[col])

def create_save_table(df: pd.DataFrame, figsize: tuple | list, colWidth: float, fontsize: int, fileName: str):
    fig, ax = plt.subplots(figsize=figsize)
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    ax.set_frame_on(False)
    tabla = table(ax, df, loc='center', colWidths=[colWidth]*len(df.columns))
    tabla.auto_set_font_size(False)
    tabla.set_fontsize(fontsize)
    tabla.scale(.2, 2)
    plt.savefig(f'{fileName}.png', transparent=True)

descriptive = df.drop('Potability', axis = 1)
descriptive = describe(descriptive, ['median', 'skew'])
descriptive = descriptive.round(2)
create_save_table(descriptive, (15, 5), .65, 13, 'descriptive')

corr_table = df.corr()
corr_table = corr_table.round(3)
print(corr_table)
missingdatapercent = missing_data_percent(df)
sns.barplot(x = missingdatapercent.index, y = missingdatapercent)

df['ph'] = imputer.fit_transform(df['ph'].values.reshape(-1, 1))[:, 0]
df['Sulfate'] = imputer.fit_transform(df['Sulfate'].values.reshape(-1, 1))[:, 0]
df['Trihalomethanes'] = imputer.fit_transform(df['Trihalomethanes'].values.reshape(-1, 1))[:, 0]

sns.heatmap(data = df.corr())
plt.show()

sns.countplot(df)
plt.show()


fig = plt.figure(1, (10, 4))
for i, j in enumerate(df.drop('Potability', axis = 1).columns):
    ax = plt.subplot(3, 3, i + 1)
    sns.histplot(df[j])
    plt.tight_layout()



shapiro_wilk_test(df, df.drop('Potability', axis = 1).columns)
sw_test = sw_test.round(2)
create_save_table(sw_test, (10, 1), 0.65, 13, 'shapiro_wilk_test')



x = df.drop('Potability', axis = 1)
y = df['Potability']


fig1, ax = plt.subplots(figsize=[15,7])
ax = sns.boxplot(data=x.drop('Solids', axis = 1), orient="h")
sns.despine(offset=10, trim=True)
plt.show()

comparing_ml_models = pd.DataFrame(columns = ['Model Name', 'precision', 'recall', 'AUC'])

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)
pipeline = Pipeline([('rus', rus)])

x_train, y_train = pipeline.fit_resample(x_train, y_train)

scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)

model = svm.SVC(kernel = 'rbf', class_weight = {0: .58, 1: .42})

clf = GridSearchCV(model, {'C': np.linspace(.01, 2, 5), 'gamma': np.linspace(.06, 1, 5)},\
                          verbose = 2, cv = 5, scoring = 'accuracy', refit = True)

clf.fit(scaled_x_train, y_train)

clf_pred = clf.predict(scaled_x_test)

conf_matrix = confusion_matrix(y_test, clf_pred)
conf_matrix = ConfusionMatrixDisplay(conf_matrix)
conf_matrix.plot()
plt.show()
print(classification_report(y_test, clf_pred))

fp, tp, th = roc_curve(y_test, clf_pred)
comparing_ml_models.loc[0, 'Model Name'] = 'SVM'
comparing_ml_models.loc[0, 'precision'] = round(precision_score(y_test, clf_pred, average = 'weighted'), 3)
comparing_ml_models.loc[0, 'recall'] = round(recall_score(y_test, clf_pred, average = 'weighted'), 3)
comparing_ml_models.loc[0, 'AUC'] = round(auc(fp, tp), 3)

# TSVM 

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 42, test_size = 0.2)
y_train = y_train.map({0: -1, 1: 1})
y_test = y_test.map({0: -1, 1: 1})

scaled_x_train = scaler.fit_transform(x_train)
scaled_x_test = scaler.transform(x_test)


C = np.linspace(.5, 2, 3)
gamma = np.linspace(.1, 1, 2)
result = {'accuracy': 0, 'C': 2.5, 'gamma': .2}

'''
for i in C:
    for j in gamma:
        print('starts')
        model = TSVM(kernel = 'rbf', C = i, gamma = j)
        model.train(scaled_x_train, y_train, scaled_x_test)
        model_pred = model.predict(scaled_x_test)
        accuracy = accuracy_score(y_test, model_pred)
        if(accuracy > result['accuracy']):
            result['accuracy'] = accuracy
            result['C'] = i
            result['gamma'] = j
        print(f'C={i}, gamma={j} checked.')
'''
model = TSVM(kernel = 'rbf', C = result['C'], gamma = result['gamma'], weights = 'balanced')
print(result)
model.train(scaled_x_train, y_train, scaled_x_test)
model_pred = model.predict(scaled_x_test)
conf_matrix = confusion_matrix(y_test, model_pred)
conf_matrix = ConfusionMatrixDisplay(conf_matrix)
conf_matrix.plot()
plt.show()
print(classification_report(y_test, model_pred))


fp, tp, th = roc_curve(y_test, clf_pred)
comparing_ml_models.loc[1, 'Model Name'] = 'TSVM'
comparing_ml_models.loc[1, 'precision'] = round(precision_score(y_test, model_pred, average = 'weighted'), 3)
comparing_ml_models.loc[1, 'recall'] = round(recall_score(y_test, model_pred, average = 'weighted'), 3)
comparing_ml_models.loc[1, 'AUC'] = round(auc(fp, tp), 3)

print(comparing_ml_models)

plt.subplots(figsize=(13,5))
sns.barplot(x="Model Name", y="precision", data = comparing_ml_models)
plt.xticks(rotation=90)
plt.title('Models precision Comparison')
plt.show()

plt.subplots(figsize=(13,5))
sns.barplot(x="Model Name", y="recall", data = comparing_ml_models)
plt.xticks(rotation=90)
plt.title('Models recall Comparison')
plt.show()


