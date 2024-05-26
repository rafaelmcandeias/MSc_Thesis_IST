# Experiment for P with only the features resultant from 3 Feature selection algorithms applied to
# all possible features.
# Df is made of only wheat crops

from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import ComplementNB
import terrasensetk as tstk
import pandas as pd


# Creates dataset
print('Creating dataset...')
eopatches_path = './EOPatches/sentinel-2'
dataset = tstk.Dataset(eopatches_path)
print(f'Extracted: {len(dataset.get_tspatches())} TSPatches')
bands = dataset.get_bands()

# From feature selection:
indices = ['TSAVI', 'MCARI', 'WDRVI', 'GNDVI']
bands = ['B:UB', 'B:VNIR8', 'B:SWIR9']

# Adds indices
print('Adding svis...')
# Adding new indices according to 
# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
dataset.add_indices({
    # Soil productivity https://www.tandfonline.com/doi/abs/10.1080/10106049209354353
    # PVI (1 / sqrt(a^2+ 1)) * (NIR - ar - b)
    # CD = [-1, 1]
    'PVI': lambda b: (1.0 / (0.5**2.0 + 1.0)**0.5) * (b[7] - 0.5 - 0.5),
    # TSAVI (B * (NIR - B * R - A)) / (RED + B * (NIR - A) + X * (1 + B^2))
    # CD = [-1, 1]
    'TSAVI': lambda b: (0.5 * (b[7] - 0.5 * b[3] - 0.5)) / (b[3] + 0.5 * (b[7] - 0.5) + 0.5 * (1.0 + 0.5**2.0)),
    # https://journals.ashs.org/jashs/view/journals/jashs/132/5/article-p611.xml
    # NDVI(780, 670) = (780nm - 670nm) / (780nm + 670nm)
    'NDVI(780, 670)': lambda b: (b[6] - b[3]) / (b[6] + b[3]),
    # GRNDVI = (NIR-(GREEN+RED))/(NIR+(GREEN+RED))
    # CD = [-1, 1]
    'GRNDVI': lambda b: (b[7] - (b[2] + b[3])) / (b[7] + (b[2] + b[3]))
})


# Paser creation
print('Creating parser...')
parser = tstk.Parser(dataset)
df = parser.create_df(indices=indices, bands=bands, groundtruth_f=['P', 'LC1_Desc'])


# Convert all P to numeric
print('Setting df...')
df.loc[:, 'P'] = df['P'].apply(lambda x: pd.to_numeric(x, errors='coerce'))
df['P'] = df['P'].astype(float)

# Classify
def classify(x, lower, upper):
    if lower <= x <= upper:
        return 1.0
    elif x < lower or x > upper:
        return 0.0
    return x
# https://turf.unl.edu/NebGuides/g2265.pdf deviam estar acima de 7 a 21, por isso sugerem 25
P_fertility_limits = [10.9, 21.4]
df['P'] = df['P'].apply(classify, args=P_fertility_limits)
df = df[df['LC1_Desc'] == 'Common wheat']
parser.set_df(df)


# Model selection
print('Creating Experiment...')
# Available models
cnb = tstk.ComplementNB(verbose=True)
hgb = tstk.HGradientBoosting(verbose=True)
knn = tstk.KNN(verbose=True)
rf = tstk.RandomForest(verbose=True, n_jobs=-1)
adaBoosting = tstk.AdaBoosting(verbose=True)

sl = [cnb, hgb, knn, rf, adaBoosting]

ssl = [
    tstk.SelfTraining(base_estimator=ComplementNB(), verbose=True),
    tstk.SelfTraining(base_estimator=HistGradientBoostingClassifier(), verbose=True),
    tstk.SelfTraining(base_estimator=KNeighborsClassifier(), verbose=True),
    tstk.SelfTraining(base_estimator=RandomForestClassifier(n_jobs=-1), verbose=True),
    tstk.SelfTraining(base_estimator=AdaBoostClassifier(), verbose=True)
]

exp = tstk.Experiment(
    name='P_fs(all)_wheat',
    parser=parser,
    models=sl+ssl,
    y='P',
    cv=tstk.StratifiedKFold(),
    verbose=True
)

print(f'Executing exp with features:\bBands:{bands}\nSVI:{indices}')
results = exp.execute(
    parameter_optimization=True,
    save_results='./results/'
)

print('Calculating metrics...')
df = exp.calculate_metrics(metrics=['acc', 'ball_acc', 'f1', 'prec', 'recall'])
print(df)
df.to_pickle(f'./results/{exp.name}/Df_metrics.pickle')