# Experiment for P only with bands and SVI read from related works.
# Models are fitted with 90% total labeled data choosen randomly.
# The remaining 10% data are converted to X_u.

from sklearn.ensemble import AdaBoostClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.naive_bayes import ComplementNB
from sklearn.neighbors import KNeighborsClassifier
import terrasensetk as tstk
import pandas as pd
import numpy as np
import os


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
    # TSAVI (B * (NIR - B * R - A)) / (RED + B * (NIR - A) + X * (1 + B^2))
    # CD = IR
    'TSAVI': lambda b: (0.5 * (b[7] - 0.5 * b[3] - 0.5)) / (b[3] + 0.5 * (b[7] - 0.5) + 0.5 * (1.0 + 0.5**2.0)),
    # MCARI ((700nm - 670nm) - 0.2 * (700nm - 550nm)) * (700nm /670nm)
    # CD = IR
    'MCARI': lambda b: ((b[4] - b[3]) - 0.2 * (b[4] - b[2])) * (b[4] / b[3]),
    # WDRVI (0.1*NIR-RED)/(0.1*NIR+RED)
    # CD = IR
    'WDRVI': lambda b: (0.1 * b[7] - b[3]) / (0.1 * b[7] + b[3]),
    # GNDVI = NIR-GREEN)/(NIR+GREEN)
    # CD = [-1, 1]
    'GNDVI': lambda b: (b[7] - b[2]) / (b[7] + b[2])
})


# Paser creation
print('Creating parser...')
parser = tstk.Parser(dataset)

for r in range(5):
    print(f'Run {r}')
    df = parser.create_df(indices=indices, bands=bands, groundtruth_f=['P'])


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
    
    print(f'Converting df...')
    # Removes all unlabeled data
    df = df.dropna(subset=['P'])
    # Make 10% unlabeled randomly
    unlabeled = 0.1
    indices_to_unlabel = df.sample(n=round(df.shape[0]*unlabeled), random_state=r).index
    df.loc[indices_to_unlabel, 'P'] = np.nan

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
        name=f'P_all_f_90%_Xl_Unbalanced_run{r}',
        parser=parser,
        models=sl+ssl,
        y='P',
        cv=tstk.StratifiedKFold(),
        verbose=True
    )

    results = exp.execute(
        parameter_optimization=True,
        save_results=f'./results/'
    )

    print('Calculating metrics...')
    df = exp.calculate_metrics(metrics=['acc', 'ball_acc', 'f1', 'prec', 'recall'])
    print(df)
    df.to_pickle(f'./results/{exp.name}/Df_metrics.pickle')
