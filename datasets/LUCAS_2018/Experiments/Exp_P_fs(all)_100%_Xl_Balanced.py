# Experiment for P only with bands and SVI read from related works, and testing only with model
# HGradientBoosting.
# Models are fitted with 100% Labeled data

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
    # Create df with 50% 50% fertile infertile balance. It is all labeled
    df_fertile = df[df['P'] == 1.0]
    df_infertile = df[df['P'] == 0.0].head(df_fertile.shape[0])
    df = pd.concat([df_fertile, df_infertile], ignore_index=True)
    print(f'Df with only {df.shape[0]} Xl data points.')

    # Make 5% labeled but keep the 50 50 balance
    labeled = 1
    Xl_rows = round(df.shape[0]*labeled*0.5)
    print(f'{Xl_rows*2} are Xl')
    # Identify indices for 2.5% of 'P' values as 1.0
    indices_1_0 = df[df['P'] == 1.0].sample(n=Xl_rows, random_state=r).index
    # Identify indices for 2.5% of 'P' values as 0.0
    indices_0_0 = df[df['P'] == 0.0].sample(n=Xl_rows, random_state=r).index
    df.loc[~df.index.isin(indices_1_0.union(indices_0_0)), 'P'] = np.nan

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

    exp = tstk.Experiment(
        name=f'P_fs(all)_100%_Xl_Balanced_run{r}',
        parser=parser,
        models=sl,
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
