# Experiment for P only with bands and SVI read from related works, and testing only with model
# HGradientBoosting and SelfTraining_HGradientBoosting.
# Models are fitted with 5% Labeled data

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


# Adds indices
print('Adding svis...')
# Adding new indices according to 
# https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/indexdb/
dataset.add_indices({
    # Soil productivity https://www.tandfonline.com/doi/abs/10.1080/10106049209354353
    # NDVI (MIR - NIR) / (MIR + NIR)
    # CD = [-1, 1]
    'NDVI': lambda b: (b[11] - b[7]) / (b[11] + b[7]),
    # EVI 2.5 * (NIR - RED) / ((NIR + 6*RED - 7.5*BLUE) + 1)
    # CD = [-1, 1]
    'EVI': lambda b: 2.5 * (b[7] - b[3]) / ((b[7] + 6.0 * b[3] - 7.5 * b[1]) + 1.0),
    # NDMI (820nm - 1600nm) / (820nm + 1600nm)
    # CD = [-1, 1]
    'NDMI': lambda b: (b[7] - b[10]) / (b[7] + b[10]),
    # PVI (1 / sqrt(a^2+ 1)) * (NIR - ar - b)
    # CD = [-1, 1]
    'PVI': lambda b: (1.0 / (0.5**2.0 + 1.0)**0.5) * (b[7] - 0.5 - 0.5),
    # SAVI (800nm - 670nm) / (800nm + 670nm + L) * (1 + L)
    # CD = [-1, 1]
    'SAVI': lambda b: (b[7] - b[3]) / (b[7] + b[3] + 0.5) * (1.0 + 0.5),
    # TSAVI (B * (NIR - B * R - A)) / (RED + B * (NIR - A) + X * (1 + B^2))
    # CD = [-1, 1]
    'TSAVI': lambda b: (0.5 * (b[7] - 0.5 * b[3] - 0.5)) / (b[3] + 0.5 * (b[7] - 0.5) + 0.5 * (1.0 + 0.5**2.0)),
    # SAVI2 NIR/(RED+b/a) -> Requires soil information that we do not have
    # CL (RED - BLUE) / RED
    # CD = [-1, 1]
    'CL': lambda b: (b[3] - b[1]) / b[3],
    # Chlorophyll Green ([760:800]/[540:560])^(-1)
    # CD = IR
    'Chlgreen': lambda b: (b[bands['B:VNIR7']] / b[bands['B:GREEN']]) ** -1,
    # Chlorophyll Index Green:
    # CD = IR
    'CIG': lambda b: b[bands['B:VNIR8a']] / b[bands['B:GREEN']] - 1,
    # Ferric iron  
    # CD = IR
    'FE2': lambda b: (b[bands['B:SWIR12']] / b[bands['B:VNIR8']]) + (b[bands['B:GREEN']] / b[bands['B:RED']]),

    # P
    # GRNDVI = (NIR-(GREEN+RED))/(NIR+(GREEN+RED))
    # CD = [-1, 1]
    'GRNDVI': lambda b: (b[7] - (b[2] + b[3])) / (b[7] + (b[2] + b[3])),
    # NDVI
    # GNDVI = NIR-GREEN)/(NIR+GREEN)
    # CD = [-1, 1]
    'GNDVI': lambda b: (b[7] - b[2]) / (b[7] + b[2]),
    # https://www.tandfonline.com/doi/abs/10.1080/01431160903439908
    # https://www.researchgate.net/profile/Ronghua-Ma-2/publication/268392354_Using_hyper-spectral_indices_to_detect_soil_phosphorus_concentration_for_various_land_use_patterns/links/54f64b780cf27d8ed71d75d6/Using-hyper-spectral-indices-to-detect-soil-phosphorus-concentration-for-various-land-use-patterns.pdf
    # NDSI(R523, R583) = (523 - 583) / (523 + 583)
    # CD = [-1, 1]
    'NDSI(R523, R583)': lambda b: (b[1] - b[2]) / (b[1] + b[2]),
    # GRI = R830/R550
    'GRI': lambda b: b[7]/b[2]
})

common_svi = ['NDVI', 'EVI', 'NDMI', 'PVI', 'SAVI', 'TSAVI', 'CL', 'Chlgreen', 'CIG', 'FE2']
P_svi = ['GRNDVI', 'NDVI', 'GNDVI', 'NDSI(R523, R583)', 'GRI']
P_svi = list(set(P_svi).union(common_svi))

# https://jast.modares.ac.ir/article-23-5138-en.html
# BLUE, RED
# https://www.sciencedirect.com/science/article/pii/S1537511006002704?casa_token=OBfL1jiQmAIAAAAA:ZbNlZXeIPmoZFlU7uTOjtX1vOHJ9_6K1IC80YCsbG1WAEyDRsKTfWrgaqfKbQZm2GZm7VykS
# 305nm - 1710nm
P_bands = list(dataset.get_bands().keys())[:-1]

# Paser creation
print('Creating parser...')
parser = tstk.Parser(dataset)

for r in range(5):
    print(f'Run {r}')
    
    df = parser.create_df(bands=P_bands, indices=P_svi, groundtruth_f=['P'])


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
    labeled = 0.75
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

    ssl = [
        tstk.SelfTraining(base_estimator=ComplementNB(), verbose=True),
        tstk.SelfTraining(base_estimator=HistGradientBoostingClassifier(), verbose=True),
        tstk.SelfTraining(base_estimator=KNeighborsClassifier(), verbose=True),
        tstk.SelfTraining(base_estimator=RandomForestClassifier(n_jobs=-1), verbose=True),
        tstk.SelfTraining(base_estimator=AdaBoostClassifier(), verbose=True)
    ]

    exp = tstk.Experiment(
        name=f'P_RWselected_f_75%_Xl_Balanced_run{r}',
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
