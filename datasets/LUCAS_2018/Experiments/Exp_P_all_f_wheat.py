# Experiment for P with all features available
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, AdaBoostClassifier
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

    # N:
    # https://www.researchgate.net/profile/Alireza-Sharifi-3/publication/369786888_Remotely_sensed_normalized_difference_red-edge_index_for_rangeland_biomass_estimation/links/64a586cdb9ed6874a5fc6285/Remotely-sensed-normalized-difference-red-edge-index-for-rangeland-biomass-estimation.pdf
    # TCARI 3 * ((700nm - 670nm) - 0.2 * (700nm - 550nm) * (700nm / 670nm))
    'TCARI': lambda b: 3.0 * ((b[4] - b[3]) - 0.2 * (b[4] - b[2]) * (b[4] / b[3])),
    # MCARI ((700nm - 670nm) - 0.2 * (700nm - 550nm)) * (700nm /670nm)
    'MCARI': lambda b: ((b[4] - b[3]) - 0.2 * (b[4] - b[2])) * (b[4] / b[3]),
    # https://d1wqtxts1xzle7.cloudfront.net/41175778/541d205e0cf241a65a15cf4e.pdf20160115-19908-1vl4y9j-libre.pdf?1452845318=&response-content-disposition=inline%3B+filename%3DFei_Li_et_al_2014.pdf&Expires=1700333481&Signature=hHB-rIKRDNf38bw~LY-nkm8zX~Boqe902KEX9JgfVrSmC6nK3e0xAnr0yDb-OB7InN9Ma0KNgfhoApyDQ-HxdQbUpqkX0pKctl98TvGeJkw6EplbEPotSAYv2Otj~RD6Lm6mBQh3tTWSXJNerkCD12fu0b06AaBjz3H3EHQdL2lHCaUbZnLDe3ETzTbmubhEX92-z~XGyEtxvWlYaqtmfNLfmwKeWxFubFyQxCq~X2yMXbx6Njn2eMsgjAbSJlM96mZnep4h7iHIcVlpWsy91b1ryQM5i7ym1PT3ktR9oFRqnG2eymkhn5dp8zSELeY8ZYP~OKtPU-iMzOfEJAOrrA__&Key-Pair-Id=APKAJLOHF5GGSLRBV4ZA
    # CCCI ((NIR - rededge)/( NIR + rededge))/((NIR - Red)/( NIR + Red))
    'CCCI': lambda b: ((b[7] - b[4]) / (b[7] + b[4])) / ((b[7] - b[3]) / (b[7] + b[3])),
    # MTCI (754nm − 709nm) / (709nm − 681nm) -> 681nm unavailable for S-2A
    # NDRE (NIR - rededge)/( NIR + rededge)
    'NDRE': lambda b: (b[7] - b[4]) / (b[7] + b[4]),
    # CIrededge NIR / rededge - 1
    'CIrededge': lambda b: b[7] / b[4] - 1.0,
    # https://www.mdpi.com/2072-4292/11/24/2925
    # CVI NIR*RED/GREEN ^2
    'CVI': lambda b: b[7] * b[3] / b[2]**2.0,
    # GSAVI (NIR - G) / (NIR + G + L) * (1 + L)
    'GSAVI': lambda b: (b[7] - b[2]) / (b[7] + b[2] + 0.5) * (1.0 + 0.5),
    # MSAVI (2 * NIR + 1 - sqrt((2 * NIR + 1)^2- 8 * (NIR - RED)))/2)
    'MSAVI': lambda b: (2.0 * b[7] + 1.0 - ((2.0 * b[7] + 1.0)**2 - 8.0 * (b[7] - b[3]))**0.5) / 2.0,
    # RDVI (800nm-670nm)/(800nm+670nm)^0.5
    'RDVI': lambda b: (b[7] - b[3]) / (b[7] + b[3])**0.5,
    # SAVI
    # WDRVI (0.1*NIR-RED)/(0.1*NIR+RED)
    'WDRVI': lambda b: (0.1 * b[7] - b[3]) / (0.1 * b[7] + b[3]),

    # K:
    # https://www.researchgate.net/profile/Jingshan-Lu/publication/333251007_Monitoring_leaf_potassium_content_using_hyperspectral_vegetation_indices_in_rice_leaves/links/5ce73e9592851c4eabba2139/Monitoring-leaf-potassium-content-using-hyperspectral-vegetation-indices-in-rice-leaves.pdf?_sg%5B0%5D=started_experiment_milestone&_sg%5B1%5D=started_experiment_milestone&origin=journalDetail&_rtd=e30%3D
    # NDSI ([1600:1700]-[2145:2185])/([1600:1700]+[2145:2185])
    'NDSI': lambda b: (b[10] - b[11]) / (b[10] + b[11]),
    # RSI(1385, 1705) b:1385/b:1705 
    'RSI(1385, 1705)': lambda b: b[9] / b[10],
    # DSI(R1705, R1385) b:1705 - b:1385 
    'DSI(R1705, R1385)': lambda b: b[10] - b[9],
    # https://www.sciencedirect.com/science/article/pii/S0303243421001197
    # RSI(2275, 1875) b:2275/b:1875 -> 1875 unavailable in S-2A
    # https://journals.ashs.org/jashs/view/journals/jashs/132/5/article-p611.xml
    # NDVI(780, 670) = (780nm - 670nm) / (780nm + 670nm)
    'NDVI(780, 670)': lambda b: (b[6] - b[3]) / (b[6] + b[3]),

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


# Paser creation
print('Creating parser...')
parser = tstk.Parser(dataset)
df = parser.create_df(groundtruth_f=['P', 'LC1_Desc'])


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
    name='P_all_f_wheat',
    parser=parser,
    models=sl+ssl,
    y='P',
    cv=tstk.StratifiedKFold(),
    verbose=True
)

results = exp.execute(
    parameter_optimization=True,
    save_results='./results/'
)

print('Calculating metrics...')
df = exp.calculate_metrics(metrics=['acc', 'ball_acc', 'f1', 'prec', 'recall'])
print(df)
df.to_pickle(f'./results/{exp.name}/Df_metrics.pickle')
