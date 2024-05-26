# NCSS DATASET CHANGES REPORT

This report is used to track the changes that were applied to the original NCSS dataset fetched from: <https://ncsslabdatamart.sc.egov.usda.gov/database_download.aspx>

## Targeting-features

This section presents the only features that were extracted and will be used as labels. All others were removed, as they do not correlate to nutrient values.
All nutrient values are converted to $ppm$, according to the formula $$ppm = {{part \over total} \times 10^6}$$

### Convertions

#### Converting gravimetric % to ppm

- *Gravimetric percent on a <2mm base is calculated by dividing the weight of the specific substance under 2mm by the total weight of the sample under 2mm. Then multiplying by 100. The result is expressed as a percentage.*
Since the value corresponds to a relation, they can be converted to $ppm$.
$$part(g) = {Gravimetric(\%) \times 10^{-2} \times total(g)}$$
Therefore, merging the two formulas gives:
$$ppm = {Gravimetric(\%) \times 10^{4}}$$

#### Converting mg/kg to ppm

- *1 mg/kg is equivalent to 1 ppm because both units represent one part in one million parts. So, to convert milligrams per kilogram (mg/kg) to parts per million (ppm), there is no need to perform any mathematical calculations*
$$ppm = mg/kg$$

#### Converting meq/100g to ppm

- *Milliequivalents per 100 grams (meq/100g) is a measure used in chemistry to express the concentration of ions in a substance. It represents the number of milliequivalents of an ion (usually an ion with a charge of +1) present in 100 grams of a substance.*
- According to the follwoing website: <https://biostim.com.au/news/converter/>
$$ppm = {meq/100g \times Molar \times 10}$$

### Carbon (C)

- *total_carbon_ncs* (gravimetric % on a <2 mm base) -> $ppm = {Gravimetric(\%) \times 10^{4}}$

### Nitrogen (N)

- *total_nitrogen_ncs* (gravimetric % on a <2 mm base) -> $ppm = {Gravimetric(\%) \times 10^{4}}$

### Phosporus (P)

- *phosphorus_bray1* (mg/kg on a <2 mm base) -> $ppm = mg/kg$
- *phosphorus_bray2* (mg/kg on a <2 mm base) -> $ppm = mg/kg$
- *phosphorus_major_element* (mg/kg on a <2 mm base) -> $ppm = mg/kg$
- *phosphorus_trace_element* (mg/kg on a <2 mm base) -> $ppm = mg/kg$

### Potassium (K)

- *k_nh4_ph_7* (meq/100g on a <2 mm base) -> $ppm = {meq/100g \times Molar \times 10}$
- *potassium_major_element* (mg/kg on a <2 mm base) -> $ppm = mg/kg$

### Calcium (Ca)

- *ca_nh4_ph_7* (meq/100g on a <2 mm base) -> $ppm = {meq/100g \times Molar \times 10}$
- *calcium_major_element* (mg/kg on a <2 mm base) -> $ppm = mg/kg$

### Magnesium (Mg)

- *mg_nh4_ph_7* (meq/100g on a <2 mm base) -> $ppm = {meq/100g \times Molar \times 10}$
- *magnesium_major_element* (mg/kg on a <2 mm base) -> $ppm = mg/kg$

### Sulfur (S)

- *total_sulfur_ncs* (gravimetric % on a <2 mm base) -> $ppm = {Gravimetric(\%) \times 10^{4}}$

### Copper (Cu)

- *copper_trace_element* (mg/kg on a <2 mm base) -> $ppm = mg/kg$

### Iron (Fe)

- *fe_ammoniumoxalate_extractable* (gravimetric % on a <2 mm base) -> $ppm = {Gravimetric(\%) \times 10^{4}}$
- *iron_sodium_pyro_phosphate* (gravimetric % on a <2 mm base) -> $ppm = {Gravimetric(\%) \times 10^{4}}$
- *iron_major_element* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$

### Manganese (Mn)

- *manganese_ammonium_oxalate* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$
- *manganese_dithionite_citrate* (gravimetric % on a <2 mm base) -> $ppm = {Gravimetric(\%) \times 10^{4}}$
- *manganese_kcl_extractable* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$
- *manganese_major_element* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$
- *manganese_trace_element* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$

### Molybdenum (Mo)

- *molybdenum_trace_element* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$

### Zinc (Zn)

- *zinc_trace_element* (mg/kg on a < 2mm fraction) -> $ppm = mg/kg$

## Creating tables

All the merges were stricly followed according to the following image, which showed *Primary keys* and *Foreign keys* for each table correspondence.
![plot](./NCSS_ER_Diagram.jpg)

## Associating *ids* with *location*

The latitude and longitude of each sample was stored as *latitude_std_decimal_degrees* and *longitude_std_decimal_degrees*, both stored in table *site_key*.
Therefore, a table *ids_loc* was created by merging tables *lab_layer* with *lab_site* through *site_key* column values. Thus, *ids_loc* contained the geographical location for each analysis properly identified.
**This resulted in 417.652 of datapoints**

## Connecting *ids_loc* with *features of interest*

All the properties mentioned in section [Targeting features](#targeting-features) are speaded throught tables *lab_chemical_properties*, *lab_physical_properties* and *lab_major_and_trace_elements_and_oxides*.
Therefore, these three tables were merged with *ids_loc* through (*labsampnum*, *layer_key*), thus creating a table *ids_prop* that contained the a geographical location for each id analysis, plus their respective nutrient values.
**This resulted in 32.586 of datapoints**

## Linking *ids_loc_prop* with *time*

The date of extraction, found in the *observation_date* column, represents the moment of time when the soil was sampled. This attribute is found in table *lab_pedon*, and so it was combined with *ids_loc_prop* by merging both in (*pedon_key*, *site_key*) column values.
**This resulted in 32.583 of datapoints**.

## Dataset

In this final section, we describe how the dataset was manipulated and which transformations took effect.
The code for all of them can be found in the [python-script](./main.py)

### Date of extraction

- The minimum and maximum value found for *observation_date* was 14.763 and 44.474, respectively.
- From the [NCSS](<https://ncsslabdatamart.sc.egov.usda.gov/querypage.aspx>), where we can insert the *pedlabsampnum* and get the observation date in MM-DD-YYYY values, we can create the mapping:
  - 36348 = Jul 7 1999;
  - 24016 = Oct 1 1965;
  - 20349 = Sep 17 1955;
  - 41240 = Nov 27 2012;
- Therefore we can attempt to establish a mapping from *observation_date* to a *date-time* value.
- These examples might imply that the dates are stored as an the number of days that have passed since 30/12/1899.
- Given 4 random numbers between the minimum and maximum values, after applying the mapping we can report that:
  - 19955 = Aug 19 1954 **is Correct**
  - 30498 = Jul 1 1983 **misses by 1 day**
  - 39337 = Sep 12 2007 **Correct**
  - 41044 = May 15 2012 **Correct**

**Therefore we assume the hypothesis above.**

### Missing values

- **C**: 54.45
- **N**: 41.58
- **S**: 64.62
- **Cu**: 71.07
- **Cl**: 77.08
- **Mo**: 77.16
- **Z**: 73.15
- **pH**: 0.63
- **P**: 58.54
- **K**: 0.39
- **Ca**: 0.38
- **Mg**: 0.37
- **Fe**: 43.32
- **Mn**: 22.69
- **Cec**: 1.09
