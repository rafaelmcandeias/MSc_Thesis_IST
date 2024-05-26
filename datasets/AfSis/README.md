# AfSis

This dataset consists of several nutrients values plotted for the whole continent of Africa in a 30meter resoluiton, excluding water and deserts.
It results of predictions made from a Random Forest model trained with EO data. This band values were exctrated from the satelites Landsat, Sentinel2 and Sentinel1.
Each directory contains a .tif file which contains data of the nutrient for the whole continent. To be extracted, one must follow this pythonjupiter notebook <https://github.com/iSDA-Africa/isdasoil-tutorial/blob/main/iSDAsoil-tutorial.ipynb>

Even though the Dataset includes considerable amounts of quality data, such information becomes useless to our cause, as it is impossible to associate them to the band values.
To clarify, the dataset only contains nutrient values that can be converted to ppm, no band data whatsoever, and no date or time associated to these nutrient values prediction. Therefore, even if we made a dataset with gps location and its corresponding nutrient values, we did not know which band values to look for. The land surface changes with time, and its band values from one year might be considerably different from another year.

It is also important to notice that the Afsis dataset is also discarded, since our investigation is focused on exploring SSL algorithms, which require Xl and Xu. The dataset we would be extracting does not fit to any of those, as it only contains labels, and not features.
