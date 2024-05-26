# BigEarthNet

The BigEarthNet archive was constructed by the Remote Sensing Image Analysis (RSiM) Group and the Database Systems and Information Management (DIMA) Group at the Technische Universität Berlin (TU Berlin). This work is supported by the European Research Council under the ERC Starting Grant BigEarth and by the Berlin Institute for the Foundations of Learning and Data (BIFOLD). Before BIFOLD, the Berlin Big Data Center (BBDC) supported the work.

BigEarthNet is a benchmark archive, consisting of 590,326 pairs of Sentinel-1 and Sentinel-2 image patches. The first version (v1.0-beta) of BigEarthNet includes only Sentinel 2 images. Recently, it has been enriched by Sentinel-1 images to create a multi-modal BigEarthNet benchmark archive (called also as BigEarthNet-MM).

More information in: <https://bigearth.net/#faq>

## BigEarthNet-S1

Check the pdf [Description_S1](Description_S1.pdf)

## BigEarthNet-S2

Check the pdf [Description_S2](Description_S2.pdf)

The original dataset BigEarthNet-S2 is made of several directories, each corresponding to a patch. A patch can be portraid as a picture taken from the satelite.
Each patch, named as \<sentinel-id>_MSIL2A_\<YYYYMMDD>T\<HHMMSS>_\<h-order>_\<v-order>, has 13 tiff files, each representing the value of one of the 13 bands of the satelite.

281GB of band values from S2 satelite.
164.274 patches/files. Each with 13 band values => 2.135.562 band values

The dataset was extracted with a [shell script](S2/run.sh), and created X rows of data with:

- **patch** are extracted from the directory name
- **date** are extracted from the directory name
- **time** are extracted from the directory name
- **B01** shape originaly is (20, 20)
- **B02** shape originaly is (120, 120)
- **B03** shape originaly is (120, 120)
- **B04** shape originaly is (120, 120)
- **B05** shape originaly is (60, 60)
- **B06** shape originaly is (60, 60)
- **B07** shape originaly is (60, 60)
- **B08** shape originaly is (120, 120)
- **B8A** shape originaly is (60, 60)
- **B09** shape originaly is (20, 20)
- **B11** shape originaly is (60, 60)
- **B12** shape originaly is (60, 60)
