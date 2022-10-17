# NIRScanNano Student Course

This repository contains simple functions to interactively analyze results from the [DLP NIRScan Nano EVM](https://www.ti.com/tool/DLPNIRNANOEVM) by Texas Instruments. It was initially created for a student course to hide any boilerplate code and significantly reduce the programming experience requirements.

## Installation

Installation via a package manager is not yet supported. Instead, copy the library folder `NIRScanNano` and add the directory to your `PYTHONPATH` environment variable.

The depencies can be installed via `pip` or `conda` (Python >= 3.7):

```
pip install -r requirements.txt

conda create -n nirscan --file requirements.txt
```


## Examples

A full example of a complete student course (20 groups) can be found in the `examples` folder.

### Read a spectrum file

Reading a single spectrum returns a `NIRSpetrum` object containing the data and all header information.

```python
from NIRScanNano.spectrum import read_spectrum

spectrum = read_spectrum("caffeine.csv")
```

It is also supported to read a batch of spectra using a data file. This is especially useful for performing a principal component analysis of all measured spectra in the course.

```python
from NIRScanNano.course import DataReader

data = DataReader("datafile.csv")
```

One can specify the column names and delimiters. The default is: `"Group";"Name";"File"` with `Group` as the key value to access the data later, `Name` as the compound name, and `File` is the path to the file relative to the data file directory.

```python
# returns a list of spectra for the key value (e.g. group id)
spectra = data.spectra_by_group(key_value) 
```

The `DataReader` also provides a method to select a random sample.

```python
spectrum = data.random_sample()
```

The compound name will be saved for each spectrum as an entry in the header dictionary.

```python
spectrum.header["Name"]
```

The list can be filtered using list comprehensions to obtain all spectra for the desired substance.

```python
caffeine_spectra = [s for s in spectra if s.header["Name"] == "Caffeine"]
```

### Preprocessing

We currently support the most common methods for preprocessing NIR spectra.

```python
from NIRScanNano.analysis import snv, savgol, msc
```

#### Normalization

```python
norm_spectrum = snv(spectrum, norm=True)
```

#### Standard Normal Variate (SNV)

```python
snv_spectrum = snv(spectrum)
```

#### Multiplicative Scatter Correction (MSC)

If no reference spectrum is available, it is also possible to average multiple spectra of a compound. Please be aware that no checks are performed.

```python
avg_spectrum = average_spectra(spectra)
msc_spectrum = msc(spectrum, avg_spectrum)
```

#### Savitzky-Golay Filter

The `savgol` function is a simple wrapper around the [SciPy function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html) `savgol_filter` to directly handle a `NIRSpectrum` object. 

```python
# spectrum, window, polynom order, derivative
savgol_spectrum = savgol(spectrum, 11, 2, 0)
```

### Visualization

The library provides a `plot_spectrum` function to visualize a single spectrum or multiple spectra (by passing a list) using [Matplotlib](https://matplotlib.org/).

```python
import matplotlib.pyplot as plt
from NIRScanNano.visualization import plot_spectrum

fig, ax = plt.subplots(1)
plot_spectrum(example_spectra, ax=ax)
```

### Principal Component Analysis (PCA)

We have implemented basic functionalities to perform Principal Component Analysis, export everything into a [pandas DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) and visualize it in a [seaborn pairplot](https://seaborn.pydata.org/generated/seaborn.pairplot.html).

```python
from NIRScanNano.course import pca_to_pandas
from NIRScanNano.visualization import pca_pairplot
from NIRScanNano.analysis import PCAnalysis

pca = PCAnalysis(spectra)
pca.run()

pca_df = pca_to_pandas(pca, label="Name")

pca_pairplot(pca_df)
```

The distances to all hypersphere centers of individual substances in transformed space are calculated to identify an unknown compound. The radius is defined as the maximum distance of each corresponding data point plus a quarter of the standard deviation.

```python
from NIRScanNano.course import pca_centroids, nearest_centroids, eval_distances

# column names
columns = [col for col in pca_df.columns if "PC" in col]

# centroids of all substances and maximum distances 
centroids, max_dist = pca_centroids(pca_df, columns)

# distances of the unknown substance to all centroids
distances = nearest_centroids(test_spectrum, pca, centroids, 10)

# multiple compounds are returned if there is an overlap
# of the hyperspheres and none if the spectrum is not
# within any hypersphere
possible_compounds = eval_distances(distances, max_dist)
```

## Cite this work

We are currently preparing the manuscript.



