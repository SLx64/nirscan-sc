# This file contains some useful utility functions
# to hide boilerplate code from students

from typing import Optional, List, Tuple, TypeVar
from pathlib import Path

import pandas as pd
import numpy as np

from .spectrum import NIRSpectrum, read_spectrum
from .analysis import PCAnalysis

DISTANCE_BUFFER = 0.25


T = TypeVar('T')

def _copy_name(name: str, spectrum: NIRSpectrum):
    spectrum.header["Name"] = name


class DataReader(object):
    def __init__(self, datafile: str, delimiter: str = ";",
                 read_spectra: bool = True, copy_name: bool = True,
                 name_col: str = "Name", file_col: str = "File"):
        self.data = pd.read_csv(datafile, delimiter=delimiter)

        if read_spectra:
            self.path = Path(datafile).parent.resolve()
            self.data[file_col] = self.data[file_col].apply(lambda f: str(self.path / f))
            self.data['Spectrum'] = self.data[file_col].apply(read_spectrum)

        if copy_name:
            self.data.apply(lambda x: _copy_name(x[name_col], x["Spectrum"]), axis=1)

    def spectra(self):
        return self.data["Spectrum"].tolist()

    def spectra_by_group(self, key: T, group_col: str = "Group"):
        spectra = self.data[self.data[group_col] == key]["Spectrum"]
        return spectra.tolist()

    def random_sample(self):
        return self.data.sample()["Spectrum"].values[0]


def pca_to_pandas(pca: PCAnalysis, label: Optional[str] = None) -> pd.DataFrame:
    columns = pca.column_names()
    columns.append("Label")

    column_data = []
    for i, spectrum in enumerate(pca.spectra):
        row_data = [pca.transformed[n, i]
                    for n in range(pca.n_components())]
        row_data.append(spectrum.header.get(label, ""))

        column_data.append(row_data)

    pca_df = pd.DataFrame(column_data, columns=columns)
    return pca_df


def _group_centroid_distance(df: pd.DataFrame) -> np.float64:
    dist = df.sub(df.mean()).pow(2).sum(axis=1).pow(1/2)
    dist_std = dist.std()
    return dist.max() + dist_std * DISTANCE_BUFFER


def _centroid_distance(df: pd.DataFrame, s: pd.Series) -> pd.Series:
    return df.sub(s).pow(2).sum(axis=1).pow(1/2)


def pca_centroids(df: pd.DataFrame, data_columms: List[str]) \
        -> Tuple[pd.DataFrame, pd.Series]:
    grouped_data = df.groupby('Label')[data_columms]

    centroids = grouped_data.mean()
    centroids_dist = grouped_data.apply(_group_centroid_distance)

    return centroids, centroids_dist


def nearest_centroids(spectrum: NIRSpectrum, pca: PCAnalysis,
                      centroids: pd.DataFrame, n_centroids: int = 5)\
        -> pd.Series:
    spectrum_t = pca.transform(spectrum, as_pandas=True)
    distances = _centroid_distance(centroids, spectrum_t)

    return distances.sort_values()[:n_centroids]


def eval_distances(d: pd.Series, d_max: pd.Series) -> List:
    cond = d.where(d.sub(d_max) < 0).notna()
    return d[cond].index.tolist()
