from typing import List, Union
from copy import deepcopy

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

from .spectrum import NIRSpectrum, Absorbance, DimensionError


def _snv(data: np.ndarray, norm: bool = False) -> np.array:
    data_mean = 0 if norm else np.mean(data)
    return (data - data_mean) / np.std(data)


def snv(spectrum: NIRSpectrum, norm: bool = False) -> NIRSpectrum:
    wavelength = spectrum.wavelengths.copy()
    snv_data = _snv(spectrum.absorbance.data, norm)
    header = deepcopy(spectrum.header)

    return NIRSpectrum(wavelength, Absorbance(snv_data),
                       header=header)


def savgol(spectrum: NIRSpectrum, window: int,
           order: int, deriv: int) -> NIRSpectrum:
    wavelength = spectrum.wavelengths.copy()
    savgol_data = savgol_filter(spectrum.absorbance.data,
                                window_length=window,
                                polyorder=order, deriv=deriv)
    header = deepcopy(spectrum.header)

    return NIRSpectrum(wavelength, Absorbance(savgol_data),
                       header=header)


def _msc(data: np.ndarray, ref: np.ndarray):
    fit = np.polyfit(ref, data, 1)
    return (data - fit[1]) / fit[0]


def msc(spectrum: NIRSpectrum, ref_spectrum: NIRSpectrum) -> NIRSpectrum:
    if not spectrum.wavelengths == ref_spectrum.wavelengths:
        raise DimensionError("Wavelength data should be equal.")
    wavelength = spectrum.wavelengths.copy()
    msc_data = _msc(spectrum.absorbance.data,
                    ref_spectrum.absorbance.data)
    header = deepcopy(spectrum.header)

    return NIRSpectrum(wavelength, Absorbance(msc_data),
                       header=header)


class PCAnalysis(object):
    def __init__(self, spectra: List[NIRSpectrum], ncomp: int = 5):
        self.spectra = spectra
        self._pca = PCA(n_components=ncomp)
        self.transformed = None

    def n_components(self):
        return self._pca.n_components

    def column_names(self):
        columns = [f"PC{n+1}" for n in range(self.n_components())]
        return columns

    def _data_matrix(self) -> np.ndarray:
        return np.array([s.absorbance.data for s in self.spectra])

    def run(self):
        for spectrum in self.spectra:
            if not spectrum.wavelengths == self.spectra[0].wavelengths:
                raise DimensionError("All spectra should have the"
                                     "same wavelength data.")
        self.transformed = self._pca.fit_transform(self._data_matrix()).T

    def transform(self, spectrum: NIRSpectrum, as_pandas: bool = False)\
            -> Union[np.ndarray, pd.Series]:
        data_t = self._pca.transform([spectrum.absorbance.data])[0]
        if as_pandas:
            return pd.Series(data_t, index=self.column_names())
        else:
            return data_t
