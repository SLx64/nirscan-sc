from typing import Optional, Union, List

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.axes import Axes

import seaborn as sns

from .spectrum import NIRSpectrum


def _minmax(arr):
    return np.min(arr), np.max(arr)


def plot_spectrum(spectra: Union[NIRSpectrum, List[NIRSpectrum]], wavenumbers: bool = False,
                  reflectance: bool = False, ax: Optional[Axes] = None, title="",
                  **kwargs):
    if isinstance(spectra, NIRSpectrum):
        spectra = [spectra]

    if ax is None:
        ax = plt.gca()

    xlabel = "Wavenumbers [cm-1]" if wavenumbers else "Wavelength [nm]"
    ylabel = "Reflectance" if reflectance else "Absorbance"

    boundaries = []
    for spectrum in spectra:
        if wavenumbers:
            xdata = spectrum.wavelengths.as_wavenumbers()
        else:
            xdata = spectrum.wavelengths.data

        if reflectance:
            ydata = spectrum.absorbance.as_reflectance()
        else:
            ydata = spectrum.absorbance.data

        ax.plot(xdata, ydata, **kwargs)

        boundaries.append([*_minmax(xdata), *_minmax(ydata)])
    boundaries = np.array(boundaries)

    xmin, xmax = boundaries[:, 0].min(), boundaries[:, 1].max()
    ymin, ymax = boundaries[:, 2].min(), boundaries[:, 3].max()

    yf = (ymax-ymin)*0.1

    ax.axis([xmin, xmax, ymin, ymax])
    ax.set_title(title)
    ax.set(xticks=np.linspace(xmin, xmax, 10, dtype=int),
           yticks=np.linspace(ymin-yf, ymax+yf, 10, dtype=float),
           xlabel=xlabel, ylabel=ylabel)


def pca_pairplot(df: pd.DataFrame, hue="Label", **kwargs):
    return sns.pairplot(df, hue=hue, **kwargs)
