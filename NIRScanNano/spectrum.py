from typing import Optional, Sequence
from collections.abc import Mapping
from io import StringIO

import numpy as np


class DimensionError(Exception):
    pass


class DataContainer(object):
    def __init__(self, data: np.ndarray):
        self.data = data

    def min(self):
        return np.min(self.data)

    def max(self):
        return np.max(self.data)

    def __len__(self):
        return len(self.data)

    def __eq__(self, other):
        return np.allclose(self.data, other.data)

    def copy(self):
        return type(self)(self.data.copy())


class Wavelength(DataContainer):
    def __init__(self, data: np.ndarray):
        super().__init__(data)

    def as_wavenumbers(self):
        return 10**7 / self.data


class Absorbance(DataContainer):
    def __init__(self, data: np.ndarray):
        super().__init__(data)

    def as_reflectance(self):
        return 10**(-1 * self.data)

    @staticmethod
    def zero_from_wavelength(wavelength: Wavelength):
        return Absorbance(np.zeros_like(wavelength.data))


class Signal(DataContainer):
    def __init__(self, data: np.ndarray):
        super().__init__(data)


class NIRSpectrumHeader(Mapping):
    def __init__(self, *args, **kwargs):
        self._storage = dict(*args, **kwargs)

    def __setitem__(self, key, value):
        self._storage[key] = value

    def __getitem__(self, key):
        return self._storage[key]

    def __len__(self):
        return len(self._storage)

    def __iter__(self):
        return iter(self._storage)


class NIRSpectrum(object):
    def __init__(self, wavelength: Wavelength, absorbance: Absorbance,
                 reference_signal: Optional[Signal] = None,
                 sample_signal: Optional[Signal] = None,
                 header: Optional[NIRSpectrumHeader] = None):
        if len(wavelength) != len(absorbance):
            raise DimensionError("Dimension mismatch between"
                                 "wavelength and absorbance data!")

        self.header = header
        self._wavelength = wavelength
        self._absorbance = absorbance
        self._reference_signal = reference_signal
        self._sample_signal = sample_signal

    @property
    def wavelengths(self) -> Wavelength:
        return self._wavelength

    @property
    def absorbance(self) -> Absorbance:
        return self._absorbance

    @property
    def sample_signal(self) -> Signal:
        return self._sample_signal

    @property
    def reference_signal(self) -> Signal:
        return self._reference_signal

    def __repr__(self):
        return "NIRSpectrum"

    def __len__(self):
        return len(self._wavelength)


def read_spectrum(filename: str, nhead: int = 19,
                  delimiter: str = ",") -> NIRSpectrum:
    with open(filename, "r") as spec_file:
        header = NIRSpectrumHeader()
        for _ in range(nhead):
            line = next(spec_file)
            key, *values = line.split(delimiter)

            key = key.strip().replace(":", "")
            values = [v for v in values if v.strip()]

            # coefficients are saved as float array
            if key.startswith("Pixel") or key.startswith("Shift"):
                values = [float(v) for v in values]
            else:
                values = values[0]
                if values.isdigit():
                    values = int(values)

            header[key] = values

        _ = next(spec_file)  # skip column headers
        lines = spec_file.readlines()

        data_io = StringIO(" ".join([d.replace(",", " ") for d in lines]))
        data = np.genfromtxt(data_io)

        spectrum = NIRSpectrum(Wavelength(data[:, 0]), Absorbance(data[:, 1]),
                               Signal(data[:, 2]), Signal(data[:, 3]), header)

        return spectrum


def average_spectra(spectra: Sequence[NIRSpectrum]) -> NIRSpectrum:
    if not all(s.wavelengths == spectra[0].wavelengths for s in spectra):
        raise DimensionError("All spectra should have the"
                             "same wavelength data.")

    wavelength = spectra[0].wavelengths.copy()
    mean = np.mean([s.absorbance.data for s in spectra], axis=0)

    return NIRSpectrum(wavelength, Absorbance(mean))
