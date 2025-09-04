from typing import Optional, Tuple
import numpy as np

from mosqito.sq_metrics.loudness.loudness_zwst.loudness_zwst_freq import loudness_zwst_freq
from mosqito.sq_metrics.sharpness.sharpness_din.sharpness_din_from_loudness import sharpness_din_from_loudness
from mosqito.sq_metrics.roughness.roughness_dw.roughness_dw_freq import roughness_dw_freq

class PsychoacousticBackend:
    """
    Adapter interface to compute psychoacoustic metrics from 1/3-octave SPL time series.

    You should implement the methods below to call your preferred library:
    - loudness_time(SPL_bands: (T, n_bands), freqs_hz: (n_bands,), dt: float) -> (T,)
    - sharpness(SPL_bands: (T, n_bands), freqs_hz: (n_bands,), L_time: (T,)) -> float
    - fluctuation_strength(SPL_bands: (T, n_bands), freqs_hz: (n_bands,), dt: float) -> float
    - roughness(SPL_bands: (T, n_bands), freqs_hz: (n_bands,), dt: float) -> float

    All returned metrics are scalar averages over time (except L_time which is time series).
    Loudness in sone, Sharpness in acum, Fluctuation Strength in vacil, Roughness in asper.
    """

    def loudness_time(self, SPL_bands, freqs_hz, dt):
        raise NotImplementedError(
            "Implement loudness_time() using your psychoacoustic library (e.g., ISO 532-1 Zwicker)."
        )

    def sharpness(self, SPL_bands, freqs_hz, L_time):
        raise NotImplementedError(
            "Implement sharpness() (e.g., DIN 45692) returning a time-averaged or loudness-weighted value."
        )

    def fluctuation_strength(self, SPL_bands, freqs_hz, dt):
        raise NotImplementedError(
            "Implement fluctuation_strength() (e.g., Aures/Daniel & Weber), return scalar time-averaged FS."
        )

    def roughness(self, SPL_bands, freqs_hz, dt):
        raise NotImplementedError(
            "Implement roughness() (e.g., Daniel & Weber), return scalar time-averaged roughness."
        )
    
class PsychoacousticBackendAdapter(PsychoacousticBackend):
    """
    MoSQITo-backed implementation of the psychoacoustic backend.

    Input to all methods:
      - SPL_bands: ndarray, shape (T, n_bands), in dB SPL, 1/3-octave band *levels* over time
      - freqs_hz:  ndarray, shape (n_bands,), 1/3-octave *center* frequencies [Hz]
      - dt:        float, nominal frame spacing [s] (not strictly needed here)

    Internally, SPL bands are expanded to a uniform fine spectrum (24..24000 Hz)
    so we can call:
      - loudness_zwst_freq (ISO 532-1 time-segment loudness from spectrum)
      - sharpness_din_from_loudness (DIN 45692 sharpness from loudness)
      - roughness_dw_freq (Daniel & Weber roughness from spectrum)

    Fluctuation strength is not implemented in MoSQITo => returns 0.0.
    """

    def __init__(
        self,
        field_type: str = "free",
        sharpness_weighting: str = "din",
        fine_df_hz: float = 10.0,
        fmin_hz: float = 24.0,
        fmax_hz: float = 24000.0,
    ):
        self.field_type = field_type
        self.sharpness_weighting = sharpness_weighting
        self.fmin_hz = float(fmin_hz)
        self.fmax_hz = float(fmax_hz)
        # ensure integer number of bins covering [fmin, fmax]
        n_bins = int(np.floor((self.fmax_hz - self.fmin_hz) / float(fine_df_hz))) + 1
        self.fine_freqs = self.fmin_hz + np.arange(n_bins) * float(fine_df_hz)
        self._twoe5 = 2e-5  # Pa reference for SPL
        # Use arange for interior bins and then force endpoints explicitly.
        df = float(fine_df_hz)
        if df <= 0:
            raise ValueError("fine_df_hz must be > 0")
        # Interior bins (strictly between endpoints)
        interior = np.arange(self.fmin_hz + df, self.fmax_hz, df)
        # Concatenate explicit endpoints to ensure [24.0, ..., 24000.0]
        self.fine_freqs = np.concatenate(([self.fmin_hz], interior, [self.fmax_hz])).astype(float)

    # ---------- Public API expected by Simulation.compute_psychoacoustics_map ----------

    def loudness_time(self, SPL_bands: np.ndarray, freqs_hz: np.ndarray, dt: Optional[float]) -> np.ndarray:
        try:
            spec, freqs = self._expand_bands_to_fine_spectrum(SPL_bands, freqs_hz)
            N, N_spec, _bark = loudness_zwst_freq(spec, freqs, field_type=self.field_type)  # (T,), (Nbark,T)
            # cache for sharpness computation (avoid recomputing)
            self._cached_loudness = (np.asarray(N, dtype=float), np.asarray(N_spec, dtype=float))
            return np.asarray(N, dtype=float)
        except Exception as e:
            print(f"Error computing loudness: {e}")
            return np.zeros(SPL_bands.shape[0], dtype=float)

    def sharpness(self, SPL_bands: np.ndarray, freqs_hz: np.ndarray, L_time: np.ndarray) -> float:
        # Prefer using cached N, N_specific if available (same call context)
        try:
            if hasattr(self, "_cached_loudness"):
                N, N_spec = self._cached_loudness
            else:
                # Fallback: compute loudness from bands
                spec, freqs = self._expand_bands_to_fine_spectrum(SPL_bands, freqs_hz)
                N, N_spec, _ = loudness_zwst_freq(spec, freqs, field_type=self.field_type)

            S_time = sharpness_din_from_loudness(
                N=np.asarray(N, dtype=float),
                N_specific=np.asarray(N_spec, dtype=float),
                weighting=self.sharpness_weighting,
            )
            # Return a scalar (time-average); swap to loudness-weighted if you prefer
            return float(np.mean(np.asarray(S_time, dtype=float)))
        except Exception as e:
            print(f"Error computing sharpness: {e}")
            return 0.0

    def fluctuation_strength(self, SPL_bands: np.ndarray, freqs_hz: np.ndarray, dt: Optional[float]) -> float:
        # Not available in MoSQITo yet
        return 0.0

    def roughness(self, SPL_bands: np.ndarray, freqs_hz: np.ndarray, dt: Optional[float]) -> float:
        # Use MoSQITo's Daniel & Weber implementation from spectrum
        try:
            spec, freqs = self._expand_bands_to_fine_spectrum(SPL_bands, freqs_hz)
            R, _Rspec, _bark = roughness_dw_freq(spec, freqs)  # R is (T,) or scalar
            R = np.asarray(R, dtype=float)
            return float(np.mean(R)) if R.ndim > 0 else float(R)
        except Exception as e:
            print(f"Error computing roughness: {e}")
            return 0.0

    # ------------------------------ Helpers ------------------------------

    def _expand_bands_to_fine_spectrum(
        self, SPL_bands: np.ndarray, band_centers_hz: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert 1/3-octave SPL bands over time to a uniform fine-band amplitude spectrum.

        Energy-preserving expansion:
          - For each band at time t, the band RMS pressure p_band_rms is spread uniformly
            over the fine bins whose frequencies lie inside the band edges.
          - If a band contains K fine bins, each bin gets amplitude = p_band_rms / sqrt(K).

        Returns
        -------
        spec : np.ndarray
            Fine-band amplitude spectrum in Pa, shape (n_fine_bins, T)
        freqs : np.ndarray
            Fine frequency axis in Hz, shape (n_fine_bins,)
        """
        SPL_bands = np.asarray(SPL_bands, dtype=float)        # (T, n_bands)
        band_centers_hz = np.asarray(band_centers_hz, dtype=float)
        assert SPL_bands.ndim == 2, "SPL_bands must be (T, n_bands)"
        T, n_bands = SPL_bands.shape
        assert band_centers_hz.shape == (n_bands,), "freqs_hz must have shape (n_bands,)"

        # 1/3-octave band edges
        band_lo = band_centers_hz / (2.0 ** (1.0 / 6.0))
        band_hi = band_centers_hz * (2.0 ** (1.0 / 6.0))

        # Map each fine bin to a band index (-1 if none)
        f = self.fine_freqs
        idx_per_bin = np.full(f.shape, -1, dtype=int)
        for b in range(n_bands):
            mask = (f >= band_lo[b]) & (f < band_hi[b])
            idx_per_bin[mask] = b

        # Precompute bin counts per band
        counts = np.zeros(n_bands, dtype=int)
        for b in range(n_bands):
            counts[b] = int(np.sum(idx_per_bin == b))
        counts = np.maximum(counts, 1)  # avoid divide-by-zero for out-of-range bands

        # Allocate spectrum
        spec = np.zeros((f.size, T), dtype=float)

        # dB SPL band level -> RMS pressure [Pa] per band
        # Lp = 20*log10(p_rms / p0) => p_rms = p0 * 10^(Lp/20)
        p_rms_bands = self._twoe5 * (10.0 ** (SPL_bands / 20.0))  # (T, n_bands)

        # Distribute energy uniformly across the band's fine bins
        for b in range(n_bands):
            sel = (idx_per_bin == b)
            K = counts[b]
            if K == 0:
                continue
            # Each bin amplitude so that K * (p_bin^2) == p_band_rms^2
            # => p_bin = p_band_rms / sqrt(K)
            p_bin = p_rms_bands[:, b] / np.sqrt(K)  # (T,)
            spec[sel, :] = p_bin[None, :]

        return spec, f