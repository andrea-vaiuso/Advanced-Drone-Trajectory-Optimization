import numpy as np

class RotorSoundModel:
    """
    Angle- and RPM-dependent acoustic emission model for a multi-rotor UAV.

    The model maps a radiation angle and rotor speeds to per-band Sound Power
    Level (SWL) at the source and per-band Sound Pressure Level (SPL) at a
    receiver. Emission spectra are provided by an angle-indexed lookup table
    (LUT). Per-band propagation applies spherical spreading with optional
    atmospheric absorption and directivity corrections.
    """

    def __init__(
        self,
        rpm_reference: float = 2500.0,
        lookup_table_filename: str = "lookup_noise_model.npy",
        rpm_exponent: float = 3.0,
    ):
        """
        Initialize the model from an angle-indexed SWL lookup table.

        Parameters
        ----------
        rpm_reference : float, optional
            Reference RPM for which the LUT spectra are defined. Default is 2500.
        lookup_table_filename : str, optional
            Path to a NumPy ``.npy`` file containing per-angle SWL spectra.
            Expected shape: ``(n_angles, n_bands)`` with values in dB re 1 pW.
        rpm_exponent : float, optional
            Exponent for RPM scaling of SWL:
            ``SWL(rpm) = SWL_ref + 10 * rpm_exponent * log10(rpm / rpm_reference)``.
            Default is 3.0.
        """
        self.noise_data = np.load(lookup_table_filename, allow_pickle=True)
        data = np.asarray(self.noise_data)
        if data.ndim == 3 and data.shape[1] == 1:
            self.noise_data = data[:, 0, :]
        else:
            self.noise_data = data
        self.noise_data = np.asarray(self.noise_data, dtype=float)
        if self.noise_data.ndim != 2:
            raise ValueError("lookup_table must be a 2D array of shape (n_angles, n_bands).")
        self.rpm_reference = float(rpm_reference)
        self.rpm_exponent = float(rpm_exponent)

    def get_noise_emissions(self, zeta_angle: float, rpms: list, distance: float, **kwargs) -> tuple:
        """
        Compute per-band SPL at a receiver and per-band SWL at the source.

        Steps:
        1) Retrieve total-drone SWL spectrum from LUT at the given angle.
        2) Convert to single-rotor reference (subtract 10*log10(4) dB).
        3) Scale the single-rotor spectrum for each rotor RPM.
        4) Sum rotor contributions in power per band → total SWL per band.
        5) Apply per-band propagation to obtain SPL at the receiver.

        Propagation (per band)
        ----------------------
        ``Lp = Lw - 10*log10(4*pi*d^2) - alpha(f)*d + DI(f)``
        where:
          - ``Lw``: per-band SWL at source [dB re 1 pW],
          - ``d`` : source–receiver distance [m],
          - ``alpha(f)``: atmospheric absorption [dB/m],
          - ``DI(f)``: directivity index [dB].

        Parameters
        ----------
        zeta_angle : float
            Radiation angle in radians. Converted to degrees and clamped to the
            LUT angle index range.
        rpms : list
            Rotor speeds as a list ``[rpm1, rpm2, ...]``.
        distance : float
            Source–receiver distance in meters.
        **kwargs :
            Optional keyword-only inputs:
              - ``alpha_per_band`` (np.ndarray, shape (n_bands,)): atmospheric absorption in dB/m.
              - ``di_per_band``    (np.ndarray, shape (n_bands,)): directivity index in dB.
              - ``temperature_c``      (float): ambient temperature in °C.
              - ``relative_humidity``  (float): RH in %.
              - ``pressure_kpa``       (float): static pressure in kPa.
            If ``alpha_per_band`` is not provided but weather parameters are,
            the model computes ``alpha_per_band`` internally via ISO 9613-1.

        Returns
        -------
        tuple of (np.ndarray, np.ndarray)
            ``(SPL_per_band, SWL_per_band)`` in dB, both with shape ``(n_bands,)``.
        """
        angle_idx = self._angle_to_index(zeta_angle)
        total_drone_swl = self._get_lut_row(angle_idx)                       # (n_bands,)
        single_rotor_ref = self._to_single_rotor_reference(total_drone_swl)  # (n_bands,)

        swl_total_per_band = self._total_swl_contribution(
            swl_ref_rpm=single_rotor_ref,
            rpms=rpms,
            rpm_reference=self.rpm_reference,
            rpm_exponent=self.rpm_exponent,
        )  # (n_bands,)

        # Resolve alpha_per_band: prefer explicit input, else compute from weather
        alpha_per_band = kwargs.get("alpha_per_band", None)
        if alpha_per_band is None:
            T = kwargs.get("temperature_c", None)
            RH = kwargs.get("relative_humidity", None)
            P = kwargs.get("pressure_kpa", None)
            if (T is not None) and (RH is not None) and (P is not None):
                alpha_per_band = self.get_alpha_per_band(
                    temperature_c=float(T), relative_humidity=float(RH), pressure_kpa=float(P), preferred=True
                )

        spl_per_band = self._propagate_per_band(
            swl_total_per_band,
            distance,
            alpha_per_band=alpha_per_band,
            di_per_band=kwargs.get("di_per_band", None),
        )
        return spl_per_band, swl_total_per_band

    def get_alpha_per_band(
        self,
        temperature_c: float = 20.0,
        relative_humidity: float = 50.0,
        pressure_kpa: float = 101.325,
        preferred: bool = True,
    ) -> np.ndarray:
        """
        Compute atmospheric absorption alpha(f) per 1/3-octave band in dB/m.

        Parameters
        ----------
        temperature_c : float, optional
            Ambient temperature in °C. Default is 20.0.
        relative_humidity : float, optional
            Relative humidity in %. Default is 50.0.
        pressure_kpa : float, optional
            Static pressure in kPa. Default is 101.325 (≈ 1 atm).
        preferred : bool, optional
            If True, use preferred (ISO/ANSI) 1/3-octave center frequencies.
            Otherwise use exact geometric progression. Default is True.

        Returns
        -------
        np.ndarray
            Per-band absorption coefficients in dB/m; shape ``(n_bands,)``.
        """
        freqs = self.third_octave_center_frequencies(preferred=preferred)
        return self._atmospheric_absorption_iso9613(
            f_hz=freqs,
            temperature_c=temperature_c,
            relative_humidity=relative_humidity,
            pressure_kpa=pressure_kpa,
        )

    def third_octave_center_frequencies(self, preferred: bool = True) -> np.ndarray:
        """
        Return 1/3-octave center frequencies matching the LUT band count.

        Parameters
        ----------
        preferred : bool, optional
            If True, returns preferred ISO/ANSI normalized centers (rounded).
            Otherwise returns exact geometric centers around 1000 Hz.

        Returns
        -------
        np.ndarray
            Center frequencies in Hz; shape ``(n_bands,)``.
        """
        n_bands = self.noise_data.shape[1]
        f0 = 1000.0
        half = (n_bands - 1) // 2
        k = np.arange(-half, half + 1)
        freqs_exact = f0 * (2.0 ** (k / 3.0))
        if not preferred:
            return freqs_exact

        preferred_list = np.array([
            20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160,
            200, 250, 315, 400, 500, 630, 800, 1000, 1250,
            1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
            10000, 12500, 16000, 20000
        ], dtype=float)

        # If LUT has more/less bands than 31, map by nearest
        return np.array([preferred_list[np.abs(preferred_list - fe).argmin()] for fe in freqs_exact], dtype=float)

    @staticmethod
    def _total_swl_contribution(
        swl_ref_rpm: np.ndarray,
        rpms: list,
        rpm_reference: float,
        rpm_exponent: float = 1.0,
    ) -> np.ndarray:
        """
        Compute per-band total SWL from a single-rotor reference spectrum and multiple rotor RPMs.

        Parameters
        ----------
        swl_ref_rpm : np.ndarray
            Single-rotor reference SWL spectrum at ``rpm_reference``; shape ``(n_bands,)`` in dB re 1 pW.
        rpms : list
            Rotor speeds as a list ``[rpm1, rpm2, ...]``.
        rpm_reference : float
            Reference RPM associated with ``swl_ref_rpm``.
        rpm_exponent : float, optional
            Exponent for RPM scaling of SWL. See class docstring. Default is 1.0.

        Returns
        -------
        np.ndarray
            Total SWL per band in dB, shape ``(n_bands,)``.
        """
        bands = np.asarray(swl_ref_rpm, dtype=float)
        if bands.ndim != 1:
            raise ValueError("swl_ref_rpm must be 1D with shape (n_bands,)")

        rotor_rpms = np.asarray(rpms, dtype=float)
        safe_rpms = np.maximum(rotor_rpms, 1e-9)
        rpm_scale_db = 10.0 * float(rpm_exponent) * np.log10(safe_rpms / float(rpm_reference))  # (n_rotors,)

        swl_rotors = bands[:, None] + rpm_scale_db[None, :]     # (n_bands, n_rotors)
        swl_lin = 10.0 ** (swl_rotors / 10.0)
        swl_total_lin = np.sum(swl_lin, axis=1)                 # (n_bands,)
        return 10.0 * np.log10(swl_total_lin + 1e-300)

    @staticmethod
    def band_power_sum(levels_db: np.ndarray) -> float:
        """
        Compute broadband level (dB) by power-summing per-band levels.

        Parameters
        ----------
        levels_db : np.ndarray
            Per-band levels in dB; shape ``(n_bands,)``.

        Returns
        -------
        float
            Broadband level in dB.
        """
        x = np.asarray(levels_db, dtype=float)
        return float(10.0 * np.log10(np.sum(10.0 ** (x / 10.0)) + 1e-300))

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _angle_to_index(self, zeta_angle_rad: float) -> int:
        """
        Map a radiation angle in radians to the LUT row index.

        Angles are converted to degrees and clamped to ``[0, n_angles-1]``.

        Parameters
        ----------
        zeta_angle_rad : float
            Radiation angle in radians.

        Returns
        -------
        int
            Index into the LUT angle dimension.
        """
        n_angles = self.noise_data.shape[0]
        angle_deg = np.degrees(zeta_angle_rad)
        return int(np.clip(np.round(angle_deg), 0, n_angles - 1))

    def _get_lut_row(self, angle_idx: int) -> np.ndarray:
        """
        Retrieve the per-band total-drone SWL spectrum for a given angle index.

        Parameters
        ----------
        angle_idx : int
            Row index into the LUT.

        Returns
        -------
        np.ndarray
            Per-band total-drone SWL in dB, shape ``(n_bands,)``.
        """
        row = self.noise_data[angle_idx]
        if row.ndim != 1:
            raise ValueError("LUT row must be 1D per-band spectrum.")
        return row.astype(float)

    @staticmethod
    def _to_single_rotor_reference(total_drone_swl: np.ndarray) -> np.ndarray:
        """
        Convert a total-drone per-band SWL spectrum to a single-rotor reference.

        Subtracts ``10*log10(4)`` dB assuming four identical rotors.

        Parameters
        ----------
        total_drone_swl : np.ndarray
            Per-band total-drone SWL in dB; shape ``(n_bands,)``.

        Returns
        -------
        np.ndarray
            Per-band single-rotor reference SWL in dB; shape ``(n_bands,)``.
        """
        return np.asarray(total_drone_swl, dtype=float) - 10.0 * np.log10(4.0)

    @staticmethod
    def _spherical_spreading_db(distance_m: float) -> float:
        """
        Compute spherical spreading loss in dB for a given distance.

        Uses ``10*log10(4*pi*d^2)``. A small numerical guard is applied for d→0.

        Parameters
        ----------
        distance_m : float
            Source–receiver distance in meters.

        Returns
        -------
        float
            Spreading loss in dB (positive value).
        """
        d = max(float(distance_m), 1e-6)
        return float(10.0 * np.log10(4.0 * np.pi * d**2))

    def _propagate_per_band(
        self,
        swl_per_band: np.ndarray,
        distance_m: float,
        alpha_per_band: np.ndarray = None,
        di_per_band: np.ndarray = None,
    ) -> np.ndarray:
        """
        Apply per-band propagation (spherical spreading + optional absorption/DI).

        Parameters
        ----------
        swl_per_band : np.ndarray
            Per-band SWL at source in dB; shape ``(n_bands,)``.
        distance_m : float
            Source–receiver distance in meters.
        alpha_per_band : np.ndarray or None
            Atmospheric absorption in dB/m per band; shape ``(n_bands,)``.
            If ``None``, zeros are used.
        di_per_band : np.ndarray or None
            Directivity index correction in dB per band; shape ``(n_bands,)``.
            If ``None``, zeros are used.

        Returns
        -------
        np.ndarray
            Per-band SPL at receiver in dB; shape ``(n_bands,)``.
        """
        swl = np.asarray(swl_per_band, dtype=float)
        if swl.ndim != 1:
            raise ValueError("swl_per_band must be 1D with shape (n_bands,)")

        n_bands = swl.shape[0]
        alpha = self._validate_per_band(alpha_per_band, n_bands, name="alpha_per_band")
        di = self._validate_per_band(di_per_band, n_bands, name="di_per_band")

        spreading_db = self._spherical_spreading_db(distance_m)
        d = max(float(distance_m), 1e-6)

        # Lp = Lw - spreading - alpha*d + DI
        return swl - spreading_db - alpha * d + di

    @staticmethod
    def _validate_per_band(arr: np.ndarray, n_bands: int, name: str) -> np.ndarray:
        """
        Validate or create a per-band array of given length.

        Parameters
        ----------
        arr : np.ndarray or None
            Candidate per-band array or ``None``.
        n_bands : int
            Target length.
        name : str
            Parameter name for error messages.

        Returns
        -------
        np.ndarray
            Array of shape ``(n_bands,)``.
        """
        if arr is None:
            return np.zeros(n_bands, dtype=float)
        x = np.asarray(arr, dtype=float)
        if x.shape != (n_bands,):
            raise ValueError(f"{name} must have shape (n_bands,), got {x.shape}.")
        return x

    # -------------------------------------------------------------------------
    # ISO 9613-1 absorption (simplified implementation)
    # -------------------------------------------------------------------------

    @staticmethod
    def _atmospheric_absorption_iso9613(
        f_hz: np.ndarray,
        temperature_c: float = 20.0,
        relative_humidity: float = 50.0,
        pressure_kpa: float = 101.325,
    ) -> np.ndarray:
        """
        Compute atmospheric absorption alpha(f) in dB/m (ISO 9613-1, simplified).

        Parameters
        ----------
        f_hz : array-like
            Frequencies in Hz.
        temperature_c : float, optional
            Air temperature in °C. Default is 20.0.
        relative_humidity : float, optional
            Relative humidity in %. Default is 50.0.
        pressure_kpa : float, optional
            Static pressure in kPa. Default is 101.325 (≈1 atm).

        Returns
        -------
        np.ndarray
            Alpha in dB/m for each input frequency.

        Notes
        -----
        This is a commonly used simplified implementation. For high-accuracy
        work, validate against a trusted ISO 9613-1 reference.
        """
        f = np.asarray(f_hz, dtype=float)
        T = temperature_c + 273.15  # K
        p_rel = pressure_kpa / 101.325  # relative to 1 atm

        # Saturation vapor pressure (hPa), empirical (Buck-like)
        C = -6.8346 * (273.16 / T) ** 1.261 + 4.6151
        Psat_hPa = 10 ** C  # hPa
        h = (relative_humidity / 100.0) * (Psat_hPa / (pressure_kpa))  # molar humidity ratio (approx)

        # Relaxation frequencies for O2 and N2 (ISO-like forms)
        frO = p_rel * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h))
        frN = p_rel * ( (T / 293.15) ** -0.5 ) * (
            9.0 + 280.0 * h * np.exp(-4.170 * ((T / 293.15) ** -1/3 - 1.0))
        )

        term_classical = 1.84e-11 * (1.0 / p_rel) * (T / 293.15) ** 0.5
        term_O2 = 0.01275 * np.exp(-2239.1 / T) / (frO + (f**2 / frO))
        term_N2 = 0.1068  * np.exp(-3352.0 / T) / (frN + (f**2 / frN))

        alpha = 8.686 * (f**2) * ( term_classical + (T / 293.15) ** -2.5 * (term_O2 + term_N2) )
        return alpha
