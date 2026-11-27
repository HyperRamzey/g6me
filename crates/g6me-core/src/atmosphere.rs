//! Atmospheric thermodynamics and quantum molecular sound absorption.
//!
//! Replaces the hardcoded `343 m/s` constant with:
//!
//! 1. **Owen Cramer (JASA 1993)** — *"The variation of the specific heat ratio and
//!    the speed of sound in air with temperature, pressure, humidity, and CO₂
//!    concentration"*: exact virial-equation speed of sound `c(T, P, RH, CO₂)`.
//! 2. **Bass, Sutherland, Zuckerwar, Blackstock & Hester (JASA 1995 / ISO 9613-1)** —
//!    *"Atmospheric absorption of sound: Further developments"*: exact vibrational
//!    relaxation frequencies of oxygen (`O₂`) and nitrogen (`N₂`) molecules in air
//!    plus classical thermoviscous attenuation, yielding frequency-dependent
//!    atmospheric attenuation `α(f)` in dB/m.
//!
//! All arithmetic uses plain IEEE-754 `f64` operations with zero FMA contraction.

use serde::{Deserialize, Serialize};

/// Thermodynamic & chemical state of the virtual listening room air.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AtmosphereCfg {
    /// Ambient air temperature in degrees Celsius (`-20.0 ..= 50.0`, default `21.5`).
    pub temp_c: f64,
    /// Relative humidity percentage (`0.0 ..= 100.0`, default `45.0`).
    pub rh_pct: f64,
    /// Atmospheric carbon dioxide concentration in ppm (`0.0 ..= 10_000.0`, default `420.0`).
    pub co2_ppm: f64,
    /// Ambient barometric pressure in kilopascals (`50.0 ..= 120.0`, default `101.325`).
    pub pressure_kpa: f64,
}

impl Default for AtmosphereCfg {
    fn default() -> Self {
        Self {
            temp_c: 21.5,
            rh_pct: 45.0,
            co2_ppm: 420.0,
            pressure_kpa: 101.325,
        }
    }
}

impl AtmosphereCfg {
    pub fn validate(&self) -> std::result::Result<(), String> {
        if !self.temp_c.is_finite() || !(-20.0..=50.0).contains(&self.temp_c) {
            return Err(format!(
                "atmosphere.temp_c must be in -20.0..=50.0 °C, got {}",
                self.temp_c
            ));
        }
        if !self.rh_pct.is_finite() || !(0.0..=100.0).contains(&self.rh_pct) {
            return Err(format!(
                "atmosphere.rh_pct must be in 0.0..=100.0 %, got {}",
                self.rh_pct
            ));
        }
        if !self.co2_ppm.is_finite() || !(0.0..=10_000.0).contains(&self.co2_ppm) {
            return Err(format!(
                "atmosphere.co2_ppm must be in 0..=10000 ppm, got {}",
                self.co2_ppm
            ));
        }
        if !self.pressure_kpa.is_finite() || !(50.0..=120.0).contains(&self.pressure_kpa) {
            return Err(format!(
                "atmosphere.pressure_kpa must be in 50.0..=120.0 kPa, got {}",
                self.pressure_kpa
            ));
        }
        Ok(())
    }

    /// Exact thermodynamic speed of sound in m/s (Owen Cramer, JASA 1993, Table II).
    ///
    /// Incorporates the Davis (1992) saturation vapor pressure and enhancement factor
    /// for water-vapor mole fraction `x_w`, barometric pressure `P` (Pa), temperature
    /// `t` (°C), and `CO₂` mole fraction `x_c = co2_ppm * 1e-6`.
    #[inline]
    pub fn speed_of_sound_m_s(&self) -> f64 {
        let t = self.temp_c;
        let t_k = t + 273.15;
        let p_pa = self.pressure_kpa * 1_000.0;

        // Davis (1992) / Cramer (1993) saturation vapor pressure of water (Pa).
        let p_sv = (1.237_884_7e-5 * t_k * t_k - 1.912_131_6e-2 * t_k + 33.937_110_47
            - 6.343_164_5e3 / t_k)
            .exp();
        // Enhancement factor for moist air.
        let f_enh = 1.000_62 + 3.14e-8 * p_pa + 5.6e-7 * t * t;
        let x_w = ((self.rh_pct * 0.01) * f_enh * p_sv / p_pa).clamp(0.0, 0.2);
        let x_c = (self.co2_ppm * 1e-6).max(0.0);

        // Cramer (1993) 16-coefficient virial polynomial (Table II).
        const A0: f64 = 331.5024;
        const A1: f64 = 0.603_055;
        const A2: f64 = -0.000_528;
        const A3: f64 = 51.471_935;
        const A4: f64 = 0.149_587_4;
        const A5: f64 = -0.000_782;
        const A6: f64 = -1.82e-7;
        const A7: f64 = 3.73e-8;
        const A8: f64 = -2.93e-10;
        const A9: f64 = -85.209_31;
        const A10: f64 = -0.228_525;
        const A11: f64 = 5.91e-5;
        const A12: f64 = -2.835_149;
        const A13: f64 = -2.15e-13;
        const A14: f64 = 29.179_762;
        const A15: f64 = 0.000_486;

        let t2 = t * t;
        A0 + A1 * t
            + A2 * t2
            + (A3 + A4 * t + A5 * t2) * x_w
            + (A6 + A7 * t + A8 * t2) * p_pa
            + (A9 + A10 * t + A11 * t2) * x_c
            + A12 * x_w * x_w
            + A13 * p_pa * p_pa
            + A14 * x_c * x_c
            + A15 * x_w * p_pa * x_c
    }

    /// Molar concentration of water vapor `h` in percent per ISO 9613-1 / Bass (1995).
    #[inline]
    pub fn water_vapor_pct(&self) -> f64 {
        let t_k = self.temp_c + 273.15;
        const T01: f64 = 273.16;
        const PR_KPA: f64 = 101.325;
        let c_sat = -6.8346 * (T01 / t_k).powf(1.261) + 4.6151;
        let psat_pr = 10.0_f64.powf(c_sat);
        self.rh_pct * psat_pr / (self.pressure_kpa / PR_KPA)
    }

    /// Molecular vibrational relaxation frequencies `(f_rO2, f_rN2)` in Hz
    /// (Bass, Sutherland, Zuckerwar, Blackstock & Hester, JASA 1995 / ISO 9613-1).
    #[inline]
    pub fn relaxation_frequencies_hz(&self) -> (f64, f64) {
        let t_k = self.temp_c + 273.15;
        const T0: f64 = 293.15;
        const PR_KPA: f64 = 101.325;
        let p_ratio = self.pressure_kpa / PR_KPA;
        let t_ratio = t_k / T0;
        let h = self.water_vapor_pct();

        let fr_o2 = p_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
        let fr_n2 = p_ratio
            * t_ratio.powf(-0.5)
            * (9.0 + 280.0 * h * (-4.170 * (t_ratio.powf(-1.0 / 3.0) - 1.0)).exp());
        (fr_o2, fr_n2)
    }

    /// Frequency-dependent atmospheric attenuation coefficient `α(f)` in dB/m
    /// (ISO 9613-1 / Bass et al. JASA 1995).
    ///
    /// Combines classical thermoviscous absorption (`∝ f²`) with quantum vibrational
    /// relaxation of `O₂` (`Θ_O2 = 2239.1 K`) and `N₂` (`Θ_N2 = 3352.0 K`).
    #[inline]
    pub fn alpha_db_per_m(&self, f_hz: f64) -> f64 {
        if f_hz <= 0.0 {
            return 0.0;
        }
        let t_k = self.temp_c + 273.15;
        const T0: f64 = 293.15;
        const PR_KPA: f64 = 101.325;
        let p_ratio = self.pressure_kpa / PR_KPA;
        let t_ratio = t_k / T0;
        let (fr_o2, fr_n2) = self.relaxation_frequencies_hz();
        let f2 = f_hz * f_hz;

        let classical = 1.84e-11 * (1.0 / p_ratio) * t_ratio.sqrt();
        let relax_o2 = 0.012_75 * (-2239.1 / t_k).exp() / (fr_o2 + f2 / fr_o2);
        let relax_n2 = 0.106_8 * (-3352.0 / t_k).exp() / (fr_n2 + f2 / fr_n2);
        8.686 * f2 * (classical + t_ratio.powf(-2.5) * (relax_o2 + relax_n2))
    }

    /// Linear propagation amplitude gain (`<= 1.0`) due to molecular air absorption
    /// over propagation distance `distance_m` at frequency `f_hz`.
    #[inline]
    pub fn air_gain_linear(&self, f_hz: f64, distance_m: f64) -> f64 {
        if f_hz <= 0.0 || distance_m <= 0.0 {
            return 1.0;
        }
        let atten_db = self.alpha_db_per_m(f_hz) * distance_m;
        10.0_f64.powf(-atten_db / 20.0)
    }

    /// Compact human-readable telemetry string for CLI/GUI readouts.
    pub fn summary(&self) -> String {
        let c = self.speed_of_sound_m_s();
        let a20 = self.alpha_db_per_m(20_000.0);
        format!(
            "{:.1}°C, {:.0}% RH, {:.0} ppm CO₂, {:.2} kPa (c = {:.3} m/s, α₂₀k = {:.3} dB/m)",
            self.temp_c, self.rh_pct, self.co2_ppm, self.pressure_kpa, c, a20
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cramer_speed_of_sound_matches_reference_values() {
        // Standard studio air: 21.5°C, 45% RH, 420 ppm CO2, 101.325 kPa -> 344.854 m/s.
        let studio = AtmosphereCfg::default();
        let c_studio = studio.speed_of_sound_m_s();
        assert!(
            (c_studio - 344.854_446).abs() < 1e-4,
            "unexpected studio speed of sound: {c_studio}"
        );

        // Dry air at 20°C, 314 ppm CO2 (Cramer 1993 benchmark) -> 343.367 m/s.
        let dry20 = AtmosphereCfg {
            temp_c: 20.0,
            rh_pct: 0.0,
            co2_ppm: 314.0,
            pressure_kpa: 101.325,
        };
        let c_dry = dry20.speed_of_sound_m_s();
        assert!(
            (c_dry - 343.367_184).abs() < 1e-4,
            "unexpected dry 20C speed of sound: {c_dry}"
        );

        // Elevated CO2 (heavier molar mass -> lower gamma & lower c) must reduce c.
        let high_co2 = AtmosphereCfg {
            co2_ppm: 5000.0,
            ..studio
        };
        assert!(
            high_co2.speed_of_sound_m_s() < c_studio - 0.3,
            "elevated CO2 did not reduce speed of sound"
        );

        // Higher humidity (lighter H2O vs N2/O2) must increase c.
        let humid = AtmosphereCfg {
            rh_pct: 90.0,
            ..studio
        };
        assert!(
            humid.speed_of_sound_m_s() > c_studio + 0.5,
            "higher humidity did not increase speed of sound"
        );
    }

    #[test]
    fn iso9613_molecular_absorption_matches_physics() {
        let atm = AtmosphereCfg::default();
        let (fr_o2, fr_n2) = atm.relaxation_frequencies_hz();
        assert!(fr_o2 > 20_000.0 && fr_o2 < 50_000.0, "fr_o2={fr_o2}");
        assert!(fr_n2 > 200.0 && fr_n2 < 500.0, "fr_n2={fr_n2}");

        let a1k = atm.alpha_db_per_m(1_000.0);
        let a10k = atm.alpha_db_per_m(10_000.0);
        let a20k = atm.alpha_db_per_m(20_000.0);
        assert!(a1k > 0.0 && a1k < a10k && a10k < a20k);
        // At 21.5°C, 45% RH, 20 kHz alpha is ~0.5404 dB/m (~1.08 dB over 2 m).
        assert!(
            (a20k - 0.5404).abs() < 0.005,
            "unexpected 20kHz alpha: {a20k}"
        );

        // Gain at 20 kHz over 2 m is strictly < 1.0 (~0.883).
        let g2m = atm.air_gain_linear(20_000.0, 2.0);
        assert!(g2m > 0.85 && g2m < 0.92, "20kHz 2m air gain={g2m}");
        assert_eq!(atm.air_gain_linear(0.0, 2.0), 1.0);
    }
}
