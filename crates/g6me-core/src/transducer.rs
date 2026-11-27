//! Klippel (2006) & King–Agerkvist (JAES 2018) electro-mechanical-thermal
//! nonlinear transducer assessment monitor.
//!
//! Runs a non-invasive parallel stereo electro-mechanical-thermal ODE model on
//! the post-limiter output stream to quantify physical driver stress without
//! coloring the delivered audio:
//!
//! - **Semi-implicit electrical ODE** with displacement-dependent voice-coil
//!   inductance `L_e(x) = L_e0 * (1 + l_1 * x / X_max)`, reluctance force
//!   `F_rel = 0.5 * (dL_e/dx) * i^2`, and thermal resistance drift
//!   `R_e(T) = R_e0 * (1 + alpha_Cu * dT)` (`alpha_Cu = 0.00393 / K`).
//! - **Nonlinear motor force factor** `Bl(x) = Bl_0 * (1 - bl_2 * (x / X_max)^2)`.
//! - **Nonlinear suspension stiffness + 5-pole viscoelastic creep**
//!   (Ritter–Klippel / King–Agerkvist 5-pole log-spaced exponential integrator
//!   spanning 1–100 Hz) integrated with 4x sub-stepped symplectic Euler.
//! - **Voice-coil thermal ODE** `C_th * d(dT)/dt = i^2 * R_e(T) - dT / R_th`.
//! - **Parallel linear Thiele–Small reference oscillator** sharing the exact
//!   linear parameters and 5-pole linear creep so orthogonal nonlinear velocity
//!   distortion (`est_thd_pct`) is measured directly on arbitrary music program.

use serde::{Deserialize, Serialize};

/// Selectable reference transducer profiles for post-limiter physical stress assessment.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Serialize, Deserialize)]
pub enum TransducerProfile {
    /// Assessment disabled (zero CPU overhead, `Report::transducer == None`).
    #[default]
    None,
    /// 32-ohm dynamic over-ear headphone driver (0.80 mm Xmax, 1.1 V peak FS).
    HeadphoneDynamic32,
    /// 32-ohm planar-magnetic over-ear headphone driver (0.55 mm Xmax, low Bl/Kms nonlinearity).
    HeadphonePlanar,
    /// 5-inch active studio monitor woofer (6.4 ohm, 4.0 mm Xmax, 11.0 V peak FS).
    StudioMonitor5In,
}

impl TransducerProfile {
    pub const ALL: [TransducerProfile; 4] = [
        TransducerProfile::None,
        TransducerProfile::HeadphoneDynamic32,
        TransducerProfile::HeadphonePlanar,
        TransducerProfile::StudioMonitor5In,
    ];

    pub fn as_str(self) -> &'static str {
        match self {
            TransducerProfile::None => "none",
            TransducerProfile::HeadphoneDynamic32 => "headphone-dynamic",
            TransducerProfile::HeadphonePlanar => "headphone-planar",
            TransducerProfile::StudioMonitor5In => "studio-monitor",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "none" | "off" => Some(TransducerProfile::None),
            "headphone-dynamic" | "dynamic" | "hp-dynamic" => {
                Some(TransducerProfile::HeadphoneDynamic32)
            }
            "headphone-planar" | "planar" | "hp-planar" => Some(TransducerProfile::HeadphonePlanar),
            "studio-monitor" | "monitor" | "woofer" | "5in" => {
                Some(TransducerProfile::StudioMonitor5In)
            }
            _ => None,
        }
    }

    pub fn label(self) -> &'static str {
        match self {
            TransducerProfile::None => "off",
            TransducerProfile::HeadphoneDynamic32 => "headphone dynamic (32 ohm)",
            TransducerProfile::HeadphonePlanar => "headphone planar (32 ohm)",
            TransducerProfile::StudioMonitor5In => "studio monitor (5\" woofer)",
        }
    }

    pub fn params(self) -> Option<TransducerParams> {
        match self {
            TransducerProfile::None => None,
            TransducerProfile::HeadphoneDynamic32 => Some(TransducerParams {
                re0_ohm: 32.0,
                le0_h: 0.28e-3,
                bl0_tm: 1.8,
                mms_kg: 0.25e-3,
                kms0_nm: 45.0,
                rms_nsm: 0.085,
                xmax_m: 0.80e-3,
                v_peak_fs: 1.1,
                r_th_kw: 35.0,
                c_th_jk: 0.45,
                bl2: 0.25,
                k2: 0.35,
                k4: 0.15,
                l1: -0.15,
                creep_frac: 0.12,
            }),
            TransducerProfile::HeadphonePlanar => Some(TransducerParams {
                re0_ohm: 32.0,
                le0_h: 0.02e-3,
                bl0_tm: 2.2,
                mms_kg: 0.35e-3,
                kms0_nm: 55.0,
                rms_nsm: 0.120,
                xmax_m: 0.55e-3,
                v_peak_fs: 0.85,
                r_th_kw: 18.0,
                c_th_jk: 1.20,
                bl2: 0.04,
                k2: 0.10,
                k4: 0.04,
                l1: -0.02,
                creep_frac: 0.06,
            }),
            TransducerProfile::StudioMonitor5In => Some(TransducerParams {
                re0_ohm: 6.4,
                le0_h: 0.48e-3,
                bl0_tm: 5.8,
                mms_kg: 9.50e-3,
                kms0_nm: 1500.0,
                rms_nsm: 0.72,
                xmax_m: 4.00e-3,
                v_peak_fs: 11.0,
                r_th_kw: 4.5,
                c_th_jk: 14.0,
                bl2: 0.32,
                k2: 0.45,
                k4: 0.20,
                l1: -0.20,
                creep_frac: 0.14,
            }),
        }
    }
}

/// Physical Thiele–Small, Klippel nonlinear, and thermal parameters for one driver.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TransducerParams {
    pub re0_ohm: f64,
    pub le0_h: f64,
    pub bl0_tm: f64,
    pub mms_kg: f64,
    pub kms0_nm: f64,
    pub rms_nsm: f64,
    pub xmax_m: f64,
    /// Peak amplifier drive voltage corresponding to 0 dBFS (`1.0`).
    pub v_peak_fs: f64,
    pub r_th_kw: f64,
    pub c_th_jk: f64,
    /// Quadratic `Bl(x)` droop coefficient at `|x| = X_max`.
    pub bl2: f64,
    /// Quadratic stiffness hardening coefficient at `|x| = X_max`.
    pub k2: f64,
    /// Quartic stiffness hardening coefficient at `|x| = X_max`.
    pub k4: f64,
    /// Linear inductance slope `l_1` at `x = X_max`.
    pub l1: f64,
    /// Total fractional viscoelastic creep stiffness ratio across 5 log-spaced poles.
    pub creep_frac: f64,
}

/// Summary report emitted in [`crate::pipeline::Report::transducer`].
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct TransducerAssessment {
    pub profile: String,
    pub peak_excursion_mm: f64,
    pub peak_excursion_pct_xmax: f64,
    pub thermal_rise_c: f64,
    pub peak_power_w: f64,
    pub min_bl_ratio: f64,
    pub est_thd_pct: f64,
    pub excursion_headroom_db: f64,
}

impl TransducerAssessment {
    pub fn detail_line(&self) -> String {
        format!(
            "transducer ({}): peak excursion {:.3} mm ({:.1}% Xmax, headroom {:+.1} dB), \
             min Bl/Bl0 {:.3}, thermal +{:.2} C, peak {:.3} W, est nonlinear distortion {:.2}%",
            self.profile,
            self.peak_excursion_mm,
            self.peak_excursion_pct_xmax,
            self.excursion_headroom_db,
            self.min_bl_ratio,
            self.thermal_rise_c,
            self.peak_power_w,
            self.est_thd_pct,
        )
    }
}

/// 5-pole log-spaced creep frequencies (Hz) spanning 1 Hz .. 100 Hz (King–Agerkvist 2018).
const CREEP_POLES_HZ: [f64; 5] = [1.0, 3.162_277_66, 10.0, 31.622_776_6, 100.0];
/// Copper voice-coil temperature coefficient of resistance (`1 / K`).
const ALPHA_CU: f64 = 0.00393;
/// Mechanical/electrical sub-steps per audio sample for symplectic ODE accuracy.
const SUBSTEPS: usize = 4;

#[derive(Clone, Debug)]
struct ChannelState {
    x: f64,
    v: f64,
    i: f64,
    dt_temp: f64,
    z_creep: [f64; 5],
    // Parallel linear reference state for orthogonal nonlinear distortion estimation.
    x_lin: f64,
    v_lin: f64,
    i_lin: f64,
    z_lin: [f64; 5],
    // Running metrics.
    max_abs_x: f64,
    max_dt_temp: f64,
    max_power_w: f64,
    min_bl_ratio: f64,
    sum_v2: f64,
    sum_vl2: f64,
    sum_v_vl: f64,
}

impl Default for ChannelState {
    fn default() -> Self {
        Self {
            x: 0.0,
            v: 0.0,
            i: 0.0,
            dt_temp: 0.0,
            z_creep: [0.0; 5],
            x_lin: 0.0,
            v_lin: 0.0,
            i_lin: 0.0,
            z_lin: [0.0; 5],
            max_abs_x: 0.0,
            max_dt_temp: 0.0,
            max_power_w: 0.0,
            min_bl_ratio: 1.0,
            sum_v2: 0.0,
            sum_vl2: 0.0,
            sum_v_vl: 0.0,
        }
    }
}

/// Non-invasive stereo electro-mechanical-thermal transducer assessment monitor.
#[derive(Clone, Debug)]
pub struct TransducerMonitor {
    profile: TransducerProfile,
    p: TransducerParams,
    dt: f64,
    inv_dt: f64,
    inv_mms: f64,
    inv_xmax: f64,
    dldx: f64,
    alpha_creep_k: f64,
    creep_exp: [f64; 5],
    creep_one_minus: [f64; 5],
    lin_elec_denom: f64,
    ch: [ChannelState; 2],
}

impl TransducerMonitor {
    /// Construct a monitor for `profile` at `sample_rate`. Returns `None` for
    /// [`TransducerProfile::None`].
    pub fn new(profile: TransducerProfile, sample_rate: u32) -> Option<Self> {
        let p = profile.params()?;
        let sr = f64::from(sample_rate.max(1));
        let dt = 1.0 / (sr * (SUBSTEPS as f64));
        let inv_dt = 1.0 / dt;
        let mut creep_exp = [0.0f64; 5];
        let mut creep_one_minus = [0.0f64; 5];
        for (k, &f_hz) in CREEP_POLES_HZ.iter().enumerate() {
            let e = (-2.0 * std::f64::consts::PI * f_hz * dt).exp();
            creep_exp[k] = e;
            creep_one_minus[k] = 1.0 - e;
        }
        Some(Self {
            profile,
            p,
            dt,
            inv_dt,
            inv_mms: 1.0 / p.mms_kg,
            inv_xmax: 1.0 / p.xmax_m,
            dldx: p.le0_h * p.l1 / p.xmax_m,
            alpha_creep_k: (p.creep_frac / 5.0) * p.kms0_nm,
            creep_exp,
            creep_one_minus,
            lin_elec_denom: p.le0_h * inv_dt + p.re0_ohm,
            ch: [ChannelState::default(), ChannelState::default()],
        })
    }

    #[inline]
    fn step_channel(&mut self, idx: usize, sample: f64) {
        let p = self.p;
        let dt = self.dt;
        let inv_dt = self.inv_dt;
        let inv_mms = self.inv_mms;
        let inv_xmax = self.inv_xmax;
        let dldx = self.dldx;
        let alpha_ck = self.alpha_creep_k;
        let c_exp = self.creep_exp;
        let c_om = self.creep_one_minus;
        let lin_denom = self.lin_elec_denom;
        let u = p.v_peak_fs * sample;
        let st = &mut self.ch[idx];

        for _ in 0..SUBSTEPS {
            // 1. Nonlinear driver state
            let xn = st.x * inv_xmax;
            let xn2 = xn * xn;
            let bl_ratio = (1.0 - p.bl2 * xn2).clamp(0.15, 1.0);
            let bl = p.bl0_tm * bl_ratio;
            let le = (p.le0_h * (1.0 + p.l1 * xn)).max(0.2 * p.le0_h);
            let re = p.re0_ohm * (1.0 + ALPHA_CU * st.dt_temp);

            // Unconditionally stable semi-implicit electrical update (explicit ops, zero FMA)
            let denom = (le * inv_dt + re + dldx * st.v).max(0.1 * re);
            let num = (le * st.i) * inv_dt + u - bl * st.v;
            let cur = num / denom;

            // Nonlinear stiffness + 5-pole exponential-integrator viscoelastic creep
            let k_nl = p.kms0_nm * (1.0 + p.k2 * xn2 + p.k4 * xn2 * xn2);
            let mut f_creep = 0.0;
            for k in 0..5 {
                let zk = c_exp[k] * st.z_creep[k] + c_om[k] * st.x;
                st.z_creep[k] = zk;
                f_creep += alpha_ck * (st.x - zk);
            }
            let f_spring = k_nl * st.x + f_creep;
            let f_reluct = 0.5 * dldx * cur * cur;
            let acc = (bl * cur + f_reluct - p.rms_nsm * st.v - f_spring) * inv_mms;

            // Symplectic Euler mechanical update
            let v_next = st.v + acc * dt;
            let x_next = st.x + v_next * dt;

            // Voice-coil thermal ODE
            let p_elec = cur * cur * re;
            let dt_next = (st.dt_temp + (p_elec - st.dt_temp / p.r_th_kw) * (dt / p.c_th_jk))
                .clamp(0.0, 250.0);

            st.i = cur;
            st.v = v_next;
            st.x = x_next;
            st.dt_temp = dt_next;

            // 2. Parallel linear reference state (identical linear TS + 5-pole creep)
            let cur_l = ((p.le0_h * st.i_lin) * inv_dt + u - p.bl0_tm * st.v_lin) / lin_denom;
            let mut f_creep_l = 0.0;
            for k in 0..5 {
                let zkl = c_exp[k] * st.z_lin[k] + c_om[k] * st.x_lin;
                st.z_lin[k] = zkl;
                f_creep_l += alpha_ck * (st.x_lin - zkl);
            }
            let acc_l =
                (p.bl0_tm * cur_l - p.rms_nsm * st.v_lin - p.kms0_nm * st.x_lin - f_creep_l)
                    * inv_mms;
            let vl_next = st.v_lin + acc_l * dt;
            let xl_next = st.x_lin + vl_next * dt;
            st.i_lin = cur_l;
            st.v_lin = vl_next;
            st.x_lin = xl_next;

            // 3. Peak & distortion accumulators
            let ax = x_next.abs();
            if ax > st.max_abs_x {
                st.max_abs_x = ax;
            }
            if dt_next > st.max_dt_temp {
                st.max_dt_temp = dt_next;
            }
            if p_elec > st.max_power_w {
                st.max_power_w = p_elec;
            }
            if bl_ratio < st.min_bl_ratio {
                st.min_bl_ratio = bl_ratio;
            }
            st.sum_v2 += v_next * v_next;
            st.sum_vl2 += vl_next * vl_next;
            st.sum_v_vl += v_next * vl_next;
        }
    }

    /// Feed one post-limiter stereo block into the assessment monitor.
    pub fn push(&mut self, left: &[f64], right: &[f64]) {
        for (&l, &r) in left.iter().zip(right.iter()) {
            self.step_channel(0, l);
            self.step_channel(1, r);
        }
    }

    /// Summarize worst-case excursion, thermal rise, Bl compression, and distortion
    /// across both stereo channels.
    pub fn summarize(&self) -> TransducerAssessment {
        let max_x = self.ch[0].max_abs_x.max(self.ch[1].max_abs_x);
        let pct_xmax = 100.0 * max_x * self.inv_xmax;
        let thermal_rise_c = self.ch[0].max_dt_temp.max(self.ch[1].max_dt_temp);
        let peak_power_w = self.ch[0].max_power_w.max(self.ch[1].max_power_w);
        let min_bl_ratio = self.ch[0].min_bl_ratio.min(self.ch[1].min_bl_ratio);

        let mut worst_thd = 0.0f64;
        for st in &self.ch {
            if st.sum_v2 > 1e-24 && st.sum_vl2 > 1e-24 {
                let proj = (st.sum_v_vl * st.sum_v_vl) / st.sum_vl2;
                let orth = (st.sum_v2 - proj).max(0.0);
                let pct = 100.0 * (orth / st.sum_v2).sqrt();
                if pct > worst_thd {
                    worst_thd = pct;
                }
            }
        }
        let excursion_headroom_db = if max_x > 1e-12 {
            20.0 * (self.p.xmax_m / max_x).log10()
        } else {
            f64::INFINITY
        };

        TransducerAssessment {
            profile: self.profile.as_str().to_string(),
            peak_excursion_mm: max_x * 1000.0,
            peak_excursion_pct_xmax: pct_xmax,
            thermal_rise_c,
            peak_power_w,
            min_bl_ratio,
            est_thd_pct: worst_thd,
            excursion_headroom_db,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn klippel_transducer_scales_excursion_bl_compression_and_distortion_physically() {
        let sr = 48_000u32;
        let n = 24_000usize; // 0.5 s at 64 Hz (exact 32 cycles)
        let sine = |amp: f64| -> Vec<f64> {
            (0..n)
                .map(|i| {
                    amp * (2.0 * std::f64::consts::PI * 64.0 * (i as f64) / f64::from(sr)).sin()
                })
                .collect()
        };

        let quiet = sine(0.10);
        let loud = sine(1.00);

        let mut m_q = TransducerMonitor::new(TransducerProfile::HeadphoneDynamic32, sr).unwrap();
        m_q.push(&quiet, &quiet);
        let a_q = m_q.summarize();

        let mut m_l = TransducerMonitor::new(TransducerProfile::HeadphoneDynamic32, sr).unwrap();
        m_l.push(&loud, &loud);
        let a_l = m_l.summarize();

        // Low drive stays linear: high Bl/Bl0, low distortion, positive headroom.
        assert!(
            a_q.min_bl_ratio > 0.99,
            "expected near-unity Bl/Bl0 at -20 dBFS, got {}",
            a_q.min_bl_ratio
        );
        assert!(
            a_q.est_thd_pct < 0.5,
            "expected low distortion at -20 dBFS, got {}%",
            a_q.est_thd_pct
        );
        assert!(a_q.excursion_headroom_db > 15.0);

        // Full-scale 64 Hz resonance drive compresses Bl(x), heats voice coil, raises distortion.
        assert!(
            a_l.peak_excursion_mm > a_q.peak_excursion_mm * 5.0,
            "full-scale excursion did not scale: {} vs {}",
            a_l.peak_excursion_mm,
            a_q.peak_excursion_mm
        );
        assert!(
            a_l.min_bl_ratio < 0.85,
            "expected substantial Bl(x) compression near Xmax, got {}",
            a_l.min_bl_ratio
        );
        assert!(
            a_l.est_thd_pct > a_q.est_thd_pct * 10.0,
            "expected nonlinear distortion jump near Xmax: loud={}%, quiet={}%",
            a_l.est_thd_pct,
            a_q.est_thd_pct
        );
        assert!(a_l.thermal_rise_c > 0.0 && a_l.peak_power_w > 0.0);

        // Planar headphone must show strictly lower distortion than dynamic at full scale.
        let mut m_p = TransducerMonitor::new(TransducerProfile::HeadphonePlanar, sr).unwrap();
        m_p.push(&loud, &loud);
        let a_p = m_p.summarize();
        assert!(
            a_p.est_thd_pct < a_l.est_thd_pct,
            "planar distortion ({:.2}%) should be lower than dynamic ({:.2}%)",
            a_p.est_thd_pct,
            a_l.est_thd_pct
        );

        // Studio monitor at full scale dissipates multi-watt electrical power and heats up.
        let mut m_s = TransducerMonitor::new(TransducerProfile::StudioMonitor5In, sr).unwrap();
        m_s.push(&loud, &loud);
        let a_s = m_s.summarize();
        assert!(
            a_s.peak_power_w > 2.0 && a_s.peak_excursion_mm > 2.0,
            "unexpected studio monitor physics: {a_s:?}"
        );
    }

    #[test]
    fn klippel_transducer_profile_parsing_round_trips() {
        for p in TransducerProfile::ALL {
            assert_eq!(TransducerProfile::parse(p.as_str()), Some(p));
        }
        assert!(TransducerMonitor::new(TransducerProfile::None, 48_000).is_none());
    }
}
