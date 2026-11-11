//! Channel identity: WAVEFORMATEXTENSIBLE slots, masks, and the mapping from
//! what a container/decoder reports to the slot table the DSP expects.

use crate::dsp::*;
use serde::{Deserialize, Serialize};

/// A channel of the module's internal table.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash, Serialize, Deserialize)]
pub enum Slot {
    Fl,
    Fr,
    Fc,
    Lfe,
    Bl,
    Br,
    Flc,
    Frc,
    Bc,
    Sl,
    Sr,
    Tc,
}

impl Slot {
    pub const ALL: [Slot; 12] = [
        Slot::Fl,
        Slot::Fr,
        Slot::Fc,
        Slot::Lfe,
        Slot::Bl,
        Slot::Br,
        Slot::Flc,
        Slot::Frc,
        Slot::Bc,
        Slot::Sl,
        Slot::Sr,
        Slot::Tc,
    ];

    pub fn bit(self) -> u32 {
        match self {
            Slot::Fl => CH_FL,
            Slot::Fr => CH_FR,
            Slot::Fc => CH_FC,
            Slot::Lfe => CH_LFE,
            Slot::Bl => CH_BL,
            Slot::Br => CH_BR,
            Slot::Flc => CH_FLC,
            Slot::Frc => CH_FRC,
            Slot::Bc => CH_BC,
            Slot::Sl => CH_SL,
            Slot::Sr => CH_SR,
            Slot::Tc => CH_TC,
        }
    }

    /// Index into the DSP's `ch[SLOTS]`.
    pub fn index(self) -> usize {
        slot_of(self.bit())
    }

    pub fn name(self) -> &'static str {
        match self {
            Slot::Fl => "FL",
            Slot::Fr => "FR",
            Slot::Fc => "FC",
            Slot::Lfe => "LFE",
            Slot::Bl => "BL",
            Slot::Br => "BR",
            Slot::Flc => "FLC",
            Slot::Frc => "FRC",
            Slot::Bc => "BC",
            Slot::Sl => "SL",
            Slot::Sr => "SR",
            Slot::Tc => "TC",
        }
    }

    pub fn parse(s: &str) -> Option<Slot> {
        let up = s.trim().to_ascii_uppercase();
        Slot::ALL.into_iter().find(|v| v.name() == up)
    }
}

/// Channel mask implied by a stream order.
pub fn mask_of(order: &[Slot]) -> u32 {
    order.iter().fold(0u32, |m, s| m | s.bit())
}

/// A stream can be downmixed only if it carries a front pair; everything else
/// is refused rather than silently folded.
pub fn is_downmixable(order: &[Slot]) -> bool {
    let m = mask_of(order);
    m & CH_FL != 0 && m & CH_FR != 0
}

/// Pretty-print an order, e.g. `FL FR FC LFE BL BR SL SR`.
pub fn format_order(order: &[Slot]) -> String {
    order.iter().map(|s| s.name()).collect::<Vec<_>>().join(" ")
}

/// Mask as the hex literal ffmpeg's `-channel_layout` accepts, e.g. `0x63f`.
pub fn format_mask_hex(mask: u32) -> String {
    format!("0x{mask:x}")
}

/// Well-known orders, in the interleaved order ffmpeg/AVFoundation emit them.
pub const ORDERS: &[(&str, &[Slot])] = &[
    (
        "7.1 (FL FR FC LFE BL BR SL SR)",
        &[
            Slot::Fl,
            Slot::Fr,
            Slot::Fc,
            Slot::Lfe,
            Slot::Bl,
            Slot::Br,
            Slot::Sl,
            Slot::Sr,
        ],
    ),
    (
        "7.1 front-wide (FL FR FC LFE BL BR FLC FRC)",
        &[
            Slot::Fl,
            Slot::Fr,
            Slot::Fc,
            Slot::Lfe,
            Slot::Bl,
            Slot::Br,
            Slot::Flc,
            Slot::Frc,
        ],
    ),
    (
        "5.1 (FL FR FC LFE BL BR)",
        &[Slot::Fl, Slot::Fr, Slot::Fc, Slot::Lfe, Slot::Bl, Slot::Br],
    ),
    (
        "5.1 side (FL FR FC LFE SL SR)",
        &[Slot::Fl, Slot::Fr, Slot::Fc, Slot::Lfe, Slot::Sl, Slot::Sr],
    ),
    (
        "6.1 (FL FR FC LFE BL BC BR)",
        &[
            Slot::Fl,
            Slot::Fr,
            Slot::Fc,
            Slot::Lfe,
            Slot::Bl,
            Slot::Bc,
            Slot::Br,
        ],
    ),
    (
        "quad (FL FR BL BR)",
        &[Slot::Fl, Slot::Fr, Slot::Bl, Slot::Br],
    ),
    (
        "quad side (FL FR SL SR)",
        &[Slot::Fl, Slot::Fr, Slot::Sl, Slot::Sr],
    ),
    ("stereo (FL FR)", &[Slot::Fl, Slot::Fr]),
];

/// Map a decoder-reported layout name (and channel count) to a stream order.
///
/// ffmpeg's names are matched case-insensitively. `None` layout with a
/// channel count we can reason about falls back to the convention that
/// containers actually use (8ch => 7.1, 6ch => 5.1, 2ch => stereo); anything
/// else is reported as unsupported so the caller can ask for an explicit order
/// instead of guessing and folding a surround channel into the wrong ear.
pub fn order_from_layout(layout: Option<&str>, channels: usize) -> Option<Vec<Slot>> {
    use Slot::*;
    let named: &[(&str, &[Slot])] = &[
        ("mono", &[Fc]),
        ("stereo", &[Fl, Fr]),
        ("2.1", &[Fl, Fr, Lfe]),
        ("3.0", &[Fl, Fr, Fc]),
        ("3.0(back)", &[Fl, Fr, Bc]),
        ("quad", &[Fl, Fr, Bl, Br]),
        ("4.0", &[Fl, Fr, Bl, Br]),
        ("4.0(front)", &[Fl, Fr, Flc, Frc]),
        ("quad(side)", &[Fl, Fr, Sl, Sr]),
        ("5.0", &[Fl, Fr, Fc, Bl, Br]),
        ("5.0(side)", &[Fl, Fr, Fc, Sl, Sr]),
        ("5.1", &[Fl, Fr, Fc, Lfe, Bl, Br]),
        ("5.1(side)", &[Fl, Fr, Fc, Lfe, Sl, Sr]),
        ("6.0", &[Fl, Fr, Fc, Bl, Br, Bc]),
        ("6.0(front)", &[Fl, Fr, Fc, Flc, Frc, Bc]),
        ("hexagonal", &[Fl, Fr, Fc, Bl, Br, Bc]),
        ("6.1", &[Fl, Fr, Fc, Lfe, Bl, Bc, Br]),
        ("6.1(back)", &[Fl, Fr, Fc, Lfe, Bl, Bc, Br]),
        ("6.1(front)", &[Fl, Fr, Fc, Lfe, Flc, Bc, Frc]),
        ("7.0", &[Fl, Fr, Fc, Bl, Br, Sl, Sr]),
        ("7.0(front)", &[Fl, Fr, Fc, Flc, Frc, Bl, Br]),
        ("7.1", &[Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]),
        ("7.1(wide)", &[Fl, Fr, Fc, Lfe, Bl, Br, Flc, Frc]),
        ("7.1(wide-side)", &[Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]),
        ("octal", &[Fl, Fr, Fc, Bc, Bl, Br, Flc, Frc]),
        ("octagon", &[Fl, Fr, Fc, Bc, Bl, Br, Flc, Frc]),
    ];

    if let Some(l) = layout {
        let key = l.trim().to_ascii_lowercase();
        if let Some((_, slots)) = named.iter().find(|(n, _)| *n == key) {
            return Some(slots.to_vec());
        }
        // "unknown"/"undefined" and anything unlisted fall through.
    }

    match channels {
        8 => Some(vec![Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]),
        6 => Some(vec![Fl, Fr, Fc, Lfe, Bl, Br]),
        2 => Some(vec![Fl, Fr]),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use Slot::*;

    #[test]
    fn masks_match_the_c_reference() {
        assert_eq!(mask_of(&[Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]), MASK_7_1);
        assert_eq!(mask_of(&[Fl, Fr, Fc, Lfe, Bl, Br, Flc, Frc]), MASK_8CH);
        assert_eq!(mask_of(&[Fl, Fr, Fc, Lfe, Bl, Br]), MASK_5_1);
        assert_eq!(mask_of(&[Fl, Fr]), MASK_STEREO);
    }

    #[test]
    fn slot_indices_are_wfx_bit_positions() {
        assert_eq!(Slot::Fl.index(), 0);
        assert_eq!(Slot::Lfe.index(), 3);
        assert_eq!(Slot::Bc.index(), 8);
        assert_eq!(Slot::Sl.index(), 9);
        assert_eq!(Slot::Sr.index(), 10);
        assert_eq!(Slot::Tc.index(), 11);
    }

    #[test]
    fn layout_mapping() {
        assert_eq!(
            order_from_layout(Some("7.1"), 8).unwrap(),
            vec![Fl, Fr, Fc, Lfe, Bl, Br, Sl, Sr]
        );
        // Atmos beds often report 7.1; unknown 8ch still resolves.
        assert_eq!(order_from_layout(Some("unknown"), 8).unwrap().len(), 8);
        assert!(order_from_layout(Some("unknown"), 14).is_none());
        assert_eq!(order_from_layout(None, 6).unwrap().len(), 6);
    }

    #[test]
    fn mono_is_not_downmixable() {
        assert!(!is_downmixable(&[Fc]));
        assert!(is_downmixable(&[Fl, Fr]));
    }

    #[test]
    fn mask_hex_for_ffmpeg() {
        assert_eq!(format_mask_hex(MASK_7_1), "0x63f");
    }
}
