//! The window palette: follow the system, dark, light, or pure black.
//!
//! Three of these are egui's own business — `Auto`, `Dark` and `Light` just set
//! egui's theme preference, and egui already ships a light and a dark palette and
//! follows the OS when asked. The fourth is ours.
//!
//! The OLED palette is not "dark but darker". An OLED pixel that is lit at all
//! costs power and, at night, glows behind everything else, so the point of this
//! mode is that every large surface is *exactly* `#000`: panels, windows, popups,
//! text-field interiors, the meter backgrounds. Widgets keep a faint fill, because
//! a button you cannot see is not a button — but the field behind them carries no
//! current at all. Anything "near black" would be a lie about the mode's purpose,
//! which is why the exact value is asserted in a test rather than eyeballed.

use egui::{Color32, Context, Theme as EguiTheme, ThemePreference, Visuals};

/// What the user picked in the Appearance control.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Palette {
    /// Follow the operating system. This is the default: a person who set their
    /// desktop to light at noon and dark at midnight should not have to tell the
    /// encoder twice.
    #[default]
    Auto,
    Dark,
    Light,
    /// Dark, with every large surface at exactly `#000`.
    Oled,
}

impl Palette {
    pub const ALL: [Palette; 4] = [Palette::Auto, Palette::Dark, Palette::Light, Palette::Oled];

    pub fn label(self) -> &'static str {
        match self {
            Palette::Auto => "system",
            Palette::Dark => "dark",
            Palette::Light => "light",
            Palette::Oled => "pure black",
        }
    }

    /// What this palette is showing right now, once the system is taken into
    /// account. Used for the caption, and by nothing that affects sound.
    pub fn resolved(self, ctx: &Context) -> EguiTheme {
        match self {
            Palette::Auto => ctx.system_theme().unwrap_or(EguiTheme::Dark),
            Palette::Light => EguiTheme::Light,
            Palette::Dark | Palette::Oled => EguiTheme::Dark,
        }
    }

    /// Install this palette on a context.
    ///
    /// Both palettes are written every time rather than only the active one. A
    /// previous choice may have put pure black in the dark slot, and switching to
    /// `Auto` on a machine whose OS is light must not leave the dark slot polluted
    /// for the next time the system flips to dark mode.
    pub fn apply(self, ctx: &Context) {
        match self {
            Palette::Auto => {
                ctx.set_theme(ThemePreference::System);
                ctx.set_visuals_of(EguiTheme::Dark, Visuals::dark());
                ctx.set_visuals_of(EguiTheme::Light, Visuals::light());
            }
            Palette::Dark => {
                ctx.set_theme(ThemePreference::Dark);
                ctx.set_visuals_of(EguiTheme::Dark, Visuals::dark());
                ctx.set_visuals_of(EguiTheme::Light, Visuals::light());
            }
            Palette::Light => {
                ctx.set_theme(ThemePreference::Light);
                ctx.set_visuals_of(EguiTheme::Dark, Visuals::dark());
                ctx.set_visuals_of(EguiTheme::Light, Visuals::light());
            }
            Palette::Oled => {
                ctx.set_theme(ThemePreference::Dark);
                ctx.set_visuals_of(EguiTheme::Dark, oled());
                ctx.set_visuals_of(EguiTheme::Light, Visuals::light());
            }
        }
    }
}

/// The pure-black palette: egui's dark one, with the lights cut to the floor on
/// everything large enough to matter.
pub fn oled() -> Visuals {
    let mut v = Visuals::dark();
    let black = Color32::BLACK;

    // Large surfaces: exactly zero.
    v.panel_fill = black;
    v.window_fill = black;
    v.extreme_bg_color = black;
    v.faint_bg_color = black;
    v.code_bg_color = black;

    // Widgets keep a readable fill — but the ones whose "frame" is really just the
    // background behind a label or a checkbox go black too, or a label on a grey
    // plate is the same lit rectangle the mode exists to avoid.
    v.widgets.noninteractive.bg_fill = black;
    v.widgets.noninteractive.weak_bg_fill = black;
    v.widgets.inactive.weak_bg_fill = Color32::from_gray(12);

    // Text. egui mixes the weak tier down from the normal one, and on a true-black
    // backdrop the default inactive grey is the first thing to vanish, so the tiers
    // that carry units and hints come up a step.
    v.widgets.inactive.fg_stroke.color = Color32::from_gray(196);
    v.widgets.noninteractive.fg_stroke.color = Color32::from_gray(216);
    v.hyperlink_color = Color32::from_rgb(96, 168, 226);

    // Borders do the separating now that surfaces cannot.
    v.window_stroke.color = Color32::from_gray(46);
    v.widgets.noninteractive.bg_stroke.color = Color32::from_gray(40);
    v.widgets.inactive.bg_stroke.color = Color32::from_gray(58);
    v.dark_mode = true;
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_follows_the_system() {
        assert_eq!(Palette::default(), Palette::Auto);
    }

    #[test]
    fn every_palette_has_a_name_and_they_are_all_reachable() {
        let names: Vec<&str> = Palette::ALL.iter().map(|p| p.label()).collect();
        assert_eq!(names, vec!["system", "dark", "light", "pure black"]);
        // Distinct, or the picker would have two buttons that do the same thing.
        assert_eq!(
            names.iter().collect::<std::collections::HashSet<_>>().len(),
            4
        );
    }

    #[test]
    fn oled_backdrops_are_exactly_black() {
        let v = oled();
        // The claim of the mode, checked rather than asserted in prose: a lit pixel
        // is a wasted pixel.
        for (what, c) in [
            ("panel", v.panel_fill),
            ("window", v.window_fill),
            ("extreme bg", v.extreme_bg_color),
            ("faint bg", v.faint_bg_color),
            ("noninteractive bg", v.widgets.noninteractive.bg_fill),
            (
                "noninteractive weak bg",
                v.widgets.noninteractive.weak_bg_fill,
            ),
        ] {
            assert_eq!(c, Color32::BLACK, "{what} must be #000, got {c:?}");
            let [r, g, b, _] = c.to_array();
            assert_eq!(u32::from(r) + u32::from(g) + u32::from(b), 0, "{what}");
        }
        assert!(v.dark_mode, "it is still the dark palette underneath");
    }

    #[test]
    fn oled_keeps_widgets_visible() {
        let v = oled();
        // Pure black is about the surfaces, not about hiding the controls: a button
        // must still have an edge and a fill a shade off the backdrop.
        assert_ne!(
            v.widgets.inactive.weak_bg_fill,
            Color32::BLACK,
            "a button identical to its background is not a button"
        );
        assert_ne!(v.widgets.inactive.bg_stroke.color, Color32::BLACK);
        assert_ne!(v.window_stroke.color, Color32::BLACK);
        assert_ne!(
            v.widgets.noninteractive.fg_stroke.color,
            Color32::BLACK,
            "black text on black is not a hint, it is an absence"
        );
    }

    #[test]
    fn resolved_reports_the_right_family() {
        let ctx = Context::default();
        assert_eq!(Palette::Light.resolved(&ctx), EguiTheme::Light);
        assert_eq!(Palette::Dark.resolved(&ctx), EguiTheme::Dark);
        // OLED is a dark palette with the backdrops cut; widgets laid out for dark.
        assert_eq!(Palette::Oled.resolved(&ctx), EguiTheme::Dark);
    }

    #[test]
    fn applying_a_palette_does_not_panic_and_survives_a_switch() {
        // The realistic sequence: start on system, go OLED, go back to system.
        // The bug this guards is the one where the pure-black slot survives the trip
        // and the app comes back looking OLED with "system" selected.
        let ctx = Context::default();
        for p in [
            Palette::Auto,
            Palette::Oled,
            Palette::Light,
            Palette::Auto,
            Palette::Dark,
        ] {
            p.apply(&ctx);
            assert_eq!(p.resolved(&ctx), p.resolved(&ctx));
        }
        Palette::Auto.apply(&ctx);
        let dark = ctx.style_of(EguiTheme::Dark).visuals.clone();
        assert_ne!(
            dark.panel_fill,
            Color32::BLACK,
            "leaving OLED's black in the dark slot would follow the system into a lie"
        );
    }
}
