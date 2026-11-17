//! Game controller input (Steam Deck controls, Xbox/PlayStation pads) via gilrs.
//! Missing permissions or no controller simply means no gamepad input.
use gilrs::{Axis, Button, EventType, Gilrs};

pub struct Gamepad {
    gilrs: Option<Gilrs>,
}

/// One frame of controller input: sticks after a dead zone, triggers 0..1, and the
/// buttons pressed since the last poll.
#[derive(Default)]
pub struct PadState {
    /// Left stick: x right, y forward.
    pub movement: [f32; 2],
    /// Right stick: x right, y up.
    pub look: [f32; 2],
    /// Right trigger minus left trigger (up / down).
    pub vertical: f32,
    pub pressed: Vec<Button>,
}

const DEAD_ZONE: f32 = 0.15;

fn dead_zone(v: f32) -> f32 {
    if v.abs() < DEAD_ZONE {
        0.
    } else {
        (v - DEAD_ZONE * v.signum()) / (1. - DEAD_ZONE)
    }
}

impl Gamepad {
    pub fn new() -> Self {
        Self {
            gilrs: Gilrs::new().ok(),
        }
    }

    pub fn poll(&mut self) -> PadState {
        let mut state = PadState::default();
        let Some(gilrs) = &mut self.gilrs else {
            return state;
        };
        while let Some(event) = gilrs.next_event() {
            if let EventType::ButtonPressed(button, _) = event.event {
                state.pressed.push(button);
            }
        }
        if let Some((_, pad)) = gilrs.gamepads().next() {
            let trigger = |button| pad.button_data(button).map_or(0., |d| d.value());
            state.movement = [
                dead_zone(pad.value(Axis::LeftStickX)),
                dead_zone(pad.value(Axis::LeftStickY)),
            ];
            state.look = [
                dead_zone(pad.value(Axis::RightStickX)),
                dead_zone(pad.value(Axis::RightStickY)),
            ];
            state.vertical = trigger(Button::RightTrigger2) - trigger(Button::LeftTrigger2);
        }
        state
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn dead_zone_is_continuous_and_reaches_full_range() {
        assert_eq!(super::dead_zone(0.1), 0.);
        assert!(super::dead_zone(0.16).abs() < 0.02);
        assert!((super::dead_zone(1.) - 1.).abs() < 1e-6);
        assert!((super::dead_zone(-1.) + 1.).abs() < 1e-6);
    }
}
