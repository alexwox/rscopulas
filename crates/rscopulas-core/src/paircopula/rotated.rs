use super::common::Rotation;

pub fn to_base_inputs(rotation: Rotation, u1: f64, u2: f64, clip_eps: f64) -> ((f64, f64), Rotation) {
    let transformed = match rotation {
        Rotation::R0 => (u1, u2),
        Rotation::R90 => (1.0 - u1, u2),
        Rotation::R180 => (1.0 - u1, 1.0 - u2),
        Rotation::R270 => (u1, 1.0 - u2),
    };
    (
        (
            transformed.0.clamp(clip_eps, 1.0 - clip_eps),
            transformed.1.clamp(clip_eps, 1.0 - clip_eps),
        ),
        rotation,
    )
}

pub fn from_base_log_pdf(_rotation: Rotation, log_pdf: f64) -> f64 {
    log_pdf
}

pub fn transform_sample(rotation: Rotation, u1: &[f64], u2: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let mut left = Vec::with_capacity(u1.len());
    let mut right = Vec::with_capacity(u2.len());
    for (&x, &y) in u1.iter().zip(u2.iter()) {
        let (tx, ty) = match rotation {
            Rotation::R0 => (x, y),
            Rotation::R90 => (1.0 - x, y),
            Rotation::R180 => (1.0 - x, 1.0 - y),
            Rotation::R270 => (x, 1.0 - y),
        };
        left.push(tx);
        right.push(ty);
    }
    (left, right)
}
