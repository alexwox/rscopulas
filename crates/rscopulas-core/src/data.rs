use ndarray::{Array2, ArrayView2};

use crate::errors::InputError;

#[derive(Debug, Clone)]
pub struct PseudoObs {
    values: Array2<f64>,
}

impl PseudoObs {
    pub fn new(values: Array2<f64>) -> Result<Self, InputError> {
        validate_pseudo_obs(values.view())?;
        Ok(Self { values })
    }

    pub fn from_view(values: ArrayView2<'_, f64>) -> Result<Self, InputError> {
        validate_pseudo_obs(values)?;
        Ok(Self {
            values: values.to_owned(),
        })
    }

    pub fn n_obs(&self) -> usize {
        self.values.nrows()
    }

    pub fn dim(&self) -> usize {
        self.values.ncols()
    }

    pub fn as_view(&self) -> ArrayView2<'_, f64> {
        self.values.view()
    }

    pub fn into_inner(self) -> Array2<f64> {
        self.values
    }
}

fn validate_pseudo_obs(values: ArrayView2<'_, f64>) -> Result<(), InputError> {
    if values.nrows() == 0 {
        return Err(InputError::EmptyObservations);
    }

    if values.ncols() < 2 {
        return Err(InputError::DimensionTooSmall(values.ncols()));
    }

    for ((row, col), value) in values.indexed_iter() {
        if !value.is_finite() {
            return Err(InputError::NonFiniteValue);
        }

        if !(0.0 < *value && *value < 1.0) {
            return Err(InputError::OutOfUnitInterval {
                row,
                col,
                value: *value,
            });
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use ndarray::array;

    use crate::errors::InputError;

    use super::PseudoObs;

    #[test]
    fn rejects_values_outside_open_unit_interval() {
        let error =
            PseudoObs::new(array![[0.0, 0.5], [0.2, 0.8]]).expect_err("zero should be rejected");

        assert!(matches!(
            error,
            InputError::OutOfUnitInterval { row: 0, col: 0, .. }
        ));
    }
}
