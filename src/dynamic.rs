//! Dynamic version of the hidden markov model, allowing one to create a model with state and observation spaces given at runtime. This is useful for applications where the number of states and observations is not known at compile time, or when the model needs to be constructed from external data sources.

use nalgebra::{DMatrix, DVector};
type PRE = f64;

struct HmmParamsD {
    transitions: DMatrix<PRE>,
    emissions: DMatrix<PRE>,
    initial_distribution: DVector<PRE>,
}

impl HmmParamsD {
    fn new(transitions: DMatrix<PRE>, emissions: DMatrix<PRE>, initial_distribution: DVector<PRE>) -> Self {
        // Here you would typically add checks to ensure that the matrices and vector are valid probability distributions.
        Self {
            transitions,
            emissions,
            initial_distribution,
        }
    }

    fn forward<'a>(
        &'a self,
        observations: &'a [usize],
    ) -> ForwardIter<'a> {
        ForwardIter {
            params: self,
            observations: observations.iter().copied(),
            prior: self.initial_distribution.clone(),
        }
    }

    pub fn filter(&self, observations: &[usize], target: usize) -> DVector<PRE> {
        if target >= observations.len() {
            panic!("Target index out of bounds");
        }

        self.forward(observations)
            .nth(target)
            .unwrap().0
    }

    pub fn sequence_log_likelihood(&self, observations: &[usize]) -> PRE {
        self.forward(observations)
            .map(|(_, p_y)| p_y.ln())
            .sum()
    }
    // Additional methods for filtering, smoothing, and Viterbi can be implemented here.
}

struct ForwardIter<'a> {
    params: &'a HmmParamsD,
    observations: std::iter::Copied<std::slice::Iter<'a, usize>>,
    prior: DVector<PRE>,
}

impl Iterator for ForwardIter<'_> {
    type Item = (DVector<PRE>, PRE);

    fn next(&mut self) -> Option<Self::Item> {
        let obs_t = self.observations.next()?;
        // Compute the forward probabilities for the current observation.
        let emission_obs_t = self.params.emissions.column(obs_t);
        let p_y = self.prior.dot(&emission_obs_t);
        let posterior = zero_prob_helper(p_y, self.prior.component_mul(&emission_obs_t));

        self.prior = self.params.transitions.tr_mul(&posterior);
        Some((posterior, p_y))
    }
}

// We often need to normalize distributions to avoid underflow, but if the marginal probability is zero, we return a zero vector to avoid division by zero.
fn zero_prob_helper(p: PRE, dist: DVector<PRE>) -> DVector<PRE> {
    if p == 0.0 {
        DVector::zeros(dist.len())
    } else {
        dist / p
    }
}
