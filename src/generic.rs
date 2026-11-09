//! A hidden Markov model generic over its size: `HIDDEN_STATES` hidden states and
//! `OBS_N` observation symbols.

use std::array;

use nalgebra::{SMatrix, SVector};

type PRE = f64;

/// A hidden Markov model with `HIDDEN_STATES` hidden states and `OBS_N` observation
/// symbols. Build one with [`HmmParams::new`], which checks that every distribution
/// sums to 1.
pub struct HmmParams<const HIDDEN_STATES: usize, const OBS_N: usize> {
    initial_state: SVector<PRE, HIDDEN_STATES>,
    transitions: SMatrix<PRE, HIDDEN_STATES, HIDDEN_STATES>,
    emmisions: SMatrix<PRE, HIDDEN_STATES, OBS_N>,
}

impl<const HIDDEN_STATES: usize, const OBS_N: usize> HmmParams<HIDDEN_STATES, OBS_N> {
    /// Panics unless `initial_state` and every row of `transitions` and `emmisions`
    /// sums to 1 (within 1e-9).
    pub fn new(
        initial_state: SVector<PRE, HIDDEN_STATES>,
        transitions: SMatrix<PRE, HIDDEN_STATES, HIDDEN_STATES>,
        emmisions: SMatrix<PRE, HIDDEN_STATES, OBS_N>,
    ) -> Self {
        // Exact == 1.0 is too strict: 0.4 + 0.1 + 0.2 + 0.2 + 0.1 is 0.9999999999999999 in f64.
        let sums_to_one = |sum: PRE| (sum - 1.0).abs() <= 1e-9;
        assert!(sums_to_one(initial_state.sum()), "Initial state probabilities must sum to 1");
        assert!(transitions.row_iter().all(|row| sums_to_one(row.sum())), "Transition probabilities must sum to 1 for each state");
        assert!(emmisions.row_iter().all(|row| sums_to_one(row.sum())), "Emission probabilities must sum to 1 for each state");
        
        Self {
            initial_state,
            transitions,
            emmisions
        }
    }
    
    /// Returns a lazy iterator running the forward algorithm. It yields one item per
    /// observation: (P(x[t] | y[..=t]), P(y[t] | y[..t])) for t = 0, 1, ..., n-1.
    /// Once some y[t] is impossible, that step and all later ones yield (zeros, 0.0).
    fn forward<'a>(&'a self, observed_sequence: &'a [usize]) -> ForwardIter<'a, HIDDEN_STATES, OBS_N> {
        ForwardIter {
            params: self,
            observations: observed_sequence.iter().copied(),
            prior: self.initial_state,
        }
    }

    /// Returns a lazy iterator running the backward algorithm. It yields one item per
    /// observation, in reverse: β[t] ∝ P(y[t+1..] | x[t]) for t = n-1, n-2, ..., 0.
    /// Each β is normalised to avoid underflow, so only ratios between states are meaningful.
    pub fn backward<'a>(&'a self, observed_sequence: &'a [usize]) -> BackwardIter<'a, HIDDEN_STATES, OBS_N> {
        // An empty sequence has no states, so the iterator yields nothing.
        let rev_iter = observed_sequence.iter().copied().rev();
        let future = SVector::<PRE, HIDDEN_STATES>::from_element(1.0); // P(Yn+1 = yn+1 ... Yn = yn | Xn) = 1.0 for all states, since there are no future observations. 
        
        BackwardIter {
            params: self,
            observed_sequence: rev_iter,
            future
        }
    }

    /// P(x[target] | y[..=target]), where `target` is a 0-based index.
    pub fn filter(&self, observed_sequence: &[usize], target:usize) -> SVector<PRE, HIDDEN_STATES>  {
        if target >= observed_sequence.len() {
            panic!("target must be < sequence length")
        }

        // `nth` stops after target + 1 steps, so later observations are never processed.
        self.forward(observed_sequence).nth(target).unwrap().0
    }

    /// ln P(y). An empty sequence gives 0.0 (= ln 1); an impossible one gives -inf.
    pub fn sequence_log_likelihood(&self, observed_sequence: &[usize]) -> PRE {
        // ln P(y[0..n]) = Σ_t ln P(y[t] | y[0..t]); ln 0 = -inf, and the empty sum is 0.
        self.forward(observed_sequence)
            .map(|(_, p_y)| p_y.ln())
            .sum()
    }

    /// P(x[target] | y), using the whole sequence. `target` is a 0-based index.
    /// Returns zeros if the sequence is impossible.
    pub fn smooth(&self, observed_sequence: &[usize], target:usize) -> SVector<PRE, HIDDEN_STATES> {
        if target >= observed_sequence.len() {
            panic!("target must be < sequence length")
        }

        //Stores the probability dsitributions of Xt given the observations y0..yt.
        //Aswell as the cumulative probability of the observations y0..yt given these params.
        let mut forward = self.forward(observed_sequence);
        let (alpha_t, _) = forward.nth(target).unwrap();
        let beta_t = self.backward(observed_sequence).nth(observed_sequence.len() - target - 1).unwrap();
        let joint = alpha_t.component_mul(&beta_t);
        let total = joint.sum();
        if total == 0.0 {
            SVector::<PRE, HIDDEN_STATES>::zeros()
        } else {
            joint / total
        }
    }

    pub fn smooth_all(&self, observed_sequence: &[usize]) -> Vec<SVector<PRE, HIDDEN_STATES>> {
        let forward = self.forward(observed_sequence);
        let backward: Vec<SVector<PRE, HIDDEN_STATES>> = self.backward(observed_sequence).collect();
        let mut smoothed = Vec::with_capacity(observed_sequence.len());
        for ((alpha_t, _), beta_t) in forward.zip(backward.iter().rev()) {
            let joint = alpha_t.component_mul(&beta_t);
            let total = joint.sum();
            if total == 0.0 {
                smoothed.push(SVector::<PRE, HIDDEN_STATES>::zeros());
            } else {
                smoothed.push(joint / total);
            }
        }
        smoothed
    }

    pub fn viterbi(&self, observed_sequence: &[usize]) -> Option<Vec<usize>> {
        let HmmParams { initial_state, transitions, emmisions } = self;
        let n = observed_sequence.len();
        if n == 0 {
            return Some(vec![]);
        }

        let mut obs_iter = observed_sequence.iter().copied();
        let obs = obs_iter.next().unwrap();

        // We are doing maximums over the states, so using nalgebra no longer makes sense.
        let mut p: [PRE; HIDDEN_STATES] = initial_state.component_mul(&emmisions.column(obs)).into();
        let mut sequences = vec![[0; HIDDEN_STATES]];

        while let Some(obs) = obs_iter.next() {
            let mut next_in_sequence: [usize; HIDDEN_STATES] = [0; HIDDEN_STATES];
            let mut next_p = [0.0; HIDDEN_STATES];
            for s in 0..HIDDEN_STATES {
                // Gets the maximum probability of ending up at state s out all possible sequences leading up to it,
                // aswell as the last state of this maximum probability sequence.
                let (max, argmax) = (0..HIDDEN_STATES).map(|r| {
                    (p[r] * transitions.get((r,s)).unwrap() * emmisions.get((s,obs)).unwrap(),r)
                }).reduce(|m, r| {
                    if r.0 > m.0 {
                        r
                    } else {
                        m
                    }
                }).unwrap();
                
                next_p[s] = max;
                next_in_sequence[s] = argmax;
            };

            p = next_p;
            sequences.push(next_in_sequence);
        };

        let mut next_backwards = p.iter().enumerate().reduce(|m, r| {
            if r.1 > m.1 {
                r
            } else {
                m
            }
        }).unwrap().0;

        let mut most_likely_sequence = vec![0; n];
        for t in (0..n).rev() {
            most_likely_sequence[t] = next_backwards;
            next_backwards = sequences[t][next_backwards];
        }
        Some(most_likely_sequence)
    }

    /// P(y | x): the probability of the observations given a known state sequence.
    pub fn sequence_probability_given_x(&self, observed_sequence: &[usize], state_sequence: &[usize]) -> PRE {
        let n = observed_sequence.len();
        assert_eq!(state_sequence.len(), n);

        // Emissions probability
        state_sequence.iter()
                .zip(observed_sequence.iter())
                .map(|(&state, &observation)| {
                    self.emmisions[(state, observation)]
                })
                .product::<PRE>()
    }
}

/// Iterator returned by [`HmmParams::forward`].
pub struct ForwardIter<'a, const HIDDEN_STATES: usize, const OBS_N: usize> {
    params: &'a HmmParams<HIDDEN_STATES, OBS_N>,
    observations: std::iter::Copied<std::slice::Iter<'a, usize>>,
    // P(x[t] | y[0..t]): the belief about the next state before seeing y[t].
    // Before any observation this is just the initial distribution.
    prior: SVector<PRE, HIDDEN_STATES>,
}

impl<const HIDDEN_STATES: usize, const OBS_N: usize> Iterator for ForwardIter<'_, HIDDEN_STATES, OBS_N> {
    type Item = (SVector<PRE, HIDDEN_STATES>, PRE);

    fn next(&mut self) -> Option<Self::Item> {
        let obs_t = self.observations.next()?;
        let emissions_obs_t = self.params.emmisions.column(obs_t);

        // P(y[t] | y[0..t]) = Σ_j P(x[t]=j | y[0..t]) · P(y[t] | x[t]=j)
        let p_y = self.prior.dot(&emissions_obs_t);
        let posterior = if p_y == 0.0 {
            // y[t] is impossible given y[0..t], so the posterior is undefined.
            // Zeros mark that. The next prior becomes zeros too, so every later
            // step also yields (zeros, 0.0) without needing a special case.
            SVector::zeros()
        } else {
            self.prior.component_mul(&emissions_obs_t) / p_y
        };
        // Predict P(x[t+1] | y[0..=t]) ready for the next call.
        self.prior = self.params.transitions.tr_mul(&posterior);

        Some((posterior, p_y))
    }
}


/// Iterator returned by [`HmmParams::backward`].
pub struct BackwardIter<'a, const HIDDEN_STATES: usize, const OBS_N: usize> {
    params: &'a HmmParams<HIDDEN_STATES, OBS_N>,
    observed_sequence: std::iter::Rev<std::iter::Copied<std::slice::Iter<'a, usize>>>,
    // P(y[t+1..n] | x[t + 1]): the belief about the future observations after seeing y[t].
    // After the last observation this is just 1.0 for every state.
    future: SVector<PRE, HIDDEN_STATES>,
}

impl<const HIDDEN_STATES: usize, const OBS_N: usize> Iterator for BackwardIter<'_, HIDDEN_STATES, OBS_N> {
    type Item = SVector<PRE, HIDDEN_STATES>;

    fn next(&mut self) -> Option<Self::Item> {
        // At this point we ahve have the future P(y[t+1..n] | x[t+1])
        let y_t = self.observed_sequence.next()?;

        // P(y[t..n] | x[t]) = P(y[t] | x[t]) · P(y[t+1..n] | x[t])
        let emissions_y_t = self.params.emmisions.column(y_t);
        let future_over_transition = self.params.transitions * self.future;

        let msg = emissions_y_t.component_mul(&future_over_transition);
        let scale = msg.sum();
        self.future = if scale == 0.0 {
            SVector::zeros()
        } else {
            msg / scale
        };

        Some(future_over_transition)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // =================================================================================
    // Shared helpers
    // =================================================================================

    // The main test model: 5 hidden states and 6 symbols, with arbitrary probabilities.
    // Symbol 5 has emission probability 0 in every state, so any sequence containing it
    // is impossible.
    const N: usize = 5;
    const M: usize = 6;

    fn params() -> HmmParams<N, M> {
        HmmParams {
            initial_state: SVector::from_row_slice(&[0.1, 0.2, 0.3, 0.15, 0.25]),
            transitions: SMatrix::from_row_slice(&[
                0.1, 0.2, 0.3, 0.15, 0.25,
                0.4, 0.1, 0.2, 0.2, 0.1,
                0.2, 0.25, 0.15, 0.3, 0.1,
                0.3, 0.1, 0.2, 0.15, 0.25,
                0.15, 0.2, 0.25, 0.3, 0.1,
            ]),
            emmisions: SMatrix::from_row_slice(&[
                0.1, 0.15, 0.2, 0.25, 0.3, 0.0,
                0.3, 0.25, 0.2, 0.15, 0.1, 0.0,
                0.05, 0.1, 0.15, 0.3, 0.4, 0.0,
                0.4, 0.1, 0.15, 0.05, 0.3, 0.0,
                0.2, 0.1, 0.3, 0.15, 0.25, 0.0,
            ]),
        }
    }

    // Short sequences used by the brute-force comparisons.
    const SHORT_SEQUENCES: [&[usize]; 4] = [&[0], &[2, 0, 4], &[4, 3, 1, 1], &[1, 2, 3, 4, 0]];

    fn assert_probability_close(actual: PRE, expected: PRE) {
        assert!(
            (actual - expected).abs() <= 1e-12,
            "expected {expected}, got {actual}"
        );
    }

    // Log-likelihoods are compared with a looser tolerance: f64 has ~16 significant
    // digits, but rounding accumulates over every step of a sequence.
    fn assert_log_close(actual: PRE, expected: PRE) {
        assert!(
            (actual - expected).abs() <= 1e-9,
            "expected {expected}, got {actual}"
        );
    }

    // Every hidden state path of length n: all N^n of them. The brute-force references
    // below are built on this, so they share no code with the algorithms under test.
    fn all_paths(n: usize) -> impl Iterator<Item = Vec<usize>> {
        (0..N.pow(n as u32)).map(move |mut code| {
            (0..n).map(|_| { let s = code % N; code /= N; s }).collect()
        })
    }

    // ln P(x, y): log of the joint probability of a state path x and the observations y.
    // Computed in log space so it stays finite on long sequences.
    fn log_joint<const H: usize, const O: usize>(
        params: &HmmParams<H, O>,
        states: &[usize],
        observations: &[usize],
    ) -> PRE {
        let mut log_p = params.initial_state[states[0]].ln();
        for t in 1..states.len() {
            log_p += params.transitions[(states[t - 1], states[t])].ln();
        }
        for (&x, &y) in states.iter().zip(observations) {
            log_p += params.emmisions[(x, y)].ln();
        }
        log_p
    }

    // =================================================================================
    // Model construction and P(y | x)
    // =================================================================================

    #[test]
    #[should_panic(expected = "Transition probabilities must sum to 1 for each state")]
    fn new_rejects_rows_that_do_not_sum_to_one() {
        // The first transition row sums to 0.9.
        HmmParams::<2, 3>::new(
            SVector::from_row_slice(&[0.5, 0.5]),
            SMatrix::from_row_slice(&[
                0.7, 0.2,
                0.3, 0.7,
            ]),
            SMatrix::from_row_slice(&[
                0.2, 0.3, 0.5,
                0.6, 0.3, 0.1,
            ]),
        );
    }

    #[test]
    fn probability_given_states_multiplies_emissions_only() {
        // P(y | x) is the product of e_x[t](y[t]). Initial and transition probabilities
        // play no part, which the second half checks by changing them.
        let mut params = params();
        let cases: &[(&[usize], &[usize], PRE)] = &[
            (&[], &[], 1.0),
            (&[4], &[3], 0.3),
            (&[2, 0, 4], &[4, 1, 3], 0.027),
            (&[4, 3, 1], &[2, 2, 2], 0.012),
            (&[2, 5, 4], &[4, 1, 3], 0.0),
        ];
        for &(observations, states, expected) in cases {
            assert_probability_close(params.sequence_probability_given_x(observations, states), expected);
        }

        params.initial_state = SVector::from_row_slice(&[1.0, 0.0, 0.0, 0.0, 0.0]);
        params.transitions = SMatrix::identity();
        assert_probability_close(params.sequence_probability_given_x(&[2, 0, 4], &[4, 1, 3]), 0.027);
    }

    #[test]
    #[should_panic(expected = "assertion `left == right` failed")]
    fn probability_given_states_rejects_mismatched_lengths() {
        params().sequence_probability_given_x(&[0, 1], &[0]);
    }

    // =================================================================================
    // Forward, backward, filter, smooth and the likelihood
    // =================================================================================

    #[test]
    fn first_filtered_distribution_is_normalized_initial_times_emission() {
        // At t = 0, P(x[0] | y[0]) ∝ initial_state ⊙ e(y[0]). For y[0] = 0 the joint
        // weights are [0.01, 0.06, 0.015, 0.06, 0.05], which sum to P(y[0]) = 0.195.
        let mut params = params();
        let (posterior, p_y) = params.forward(&[0]).next().unwrap();
        for (actual, numerator) in posterior.iter().zip([2.0, 12.0, 3.0, 12.0, 10.0]) {
            assert_probability_close(*actual, numerator / 39.0);
        }
        assert_probability_close(p_y, 0.195);

        // Transitions and later observations can't affect the first step.
        params.transitions = SMatrix::identity();
        assert_eq!(params.forward(&[0, 4, 2]).next().unwrap().0, posterior);
    }

    #[test]
    fn sequence_log_likelihood_matches_brute_force() {
        // P(y) = Σ over every path x of P(x, y).
        let params = params();
        for observations in SHORT_SEQUENCES {
            let brute_force: PRE = all_paths(observations.len())
                .map(|x| log_joint(&params, &x, observations).exp())
                .sum();
            assert_log_close(params.sequence_log_likelihood(observations), brute_force.ln());
        }
    }

    #[test]
    fn smooth_matches_brute_force() {
        // P(x[t] = s | y) = Σ of P(x, y) over the paths with x[t] = s, divided by P(y).
        let params = params();
        for observations in SHORT_SEQUENCES {
            for t in 0..observations.len() {
                let mut by_state = [0.0; N];
                for x in all_paths(observations.len()) {
                    by_state[x[t]] += log_joint(&params, &x, observations).exp();
                }
                let total: PRE = by_state.iter().sum();

                let smoothed = params.smooth(observations, t);
                for (actual, expected) in smoothed.iter().zip(by_state) {
                    assert_probability_close(*actual, expected / total);
                }
                // smooth_all computes every t in one go and must agree with smooth(t).
                assert_eq!(params.smooth_all(observations)[t], smoothed);
            }
        }
    }

    #[test]
    fn filter_is_zero_based_and_equals_smooth_at_the_last_step() {
        let params = params();
        let observations = [2, 0, 4, 1];
        for t in 0..observations.len() {
            // filter(t) is the t-th forward step, counting from 0.
            let filtered = params.filter(&observations, t);
            assert_eq!(filtered, params.forward(&observations).nth(t).unwrap().0);
            assert_probability_close(filtered.sum(), 1.0);
        }

        // At the last step there's no future left, so β is all ones and smoothing adds
        // nothing to filtering.
        let last = observations.len() - 1;
        let first_beta = params.backward(&observations).next().unwrap();
        for b in first_beta.iter() {
            // Approximate: β is computed as transitions · ones, and some rows of the
            // transition matrix sum to 0.9999999999999999 in f64.
            assert_probability_close(*b, 1.0);
        }
        let (smoothed, filtered) = (params.smooth(&observations, last), params.filter(&observations, last));
        for (s, f) in smoothed.iter().zip(filtered.iter()) {
            assert_probability_close(*s, *f);
        }
    }

    #[test]
    #[should_panic(expected = "target must be < sequence length")]
    fn filter_rejects_out_of_range_index() {
        params().filter(&[0, 1], 2);
    }

    #[test]
    #[should_panic(expected = "target must be < sequence length")]
    fn smooth_rejects_out_of_range_index() {
        params().smooth(&[0, 1], 2);
    }

    #[test]
    fn empty_sequence() {
        // No observations: P(empty) = 1, so its log is 0; both passes yield nothing;
        // the most likely explanation is the empty path.
        let params = params();
        assert_eq!(params.sequence_log_likelihood(&[]), 0.0);
        assert_eq!(params.forward(&[]).count(), 0);
        assert_eq!(params.backward(&[]).count(), 0);
        assert_eq!(params.viterbi(&[]), Some(vec![]));
    }

    #[test]
    fn impossible_sequence() {
        // Symbol 5 is never emitted, so [0, 5, 1] has probability 0.
        let params = params();
        assert_eq!(params.sequence_log_likelihood(&[5]), PRE::NEG_INFINITY);
        assert_eq!(params.sequence_log_likelihood(&[0, 5, 1]), PRE::NEG_INFINITY);

        // Forward still yields one item per observation. From the impossible step on,
        // each is (zeros, 0.0).
        let steps: Vec<_> = params.forward(&[0, 5, 1]).collect();
        assert_eq!(steps.len(), 3);
        assert!(steps[0].1 > 0.0);
        for (posterior, p_y) in &steps[1..] {
            assert_eq!(*posterior, SVector::<PRE, N>::zeros());
            assert_eq!(*p_y, 0.0);
        }

        // Smoothing has no distribution to return at any t, before or after the
        // impossible observation.
        for t in 0..3 {
            assert_eq!(params.smooth(&[0, 5, 1], t), SVector::<PRE, N>::zeros());
        }
    }

    #[test]
    fn long_sequences_do_not_underflow() {
        // P(y) here is around e^-1600, far below f64's smallest positive value (about
        // e^-744). The forward pass and the normalised backward pass must stay finite.
        let params = params();
        let observations: Vec<usize> = (0..1000).map(|t| t % 5).collect();

        let log_p = params.sequence_log_likelihood(&observations);
        assert!(log_p.is_finite() && log_p < -100.0, "got {log_p}");

        for t in [0, 10, 500, 999] {
            let smoothed = params.smooth(&observations, t);
            assert!(smoothed.iter().all(|p| p.is_finite()), "t={t}: {smoothed:?}");
            assert_probability_close(smoothed.sum(), 1.0);
        }
    }

    // =================================================================================
    // Viterbi: the single most likely sequence of hidden states
    // =================================================================================

    #[test]
    fn viterbi_matches_brute_force() {
        // The returned path must be as likely as the best of all N^n paths. Compared by
        // probability rather than by path, so an exact tie can't make the test flaky.
        let params = params();
        for observations in SHORT_SEQUENCES {
            let path = params.viterbi(observations).unwrap();
            assert_eq!(path.len(), observations.len());

            let best = all_paths(observations.len())
                .map(|x| log_joint(&params, &x, observations))
                .fold(PRE::NEG_INFINITY, PRE::max);
            assert_log_close(log_joint(&params, &path, observations), best);
        }
    }

    #[test]
    fn viterbi_is_not_the_most_likely_state_at_each_step() {
        // Taking argmax of smooth() at every t picks each step on its own. That answers
        // a different question from "which whole path is most likely". For y = [1, 1]
        // the step-by-step choice is [1, 1] but the best path is [1, 0]:
        //   ln P([1, 0], y) ≈ -5.81  >  ln P([1, 1], y) ≈ -6.68
        let params = params();
        let observations = [1, 1];

        let step_by_step: Vec<usize> = (0..observations.len())
            .map(|t| params.smooth(&observations, t).argmax().0)
            .collect();
        let path = params.viterbi(&observations).unwrap();

        assert_eq!(step_by_step, vec![1, 1]);
        assert_eq!(path, vec![1, 0]);
        assert!(log_joint(&params, &path, &observations) > log_joint(&params, &step_by_step, &observations));
    }

    // =================================================================================
    // Textbook examples with known answers
    // =================================================================================

    // ---------------------------------------------------------------------------------
    // Umbrella world (Russell & Norvig, "AIMA", ch. 15).
    //
    // A security guard works underground and never sees the sky. Each morning the
    // director either does or doesn't come in carrying an umbrella, and the guard wants
    // to infer whether it is raining. Weather is sticky: today's weather matches
    // yesterday's 70% of the time. Rain usually means an umbrella (90%); a dry day
    // occasionally does too (20%).
    // ---------------------------------------------------------------------------------

    // Hidden states.
    const RAIN: usize = 0;
    const DRY: usize = 1;
    // Observations.
    const UMBRELLA: usize = 0;
    const NO_UMBRELLA: usize = 1;

    fn umbrella_world() -> HmmParams<2, 2> {
        HmmParams::new(
            //                          RAIN  DRY
            SVector::from_row_slice(&[0.5, 0.5]),
            SMatrix::from_row_slice(&[
                // to: RAIN DRY
                0.7, 0.3, // from RAIN
                0.3, 0.7, // from DRY
            ]),
            SMatrix::from_row_slice(&[
                // UMBRELLA NO_UMBRELLA
                0.9, 0.1, // RAIN
                0.2, 0.8, // DRY
            ]),
        )
    }

    #[test]
    fn umbrella_world_matches_textbook() {
        let hmm = umbrella_world();
        let close = |actual: PRE, expected: PRE| {
            assert!((actual - expected).abs() < 5e-4, "expected {expected}, got {actual}")
        };

        // AIMA §15.2: an umbrella on days 1 and 2 gives filtered P(rain) of 0.818 then
        // 0.883. Smoothed P(rain on day 1) is also 0.883: seeing the umbrella again on
        // day 2 makes rain on day 1 more likely in hindsight.
        let days = [UMBRELLA, UMBRELLA];
        close(hmm.filter(&days, 0)[RAIN], 0.818);
        close(hmm.filter(&days, 1)[RAIN], 0.883);
        close(hmm.smooth(&days, 0)[RAIN], 0.883);

        // AIMA figure 15.5: umbrella, umbrella, none, umbrella, umbrella. The most likely
        // explanation is rain every day except the third. One day without an umbrella
        // is enough to switch, because the umbrella evidence (0.9 vs 0.2) outweighs
        // the weather's 70% stickiness.
        let days = [UMBRELLA, UMBRELLA, NO_UMBRELLA, UMBRELLA, UMBRELLA];
        assert_eq!(hmm.viterbi(&days), Some(vec![RAIN, RAIN, DRY, RAIN, RAIN]));
    }

    // ---------------------------------------------------------------------------------
    // Rainy or sunny, from what a friend did each day (the classic Wikipedia Viterbi
    // example). A friend who lives far away tells you what they did each day: went
    // for a walk, went shopping, or cleaned the flat. What they do depends on the
    // weather where they are, which you can't see.
    // ---------------------------------------------------------------------------------

    const RAINY: usize = 0;
    const SUNNY: usize = 1;
    const WALK: usize = 0;
    const SHOP: usize = 1;
    const CLEAN: usize = 2;

    fn rainy_or_sunny() -> HmmParams<2, 3> {
        HmmParams::new(
            //                          RAINY SUNNY
            SVector::from_row_slice(&[0.6, 0.4]),
            SMatrix::from_row_slice(&[
                // to: RAINY SUNNY
                0.7, 0.3, // from RAINY
                0.4, 0.6, // from SUNNY
            ]),
            SMatrix::from_row_slice(&[
                // WALK SHOP CLEAN
                0.1, 0.4, 0.5, // RAINY: mostly indoor activities
                0.6, 0.3, 0.1, // SUNNY: mostly walks
            ]),
        )
    }

    #[test]
    fn rainy_or_sunny_viterbi_matches_known_answer() {
        // Walk, then shop, then clean: the most likely explanation is a sunny day
        // followed by two rainy ones, with joint probability
        //   0.4·0.6 (sunny, walk) · 0.4·0.4 (→ rainy, shop) · 0.7·0.5 (→ rainy, clean) = 0.01344.
        let hmm = rainy_or_sunny();
        let activities = [WALK, SHOP, CLEAN];
        let path = hmm.viterbi(&activities).unwrap();

        assert_eq!(path, vec![SUNNY, RAINY, RAINY]);
        assert_probability_close(log_joint(&hmm, &path, &activities).exp(), 0.01344);
    }

    // =================================================================================
    // Tests marked #[ignore] fail with the current viterbi, on purpose. Run them with
    //     cargo test -- --ignored
    // and delete the #[ignore] line from each one once it passes.
    // =================================================================================

    #[test]
    #[ignore = "fails until viterbi returns None when no path can explain the sequence"]
    fn viterbi_returns_none_for_impossible_sequence() {
        // Symbol 5 has emission probability 0 in every state, so every path has
        // P(x, y) = 0. There is no most likely path, and returning one (e.g. all
        // zeros) would look like a real answer.
        let params = params();
        assert_eq!(params.viterbi(&[5]), None);
        assert_eq!(params.viterbi(&[0, 5, 1]), None);
    }

    #[test]
    #[ignore = "fails until viterbi avoids underflow (e.g. works in log space)"]
    fn viterbi_beats_step_by_step_choice_on_long_sequences() {
        // Viterbi's path is the most likely of all paths, so it can never be less likely
        // than any particular path, such as the step-by-step argmax of smooth() or
        // filter(). P(x, y) here is around e^-2600, which underflows f64. Once every δ
        // is 0, all back-pointers point at state 0 and the returned path is far less
        // likely than either of these.
        let params = params();
        let observations: Vec<usize> = (0..1000).map(|t| t % 5).collect();
        let path = params
            .viterbi(&observations)
            .expect("this sequence is possible, so a path must exist");
        let best = log_joint(&params, &path, &observations);
        assert!(best.is_finite(), "returned path has probability 0: {path:?}");

        for (name, rival) in [
            ("smooth", (0..1000).map(|t| params.smooth(&observations, t).argmax().0).collect::<Vec<_>>()),
            ("filter", (0..1000).map(|t| params.filter(&observations, t).argmax().0).collect::<Vec<_>>()),
        ] {
            let rival_log_p = log_joint(&params, &rival, &observations);
            assert!(
                best >= rival_log_p - 1e-9,
                "viterbi's path (ln P = {best}) is less likely than the step-by-step {name} path (ln P = {rival_log_p})",
            );
        }
    }

    #[test]
    #[ignore = "fails until viterbi avoids underflow (e.g. works in log space)"]
    fn umbrella_world_viterbi_survives_long_spells_of_weather() {
        // 2000 days of umbrellas, then 2000 days without. The answer is obvious: rain
        // for the first spell, dry for the second, one switch in the middle.
        //
        // Each rainy day multiplies δ by about 0.63, so δ drops below f64's smallest
        // value after ~1600 days. It sticks at the smallest subnormal (4.9e-324),
        // because 0.63 · 4.9e-324 rounds back up. Then the first dry day multiplies it
        // by 0.3 · 0.8 = 0.24, which rounds to exactly 0. From then on every δ is 0 and
        // the back-pointers stop meaning anything.
        let days: Vec<usize> = [vec![UMBRELLA; 2000], vec![NO_UMBRELLA; 2000]].concat();
        let expected: Vec<usize> = [vec![RAIN; 2000], vec![DRY; 2000]].concat();

        let path = umbrella_world()
            .viterbi(&days)
            .expect("this sequence is possible, so a path must exist");
        let first_wrong = path.iter().zip(&expected).position(|(a, b)| a != b);
        assert_eq!(first_wrong, None, "path is wrong from day {first_wrong:?} on");
    }
}
