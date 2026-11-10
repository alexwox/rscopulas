//! Golden-snapshot regression net over the whole numerical core.
//!
//! Every public numerical feature is evaluated on fixed inputs and compared
//! against `fixtures/golden/v1/*.json`:
//!
//! | test              | pins                                                       |
//! |-------------------|------------------------------------------------------------|
//! | `pair_kernels`    | log-density, both h-functions, both inverse h-functions for |
//! |                   | every family × parameter regime × rotation, tails included |
//! | `pair_fits`       | `fit_pair_copula` per family and with the default set       |
//! | `tll`             | TLL fits (all orders) and their kernels                     |
//! | `stats`           | Kendall τ, Spearman ρ, Hoeffding D, independence test       |
//! | `elliptical`      | Gaussian / Student t: log-density, sampling, fitting        |
//! | `archimedean`     | Clayton / Frank / Gumbel: log-density, sampling, fitting    |
//! | `hac`             | nested Archimedean: density, composite score, sample, fit   |
//! | `factor`          | one-factor copula: density, sampling, fitting               |
//! | `vine_evaluation` | log-density, sampling, Rosenblatt and its inverse on fixed  |
//! |                   | C-, D- and R-vines                                          |
//! | `vine_fits`       | C/D/R structure + family selection under every option       |
//! | `math`            | quadrature rules, linear algebra, optimisers                |
//!
//! Three kinds of comparison are used (see `support/golden.rs`):
//! closed-form results must match to ~1e-9 relative; solutions of inverse
//! problems may alternatively satisfy their defining equation to 1e-9; fitted
//! log-likelihoods are one-sided — a fit may get better, never worse.
//!
//! The snapshots were recorded before the performance work started, so they
//! are the definition of "no regression" for it.

#[path = "support/catalog.rs"]
mod catalog;
#[path = "support/datagen.rs"]
mod datagen;
#[path = "support/golden.rs"]
mod golden;

use ndarray::Array2;
use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    ClaytonCopula, CopulaModel, EvalOptions, FactorCopula, FactorFitOptions, FitOptions,
    FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFamily, HacFitOptions,
    HacStructureMethod, HierarchicalArchimedeanCopula, PairCopulaFamily, PairCopulaParams,
    PairCopulaSpec, PseudoObs, SampleOptions, SelectionCriterion, StudentTCopula, TllOrder,
    TreeAlgorithm, TreeCriterion, VineCopula, VineFitOptions, paircopula::fit_pair_copula, stats,
};

use catalog::to_rows;
use datagen::Dependence;
use golden::{Golden, Tol};

const CLIP: f64 = 1e-12;

fn obs(matrix: Array2<f64>) -> PseudoObs {
    PseudoObs::new(matrix).expect("generated data should be valid pseudo-observations")
}

fn or_nan<E>(result: Result<f64, E>) -> f64 {
    result.unwrap_or(f64::NAN)
}

/// Snapshots the five kernels of `spec` on `points` under `prefix`.
fn snapshot_pair_kernels(
    golden: &mut Golden,
    prefix: &str,
    spec: &PairCopulaSpec,
    points: &[(f64, f64)],
) {
    let log_pdf: Vec<f64> = points
        .iter()
        .map(|&(u, v)| or_nan(spec.log_pdf(u, v, CLIP)))
        .collect();
    let h12: Vec<f64> = points
        .iter()
        .map(|&(u, v)| or_nan(spec.cond_first_given_second(u, v, CLIP)))
        .collect();
    let h21: Vec<f64> = points
        .iter()
        .map(|&(u, v)| or_nan(spec.cond_second_given_first(u, v, CLIP)))
        .collect();
    let inv12: Vec<f64> = points
        .iter()
        .map(|&(p, v)| or_nan(spec.inv_first_given_second(p, v, CLIP)))
        .collect();
    let inv21: Vec<f64> = points
        .iter()
        .map(|&(u, p)| or_nan(spec.inv_second_given_first(u, p, CLIP)))
        .collect();

    golden.values(&format!("{prefix}/log_pdf"), &log_pdf, Tol::EVAL);
    golden.values(&format!("{prefix}/h_1given2"), &h12, Tol::EVAL);
    golden.values(&format!("{prefix}/h_2given1"), &h21, Tol::EVAL);
    // An inverse h-function is the root of h(x | v) = p. Where h is flat the
    // root is ill-conditioned, so a result is also accepted when it solves
    // that equation to 1e-9.
    golden.values_or(
        &format!("{prefix}/hinv_1given2"),
        &inv12,
        Tol::SOLVE,
        |i, x| {
            let (p, v) = points[i];
            spec.cond_first_given_second(x, v, CLIP)
                .is_ok_and(|h| (h - p.clamp(CLIP, 1.0 - CLIP)).abs() <= 1e-9)
        },
    );
    golden.values_or(
        &format!("{prefix}/hinv_2given1"),
        &inv21,
        Tol::SOLVE,
        |i, x| {
            let (u, p) = points[i];
            spec.cond_second_given_first(u, x, CLIP)
                .is_ok_and(|h| (h - p.clamp(CLIP, 1.0 - CLIP)).abs() <= 1e-9)
        },
    );
}

#[test]
fn pair_kernels() {
    let mut golden = Golden::open("pair_kernels");
    let points = catalog::pair_points();
    for (name, spec) in catalog::pair_specs() {
        snapshot_pair_kernels(&mut golden, &name, &spec, &points);
    }
    golden.finish();
}

fn describe_spec(spec: &PairCopulaSpec) -> String {
    match &spec.params {
        PairCopulaParams::Khoudraji(params) => format!(
            "Khoudraji/{:?}[{:?}/{:?}, {:?}/{:?}]",
            spec.rotation,
            params.first.family,
            params.first.rotation,
            params.second.family,
            params.second.rotation
        ),
        _ => format!("{:?}/{:?}", spec.family, spec.rotation),
    }
}

#[test]
fn pair_fits() {
    use PairCopulaFamily as F;
    let mut golden = Golden::open("pair_fits");
    let datasets = [
        ("gaussian", Dependence::Gaussian, 0x51),
        ("heavy_tail", Dependence::HeavyTail, 0x52),
        ("lower_tail", Dependence::LowerTail, 0x53),
        ("upper_tail", Dependence::UpperTail, 0x54),
        ("independent", Dependence::Independent, 0x55),
    ];
    let families = [
        F::Independence,
        F::Gaussian,
        F::StudentT,
        F::Clayton,
        F::Frank,
        F::Gumbel,
        F::Joe,
        F::Bb1,
        F::Bb6,
        F::Bb7,
        F::Bb8,
        F::Tawn1,
        F::Tawn2,
        F::Tll,
    ];
    for (data_name, dependence, seed) in datasets {
        let (u1, u2) = datagen::pair(300, dependence, seed);
        // Negatively dependent variant of the same sample exercises the
        // 90/270-degree rotations.
        let flipped: Vec<f64> = u1.iter().map(|u| 1.0 - u).collect();
        for (variant, first) in [("pos", &u1), ("neg", &flipped)] {
            for family in families {
                let options = VineFitOptions {
                    family_set: vec![family],
                    ..VineFitOptions::default()
                };
                let key = format!("{data_name}/{variant}/{family:?}");
                match fit_pair_copula(first, &u2, &options) {
                    Ok(fit) => {
                        golden.text(&format!("{key}/spec"), describe_spec(&fit.spec));
                        golden.at_least(&format!("{key}/loglik"), fit.loglik, Tol::LOGLIK);
                        if family != F::Tll {
                            // A likelihood ridge (e.g. BB8 near its Frank
                            // limit) leaves parameters weakly identified;
                            // the log-likelihood above is the contract.
                            golden.values_or(
                                &format!("{key}/params"),
                                &fit.spec.flat_parameters(),
                                Tol::FIT,
                                |_, _| true,
                            );
                        }
                    }
                    Err(error) => golden.text(&format!("{key}/spec"), format!("error: {error}")),
                }
            }
            for (label, criterion) in [
                ("aic", SelectionCriterion::Aic),
                ("bic", SelectionCriterion::Bic),
            ] {
                let options = VineFitOptions {
                    criterion,
                    ..VineFitOptions::default()
                };
                let key = format!("{data_name}/{variant}/default_{label}");
                let fit =
                    fit_pair_copula(first, &u2, &options).expect("default fit should succeed");
                golden.text(&format!("{key}/spec"), describe_spec(&fit.spec));
                golden.at_least(&format!("{key}/loglik"), fit.loglik, Tol::LOGLIK);
                golden.values(
                    &format!("{key}/h_2given1_head"),
                    &fit.cond_on_first[..8],
                    Tol::FIT,
                );
            }
        }
    }
    // Khoudraji is by far the most expensive fit; one dataset is enough to
    // pin it.
    let (u1, u2) = datagen::pair(150, Dependence::LowerTail, 0x56);
    let options = VineFitOptions {
        family_set: vec![F::Khoudraji],
        ..VineFitOptions::default()
    };
    let fit = fit_pair_copula(&u1, &u2, &options).expect("khoudraji fit should succeed");
    golden.text("khoudraji/spec", describe_spec(&fit.spec));
    golden.at_least("khoudraji/loglik", fit.loglik, Tol::LOGLIK);
    golden.finish();
}

#[test]
fn tll() {
    let mut golden = Golden::open("tll");
    let points = catalog::pair_points();
    for (name, order) in [
        ("constant", TllOrder::Constant),
        ("linear", TllOrder::Linear),
        ("quadratic", TllOrder::Quadratic),
    ] {
        let spec = catalog::tll_spec(order);
        let PairCopulaParams::Tll(params) = &spec.params else {
            unreachable!("tll_spec returns TLL parameters");
        };
        golden.values(
            &format!("{name}/log_density_grid"),
            &to_rows(params.log_density()),
            Tol::SOLVE,
        );
        golden.values(
            &format!("{name}/bandwidth_and_df"),
            &[params.bandwidth(), params.effective_df()],
            Tol::SOLVE,
        );
        snapshot_pair_kernels(&mut golden, name, &spec, &points);
    }
    golden.finish();
}

#[test]
fn stats() {
    let mut golden = Golden::open("stats");
    for (name, dependence) in [
        ("gaussian", Dependence::Gaussian),
        ("heavy_tail", Dependence::HeavyTail),
        ("lower_tail", Dependence::LowerTail),
    ] {
        let data = obs(datagen::pseudo_obs(257, 5, dependence, 0x57A7));
        let exec = rscopulas::ExecPolicy::Auto;
        golden.values(
            &format!("{name}/kendall_tau"),
            &to_rows(&stats::try_kendall_tau_matrix(&data, exec).unwrap()),
            Tol::EVAL,
        );
        golden.values(
            &format!("{name}/spearman_rho"),
            &to_rows(&stats::try_spearman_rho_matrix(&data, exec).unwrap()),
            Tol::EVAL,
        );
        golden.values(
            &format!("{name}/hoeffding_d"),
            &to_rows(&stats::try_hoeffding_d_matrix(&data, exec).unwrap()),
            Tol::EVAL,
        );
    }
    // Ties: raw (non rank-transformed) discretised inputs.
    let raw = datagen::uniforms(120, 2, 0x71E5);
    let x: Vec<f64> = raw.column(0).iter().map(|u| (u * 10.0).floor()).collect();
    let y: Vec<f64> = raw
        .column(1)
        .iter()
        .zip(&x)
        .map(|(u, x)| (u * 6.0 + x * 0.4).floor())
        .collect();
    golden.values(
        "ties/tau_rho_d",
        &[
            stats::kendall_tau_bivariate(&x, &y).unwrap(),
            stats::spearman_rho_bivariate(&x, &y).unwrap(),
            stats::hoeffding_d_bivariate(&x, &y).unwrap(),
        ],
        Tol::EVAL,
    );
    let statistics: Vec<f64> = [(0.1, 50usize), (-0.3, 200), (0.02, 1000)]
        .iter()
        .map(|&(tau, n)| stats::kendall_tau_test_statistic(tau, n))
        .collect();
    golden.values("independence_test/statistic", &statistics, Tol::EVAL);
    let rejections: Vec<f64> = [(0.1, 50usize, 0.05), (-0.3, 200, 0.01), (0.02, 1000, 0.05)]
        .iter()
        .map(|&(tau, n, alpha)| {
            f64::from(u8::from(stats::kendall_tau_rejects_independence(
                tau, n, alpha,
            )))
        })
        .collect();
    golden.values("independence_test/rejects", &rejections, Tol::EVAL);
    golden.finish();
}

/// Evaluation inputs for a `dim`-dimensional model: interior uniforms plus a
/// few rows pushed into the corners.
fn eval_points(dim: usize, seed: u64) -> PseudoObs {
    let mut matrix = datagen::uniforms(48, dim, seed);
    for (row, value) in [(0usize, 1e-9), (1, 1.0 - 1e-9), (2, 1e-4), (3, 0.9999)] {
        for col in 0..dim {
            matrix[(row, col)] = value;
        }
    }
    // Opposite corners in alternating columns.
    for col in 0..dim {
        matrix[(4, col)] = if col % 2 == 0 { 1e-5 } else { 1.0 - 1e-5 };
    }
    obs(matrix)
}

/// Relative size of the input nudge used to measure conditioning (45 ulps).
const NUDGE: f64 = 1e-14;

/// `points` with every entry moved by `factor` relative to its distance from
/// the nearer end of the unit interval.
fn nudged(points: &PseudoObs, factor: f64) -> PseudoObs {
    obs(points.as_view().mapv(|u| {
        if u < 0.5 {
            u * (1.0 + factor)
        } else {
            1.0 - (1.0 - u) * (1.0 + factor)
        }
    }))
}

/// Element-wise allowance for `f(points)`: four times the largest change
/// under a ±45-ulp nudge of the inputs. Zero wherever `f` is well
/// conditioned, so it only widens the comparison on rows whose result is
/// dominated by rounding noise in the inputs.
fn conditioning_slack(
    points: &PseudoObs,
    base: &[f64],
    f: impl Fn(&PseudoObs) -> Option<Vec<f64>>,
) -> Vec<f64> {
    let mut slack = vec![0.0_f64; base.len()];
    for factor in [NUDGE, -NUDGE] {
        let Some(moved) = f(&nudged(points, factor)) else {
            continue;
        };
        for ((slot, &a), &b) in slack.iter_mut().zip(base).zip(&moved) {
            let change = (a - b).abs();
            // The nudge itself moves a well-conditioned result by ~NUDGE
            // relative; only amplification beyond that counts.
            if change.is_finite() && change > 100.0 * NUDGE * (1.0 + a.abs()) {
                *slot = slot.max(4.0 * change);
            }
        }
    }
    slack
}

fn snapshot_model<M: CopulaModel>(golden: &mut Golden, prefix: &str, model: &M, seed: u64) {
    snapshot_model_with_noise(golden, prefix, model, seed, 0.0);
}

/// Rows with a coordinate this close to 1 count as upper-tail rows.
const UPPER_TAIL: f64 = 1.0 - 1e-3;

/// Like [`snapshot_model`], with an extra allowance of
/// `upper_tail_noise · (1 + |value|)` on rows that reach into the upper tail.
///
/// Needed for TLL edges only: beyond the last knot of its grid the kernel
/// locates a point through `1 − (accumulated margin CDF)`, a difference that
/// carries ~1e-7 relative rounding noise at `u = 1 − 1e-9`. The stored value
/// there is no more accurate than any other evaluation order, so those rows
/// are compared at the noise level instead of at 1e-9.
fn snapshot_model_with_noise<M: CopulaModel>(
    golden: &mut Golden,
    prefix: &str,
    model: &M,
    seed: u64,
    upper_tail_noise: f64,
) {
    let points = eval_points(model.dim(), seed);
    let options = EvalOptions::default();
    match model.log_pdf(&points, &options) {
        Ok(values) => {
            let mut slack = conditioning_slack(&points, &values, |moved| {
                model.log_pdf(moved, &options).ok()
            });
            for ((slot, value), row) in slack.iter_mut().zip(&values).zip(points.as_view().rows()) {
                if row.iter().any(|&u| u > UPPER_TAIL) {
                    *slot = slot.max(upper_tail_noise * (1.0 + value.abs()));
                }
            }
            golden.values_with_slack(&format!("{prefix}/log_pdf"), &values, Tol::EVAL, &slack);
        }
        Err(error) => golden.text(&format!("{prefix}/log_pdf"), format!("error: {error}")),
    }
    let mut rng = StdRng::seed_from_u64(seed);
    golden.result(
        &format!("{prefix}/sample"),
        model
            .sample(40, &mut rng, &SampleOptions::default())
            .map(|sample| to_rows(&sample)),
        Tol::SOLVE,
    );
}

#[test]
fn elliptical() {
    let mut golden = Golden::open("elliptical");
    for dim in [2usize, 5] {
        snapshot_model(
            &mut golden,
            &format!("gaussian/d{dim}"),
            &catalog::gaussian_copula(dim),
            11,
        );
        for nu in [2.5, 8.0] {
            snapshot_model(
                &mut golden,
                &format!("student_t/d{dim}/nu{nu}"),
                &catalog::student_t_copula(dim, nu),
                12,
            );
        }
    }
    for (name, dependence) in [
        ("gaussian", Dependence::Gaussian),
        ("heavy_tail", Dependence::HeavyTail),
    ] {
        let data = obs(datagen::pseudo_obs(400, 4, dependence, 0xE111));
        let fit = GaussianCopula::fit(&data, &FitOptions::default()).expect("gaussian fit");
        golden.values(
            &format!("fit/{name}/gaussian/correlation"),
            &to_rows(fit.model.correlation()),
            Tol::EVAL,
        );
        golden.values(
            &format!("fit/{name}/gaussian/diagnostics"),
            &[
                fit.diagnostics.loglik,
                fit.diagnostics.aic,
                fit.diagnostics.bic,
            ],
            Tol::EVAL,
        );
        let fit = StudentTCopula::fit(&data, &FitOptions::default()).expect("student t fit");
        golden.values(
            &format!("fit/{name}/student_t/correlation"),
            &to_rows(fit.model.correlation()),
            Tol::EVAL,
        );
        golden.values(
            &format!("fit/{name}/student_t/nu"),
            &[fit.model.degrees_of_freedom()],
            Tol::FIT,
        );
        golden.at_least(
            &format!("fit/{name}/student_t/loglik"),
            fit.diagnostics.loglik,
            Tol::LOGLIK,
        );
    }
    golden.finish();
}

#[test]
fn archimedean() {
    let mut golden = Golden::open("archimedean");
    for dim in [2usize, 3, 6] {
        snapshot_model(
            &mut golden,
            &format!("clayton/d{dim}"),
            &catalog::clayton_copula(dim),
            21,
        );
        snapshot_model(
            &mut golden,
            &format!("frank/d{dim}"),
            &catalog::frank_copula(dim),
            22,
        );
        snapshot_model(
            &mut golden,
            &format!("gumbel/d{dim}"),
            &catalog::gumbel_copula(dim),
            23,
        );
    }
    // Weak and strong dependence regimes.
    snapshot_model(
        &mut golden,
        "clayton/weak",
        &ClaytonCopula::new(3, 0.05).unwrap(),
        24,
    );
    snapshot_model(
        &mut golden,
        "clayton/strong",
        &ClaytonCopula::new(3, 12.0).unwrap(),
        24,
    );
    snapshot_model(
        &mut golden,
        "frank/weak",
        &FrankCopula::new(3, 0.05).unwrap(),
        25,
    );
    snapshot_model(
        &mut golden,
        "frank/strong",
        &FrankCopula::new(3, 30.0).unwrap(),
        25,
    );
    snapshot_model(
        &mut golden,
        "gumbel/independence",
        &GumbelHougaardCopula::new(3, 1.0).unwrap(),
        26,
    );
    snapshot_model(
        &mut golden,
        "gumbel/strong",
        &GumbelHougaardCopula::new(3, 9.0).unwrap(),
        26,
    );

    let data = obs(datagen::pseudo_obs(400, 4, Dependence::LowerTail, 0xA4C));
    let options = FitOptions::default();
    let fit = ClaytonCopula::fit(&data, &options).expect("clayton fit");
    golden.values(
        "fit/clayton",
        &[fit.model.theta(), fit.diagnostics.loglik],
        Tol::SOLVE,
    );
    let fit = FrankCopula::fit(&data, &options).expect("frank fit");
    golden.values(
        "fit/frank",
        &[fit.model.theta(), fit.diagnostics.loglik],
        Tol::SOLVE,
    );
    let fit = GumbelHougaardCopula::fit(&data, &options).expect("gumbel fit");
    golden.values(
        "fit/gumbel",
        &[fit.model.theta(), fit.diagnostics.loglik],
        Tol::SOLVE,
    );
    golden.finish();
}

#[test]
fn hac() {
    let mut golden = Golden::open("hac");
    for (name, model) in catalog::hac_models() {
        snapshot_model(&mut golden, name, &model, 31);
        golden.result(
            &format!("{name}/composite_log_pdf"),
            model.composite_log_pdf(&eval_points(4, 31), &EvalOptions::default()),
            Tol::EVAL,
        );
    }
    let data = obs(datagen::pseudo_obs(300, 4, Dependence::UpperTail, 0x4AC));
    for (name, options) in [
        ("default", HacFitOptions::default()),
        (
            "agglomerative_no_collapse",
            HacFitOptions {
                structure_method: HacStructureMethod::AgglomerativeTau,
                ..HacFitOptions::default()
            },
        ),
        (
            "gumbel_only",
            HacFitOptions {
                family_set: vec![HacFamily::Gumbel],
                ..HacFitOptions::default()
            },
        ),
    ] {
        let fit = HierarchicalArchimedeanCopula::fit(&data, &options).expect("hac fit");
        golden.text(
            &format!("fit/{name}/structure"),
            format!(
                "{:?} leaves {:?}",
                fit.model.families(),
                fit.model.leaf_order()
            ),
        );
        golden.values(
            &format!("fit/{name}/parameters"),
            &fit.model.parameters(),
            Tol::FIT,
        );
        golden.at_least(
            &format!("fit/{name}/loglik"),
            fit.diagnostics.loglik,
            Tol::LOGLIK,
        );
    }
    let options = HacFitOptions {
        structure_method: HacStructureMethod::GivenTree,
        family_set: vec![HacFamily::Gumbel],
        ..HacFitOptions::default()
    };
    let fit = HierarchicalArchimedeanCopula::fit_with_tree(
        &data,
        catalog::hac_tree(HacFamily::Gumbel, 1.2, 2.0),
        &options,
    )
    .expect("hac fit with tree");
    golden.values(
        "fit/given_tree/parameters",
        &fit.model.parameters(),
        Tol::FIT,
    );
    golden.at_least("fit/given_tree/loglik", fit.diagnostics.loglik, Tol::LOGLIK);
    golden.finish();
}

#[test]
fn factor() {
    let mut golden = Golden::open("factor");
    let model = catalog::factor_model();
    snapshot_model(&mut golden, "mixed", &model, 41);

    // Fit on data drawn from a three-variable model (kept small: the factor
    // fit integrates over the latent variable for every likelihood call).
    let truth = FactorCopula::basic_1f(
        vec![
            catalog::one(PairCopulaFamily::Gaussian, rscopulas::Rotation::R0, 0.7),
            catalog::one(PairCopulaFamily::Clayton, rscopulas::Rotation::R0, 1.5),
            catalog::one(PairCopulaFamily::Gumbel, rscopulas::Rotation::R0, 1.8),
        ],
        25,
    )
    .unwrap();
    let mut rng = StdRng::seed_from_u64(42);
    let sample = truth
        .sample(250, &mut rng, &SampleOptions::default())
        .unwrap();
    // Rank-transform so the fit input does not depend on sampler round-off.
    let mut ranked = Array2::zeros(sample.raw_dim());
    for col in 0..sample.ncols() {
        let column = sample.column(col).to_vec();
        for (row, value) in datagen::rank_transform(&column).into_iter().enumerate() {
            ranked[(row, col)] = value;
        }
    }
    let fit = FactorCopula::fit(&obs(ranked), &FactorFitOptions::default()).expect("factor fit");
    let links: Vec<String> = fit.model.links().iter().map(describe_spec).collect();
    golden.text("fit/links", links.join(", "));
    let params: Vec<f64> = fit
        .model
        .links()
        .iter()
        .flat_map(|link| link.flat_parameters())
        .collect();
    golden.values_or("fit/params", &params, Tol::FIT, |_, _| true);
    golden.at_least("fit/loglik", fit.diagnostics.loglik, Tol::LOGLIK);
    golden.finish();
}

/// Reloads a model from the snapshot so that evaluations are pinned on
/// exactly the stored parameters, independent of how the fitter evolves.
fn pinned_vine(golden: &mut Golden, key: &str, fitted: &VineCopula) -> VineCopula {
    let json = golden.pinned_text(key, || {
        serde_json::to_string(fitted).expect("vine should serialize")
    });
    serde_json::from_str(&json).expect("stored vine should deserialize")
}

fn snapshot_vine(golden: &mut Golden, prefix: &str, model: &VineCopula, seed: u64) {
    snapshot_vine_with_noise(golden, prefix, model, seed, 0.0);
}

fn snapshot_vine_with_noise(
    golden: &mut Golden,
    prefix: &str,
    model: &VineCopula,
    seed: u64,
    upper_tail_noise: f64,
) {
    snapshot_model_with_noise(golden, prefix, model, seed, upper_tail_noise);
    let dim = model.dim();
    let options = SampleOptions::default();
    let points = eval_points(dim, seed + 1);
    match model.rosenblatt(points.as_view(), &options) {
        Ok(uniforms) => {
            let values = to_rows(&uniforms);
            let slack = conditioning_slack(&points, &values, |moved| {
                model
                    .rosenblatt(moved.as_view(), &options)
                    .ok()
                    .map(|out| to_rows(&out))
            });
            golden.values_with_slack(&format!("{prefix}/rosenblatt"), &values, Tol::EVAL, &slack);
        }
        Err(error) => golden.text(&format!("{prefix}/rosenblatt"), format!("error: {error}")),
    }
    match model.rosenblatt_prefix(points.as_view(), dim - 1, &options) {
        Ok(uniforms) => {
            let values = to_rows(&uniforms);
            let slack = conditioning_slack(&points, &values, |moved| {
                model
                    .rosenblatt_prefix(moved.as_view(), dim - 1, &options)
                    .ok()
                    .map(|out| to_rows(&out))
            });
            golden.values_with_slack(
                &format!("{prefix}/rosenblatt_prefix"),
                &values,
                Tol::EVAL,
                &slack,
            );
        }
        Err(error) => golden.text(
            &format!("{prefix}/rosenblatt_prefix"),
            format!("error: {error}"),
        ),
    }
    let inverse = model.inverse_rosenblatt(points.as_view(), &options);
    match inverse {
        Ok(values) => {
            // Accept any inverse whose forward transform reproduces the input.
            let forward = model.rosenblatt(values.view(), &options).ok();
            golden.values_or(
                &format!("{prefix}/inverse_rosenblatt"),
                &to_rows(&values),
                Tol::SOLVE,
                |index, _| {
                    forward.as_ref().is_some_and(|forward| {
                        let (row, col) = (index / dim, index % dim);
                        // The whole row must round-trip for one of its
                        // entries to be rescued.
                        (0..dim).all(|c| {
                            (forward[(row, c)] - points.as_view()[(row, c)].clamp(CLIP, 1.0 - CLIP))
                                .abs()
                                <= 1e-9
                        }) && col < dim
                    })
                },
            );
        }
        Err(error) => golden.text(
            &format!("{prefix}/inverse_rosenblatt"),
            format!("error: {error}"),
        ),
    }
}

#[test]
fn vine_evaluation() {
    let mut golden = Golden::open("vine_evaluation");
    snapshot_vine(&mut golden, "mixed_d", &catalog::mixed_d_vine(), 51);
    snapshot_vine(&mut golden, "mixed_c", &catalog::mixed_c_vine(), 52);
    snapshot_vine(&mut golden, "gaussian_d7", &catalog::gaussian_d_vine(7), 53);
    let correlation = datagen::correlation(4);
    let gaussian_c = VineCopula::gaussian_c_vine(vec![3, 1, 0, 2], correlation).unwrap();
    snapshot_vine(&mut golden, "gaussian_c4", &gaussian_c, 54);

    // An R-vine with a non-trivial structure, fitted once and pinned.
    let data = obs(datagen::pseudo_obs(300, 6, Dependence::HeavyTail, 0x7E));
    let fitted = VineCopula::fit_r_vine(&data, &VineFitOptions::default()).expect("r-vine fit");
    let pinned = pinned_vine(&mut golden, "fitted_r/model_json", &fitted.model);
    snapshot_vine(&mut golden, "fitted_r", &pinned, 55);

    // Truncated vine and a vine with nonparametric (TLL) edges.
    let options = VineFitOptions {
        truncation_level: Some(2),
        family_set: vec![PairCopulaFamily::Tll, PairCopulaFamily::Gaussian],
        ..VineFitOptions::default()
    };
    let fitted = VineCopula::fit_r_vine(&data, &options).expect("truncated tll r-vine fit");
    let pinned = pinned_vine(&mut golden, "fitted_tll_trunc2/model_json", &fitted.model);
    snapshot_vine_with_noise(&mut golden, "fitted_tll_trunc2", &pinned, 56, 1e-6);
    golden.finish();
}

fn snapshot_vine_fit(
    golden: &mut Golden,
    key: &str,
    fit: Result<rscopulas::fit::FitResult<VineCopula>, rscopulas::CopulaError>,
) {
    let fit = match fit {
        Ok(fit) => fit,
        Err(error) => {
            golden.text(&format!("{key}/structure"), format!("error: {error}"));
            return;
        }
    };
    let mut structure = format!(
        "{:?} order {:?} trunc {:?}",
        fit.model.structure(),
        fit.model.order(),
        fit.model.truncation_level()
    );
    let mut params = Vec::new();
    for tree in fit.model.trees() {
        for edge in &tree.edges {
            structure.push_str(&format!(
                "\n  t{} ({},{}|{:?}) {}",
                tree.level,
                edge.conditioned.0,
                edge.conditioned.1,
                edge.conditioning,
                describe_spec(&edge.copula)
            ));
            if edge.copula.family != PairCopulaFamily::Tll {
                params.extend(edge.copula.flat_parameters());
            }
        }
    }
    golden.text(&format!("{key}/structure"), structure);
    golden.values_or(&format!("{key}/params"), &params, Tol::FIT, |_, _| true);
    golden.at_least(
        &format!("{key}/loglik"),
        fit.diagnostics.loglik,
        Tol::LOGLIK,
    );
}

#[test]
fn vine_fits() {
    use PairCopulaFamily as F;
    let mut golden = Golden::open("vine_fits");
    let default = VineFitOptions::default();
    let datasets = [
        ("gaussian", Dependence::Gaussian, 5usize),
        ("heavy_tail", Dependence::HeavyTail, 5),
        ("lower_tail", Dependence::LowerTail, 4),
    ];
    for (name, dependence, dim) in datasets {
        let data = obs(datagen::pseudo_obs(300, dim, dependence, 0xF17));
        snapshot_vine_fit(
            &mut golden,
            &format!("{name}/r"),
            VineCopula::fit_r_vine(&data, &default),
        );
        snapshot_vine_fit(
            &mut golden,
            &format!("{name}/c"),
            VineCopula::fit_c_vine(&data, &default),
        );
        snapshot_vine_fit(
            &mut golden,
            &format!("{name}/d"),
            VineCopula::fit_d_vine(&data, &default),
        );
    }

    let data = obs(datagen::pseudo_obs(300, 5, Dependence::HeavyTail, 0xF18));
    let order = [4usize, 2, 0, 1, 3];
    snapshot_vine_fit(
        &mut golden,
        "options/c_with_order",
        VineCopula::fit_c_vine_with_order(&data, &order, &default),
    );
    snapshot_vine_fit(
        &mut golden,
        "options/d_with_order",
        VineCopula::fit_d_vine_with_order(&data, &order, &default),
    );
    let variants: Vec<(&str, VineFitOptions)> = vec![
        (
            "bic",
            VineFitOptions {
                criterion: SelectionCriterion::Bic,
                ..default.clone()
            },
        ),
        (
            "mbicv_select_trunc",
            VineFitOptions {
                criterion: SelectionCriterion::Mbicv { psi0: 0.9 },
                select_trunc_lvl: true,
                ..default.clone()
            },
        ),
        (
            "truncation_2",
            VineFitOptions {
                truncation_level: Some(2),
                ..default.clone()
            },
        ),
        (
            "no_rotations",
            VineFitOptions {
                include_rotations: false,
                ..default.clone()
            },
        ),
        (
            "independence_threshold",
            VineFitOptions {
                independence_threshold: Some(0.1),
                ..default.clone()
            },
        ),
        (
            "independence_test",
            VineFitOptions {
                independence_test_level: Some(0.05),
                ..default.clone()
            },
        ),
        (
            "tree_rho",
            VineFitOptions {
                tree_criterion: TreeCriterion::Rho,
                ..default.clone()
            },
        ),
        (
            "tree_hoeffding",
            VineFitOptions {
                tree_criterion: TreeCriterion::Hoeffding,
                ..default.clone()
            },
        ),
        (
            "prim",
            VineFitOptions {
                tree_algorithm: TreeAlgorithm::Prim,
                ..default.clone()
            },
        ),
        (
            "random_weighted",
            VineFitOptions {
                tree_algorithm: TreeAlgorithm::RandomWeighted,
                rng_seed: Some(7),
                ..default.clone()
            },
        ),
        (
            "random_unweighted",
            VineFitOptions {
                tree_algorithm: TreeAlgorithm::RandomUnweighted,
                rng_seed: Some(7),
                ..default.clone()
            },
        ),
        (
            "gaussian_only",
            VineFitOptions {
                family_set: vec![F::Gaussian],
                ..default.clone()
            },
        ),
        (
            "extended_families",
            VineFitOptions {
                family_set: vec![
                    F::Independence,
                    F::Gaussian,
                    F::StudentT,
                    F::Clayton,
                    F::Frank,
                    F::Gumbel,
                    F::Joe,
                    F::Bb1,
                    F::Bb6,
                    F::Bb7,
                    F::Bb8,
                    F::Tawn1,
                    F::Tawn2,
                ],
                ..default.clone()
            },
        ),
        (
            "tll_only",
            VineFitOptions {
                family_set: vec![F::Tll],
                ..default.clone()
            },
        ),
    ];
    for (name, options) in variants {
        snapshot_vine_fit(
            &mut golden,
            &format!("options/{name}"),
            VineCopula::fit_r_vine(&data, &options),
        );
    }
    // Khoudraji edges on a small problem (the fit is expensive).
    let small = obs(datagen::pseudo_obs(120, 3, Dependence::UpperTail, 0xF19));
    let options = VineFitOptions {
        family_set: vec![F::Independence, F::Gaussian, F::Khoudraji],
        ..default.clone()
    };
    snapshot_vine_fit(
        &mut golden,
        "options/khoudraji",
        VineCopula::fit_r_vine(&small, &options),
    );
    golden.finish();
}

#[test]
fn math() {
    use rscopulas::math;
    let mut golden = Golden::open("math");
    for n in [3usize, 8, 16, 25, 64] {
        let (nodes, weights) = math::gauss_legendre_01(n);
        golden.values(&format!("gauss_legendre/{n}/nodes"), &nodes, Tol::EVAL);
        golden.values(&format!("gauss_legendre/{n}/weights"), &weights, Tol::EVAL);
    }
    let correlation = datagen::correlation(5);
    let lower = math::cholesky(&correlation).unwrap();
    golden.values("cholesky", &to_rows(&lower), Tol::EVAL);
    golden.values(
        "inverse",
        &to_rows(&math::inverse(&correlation).unwrap()),
        Tol::EVAL,
    );
    golden.values(
        "log_det_and_quadratic_form",
        &[
            math::log_determinant_from_cholesky(&lower),
            math::quadratic_form_from_cholesky(&lower, &[0.3, -1.2, 0.8, 2.0, -0.1]).unwrap(),
        ],
        Tol::EVAL,
    );
    let mut indefinite = correlation.clone();
    indefinite[(0, 1)] = 0.99;
    indefinite[(1, 0)] = 0.99;
    indefinite[(0, 2)] = -0.99;
    indefinite[(2, 0)] = -0.99;
    golden.values(
        "make_spd_correlation",
        &to_rows(&math::make_spd_correlation(&indefinite).unwrap()),
        Tol::EVAL,
    );
    let brent = math::maximize_scalar_brent(-3.0, 4.0, Some(0.5), 1e-10, 200, |x: f64| {
        -(x - 1.25).powi(2) + (3.0 * x).sin() * 0.1
    });
    golden.values("brent", &[brent.x, brent.value], Tol::SOLVE);
    let nelder_mead = math::nelder_mead_maximize(
        &[0.0, 0.0],
        &[(-5.0, 5.0), (-5.0, 5.0)],
        &math::NelderMeadOptions {
            initial_step: 0.25,
            ftol: 1e-12,
            xtol: 1e-9,
            max_iter: 2000,
            restarts: 2,
        },
        |x: &[f64]| -((x[0] - 1.5).powi(2) + 2.0 * (x[1] + 0.5).powi(2) + 0.6 * x[0] * x[1]),
    );
    golden.values(
        "nelder_mead/x",
        &nelder_mead.x,
        Tol {
            abs: 1e-6,
            rel: 1e-6,
        },
    );
    golden.finish();
}
