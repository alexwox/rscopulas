//! Timing suite over every numerical feature of the crate.
//!
//! ```text
//! cargo run --release -p rscopulas --example perf_suite -- \
//!     [--filter SUBSTR] [--save out.json] [--compare baseline.json] [--quick]
//! ```
//!
//! Each case is run repeatedly for a fixed wall-clock budget and the *median*
//! iteration time is reported. `--save` writes the medians as JSON;
//! `--compare` prints the speed-up of this run against a saved file. The
//! models and inputs come from the same catalog as the golden-snapshot tests
//! (`tests/support`), so everything that is timed here is also pinned there.

#[path = "../tests/support/catalog.rs"]
mod catalog;
#[path = "../tests/support/datagen.rs"]
mod datagen;

use std::{
    collections::BTreeMap,
    hint::black_box,
    time::{Duration, Instant},
};

use rand::{SeedableRng, rngs::StdRng};
use rscopulas::{
    ClaytonCopula, CopulaModel, EvalOptions, ExecPolicy, FactorCopula, FactorFitOptions,
    FitOptions, FrankCopula, GaussianCopula, GumbelHougaardCopula, HacFitOptions,
    HierarchicalArchimedeanCopula, PairCopulaFamily, PseudoObs, SampleOptions, StudentTCopula,
    TllOrder, VineCopula, VineFitOptions, paircopula::fit_pair_copula, stats, tll_fit,
};

use datagen::Dependence;

struct Suite {
    filter: Option<String>,
    budget: Duration,
    results: BTreeMap<String, f64>,
}

impl Suite {
    /// Times `f`, reporting the median over as many iterations as fit in the
    /// budget (at least 3, at most 200).
    fn case<T>(&mut self, name: &str, mut f: impl FnMut() -> T) {
        if self
            .filter
            .as_ref()
            .is_some_and(|filter| !name.contains(filter.as_str()))
        {
            return;
        }
        black_box(f()); // warm-up: page in data, initialise lazies
        let mut samples = Vec::new();
        let started = Instant::now();
        while samples.len() < 3 || (started.elapsed() < self.budget && samples.len() < 200) {
            let tick = Instant::now();
            black_box(f());
            samples.push(tick.elapsed().as_secs_f64());
        }
        samples.sort_by(f64::total_cmp);
        let median = samples[samples.len() / 2];
        println!(
            "{name:<44} {:>12}  ({} iters)",
            format_time(median),
            samples.len()
        );
        self.results.insert(name.to_string(), median);
    }
}

fn format_time(seconds: f64) -> String {
    if seconds >= 1.0 {
        format!("{seconds:.3} s")
    } else if seconds >= 1e-3 {
        format!("{:.3} ms", seconds * 1e3)
    } else {
        format!("{:.3} µs", seconds * 1e6)
    }
}

fn obs(matrix: ndarray::Array2<f64>) -> PseudoObs {
    PseudoObs::new(matrix).expect("valid pseudo-observations")
}

fn pair_kernel_cases(suite: &mut Suite) {
    const N: usize = 20_000;
    let points = datagen::uniforms(N, 2, 0xBEEF);
    let u: Vec<f64> = points.column(0).to_vec();
    let v: Vec<f64> = points.column(1).to_vec();
    for (name, spec) in catalog::representative_pair_specs() {
        suite.case(&format!("pair/{name}/log_pdf x20k"), || {
            let mut total = 0.0;
            for (&a, &b) in u.iter().zip(&v) {
                total += spec.log_pdf(a, b, 1e-12).unwrap();
            }
            total
        });
        suite.case(&format!("pair/{name}/h x20k"), || {
            let mut total = 0.0;
            for (&a, &b) in u.iter().zip(&v) {
                total += spec.cond_first_given_second(a, b, 1e-12).unwrap();
            }
            total
        });
        suite.case(&format!("pair/{name}/hinv x20k"), || {
            let mut total = 0.0;
            for (&a, &b) in u.iter().zip(&v) {
                total += spec.inv_first_given_second(a, b, 1e-12).unwrap();
            }
            total
        });
    }
}

fn pair_fit_cases(suite: &mut Suite) {
    use PairCopulaFamily as F;
    let (u1, u2) = datagen::pair(1000, Dependence::HeavyTail, 0xF17);
    for family in [
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
        F::Tll,
    ] {
        let options = VineFitOptions {
            family_set: vec![family],
            ..VineFitOptions::default()
        };
        suite.case(&format!("pair_fit/{family:?} n=1000"), || {
            fit_pair_copula(&u1, &u2, &options).unwrap()
        });
    }
    let options = VineFitOptions::default();
    suite.case("pair_fit/default_set n=1000", || {
        fit_pair_copula(&u1, &u2, &options).unwrap()
    });
    let (k1, k2) = datagen::pair(300, Dependence::LowerTail, 0xF18);
    let options = VineFitOptions {
        family_set: vec![F::Khoudraji],
        ..VineFitOptions::default()
    };
    suite.case("pair_fit/Khoudraji n=300", || {
        fit_pair_copula(&k1, &k2, &options).unwrap()
    });
    suite.case("tll_fit/quadratic n=1000", || {
        tll_fit(&u1, &u2, TllOrder::Quadratic).unwrap()
    });
}

fn stats_cases(suite: &mut Suite) {
    let data = obs(datagen::pseudo_obs(2000, 10, Dependence::Gaussian, 0x57A7));
    suite.case("stats/kendall_tau n=2000 d=10", || {
        stats::try_kendall_tau_matrix(&data, ExecPolicy::Auto).unwrap()
    });
    suite.case("stats/spearman_rho n=2000 d=10", || {
        stats::try_spearman_rho_matrix(&data, ExecPolicy::Auto).unwrap()
    });
    let small = obs(datagen::pseudo_obs(1000, 5, Dependence::Gaussian, 0x57A8));
    suite.case("stats/hoeffding_d n=1000 d=5", || {
        stats::try_hoeffding_d_matrix(&small, ExecPolicy::Auto).unwrap()
    });
}

fn model_cases<M: CopulaModel>(suite: &mut Suite, name: &str, model: &M, n: usize) {
    let points = obs(datagen::uniforms(n, model.dim(), 0xE7A1));
    let eval = EvalOptions::default();
    let sample = SampleOptions::default();
    suite.case(&format!("{name}/log_pdf n={n}"), || {
        model.log_pdf(&points, &eval).unwrap()
    });
    suite.case(&format!("{name}/sample n={n}"), || {
        let mut rng = StdRng::seed_from_u64(7);
        model.sample(n, &mut rng, &sample).unwrap()
    });
}

fn single_family_cases(suite: &mut Suite) {
    model_cases(suite, "gaussian d=5", &catalog::gaussian_copula(5), 10_000);
    model_cases(
        suite,
        "student_t d=5",
        &catalog::student_t_copula(5, 4.5),
        10_000,
    );
    model_cases(suite, "clayton d=5", &catalog::clayton_copula(5), 10_000);
    model_cases(suite, "frank d=5", &catalog::frank_copula(5), 10_000);
    model_cases(suite, "gumbel d=5", &catalog::gumbel_copula(5), 10_000);

    let options = FitOptions::default();
    let data = obs(datagen::pseudo_obs(2000, 5, Dependence::HeavyTail, 0xE111));
    suite.case("gaussian d=5/fit n=2000", || {
        GaussianCopula::fit(&data, &options).unwrap()
    });
    suite.case("student_t d=5/fit n=2000", || {
        StudentTCopula::fit(&data, &options).unwrap()
    });
    let data = obs(datagen::pseudo_obs(2000, 5, Dependence::LowerTail, 0xE112));
    suite.case("clayton d=5/fit n=2000", || {
        ClaytonCopula::fit(&data, &options).unwrap()
    });
    suite.case("frank d=5/fit n=2000", || {
        FrankCopula::fit(&data, &options).unwrap()
    });
    suite.case("gumbel d=5/fit n=2000", || {
        GumbelHougaardCopula::fit(&data, &options).unwrap()
    });
}

fn hac_cases(suite: &mut Suite) {
    for (name, model) in catalog::hac_models() {
        let points = obs(datagen::uniforms(5000, 4, 0x4AC));
        if name.starts_with("flat") {
            suite.case(&format!("hac/{name}/log_pdf n=5000"), || {
                model.log_pdf(&points, &EvalOptions::default()).unwrap()
            });
        } else {
            suite.case(&format!("hac/{name}/composite_log_pdf n=5000"), || {
                model
                    .composite_log_pdf(&points, &EvalOptions::default())
                    .unwrap()
            });
            suite.case(&format!("hac/{name}/sample n=5000"), || {
                let mut rng = StdRng::seed_from_u64(7);
                model
                    .sample(5000, &mut rng, &SampleOptions::default())
                    .unwrap()
            });
        }
    }
    let data = obs(datagen::pseudo_obs(1000, 4, Dependence::UpperTail, 0x4AD));
    let options = HacFitOptions::default();
    suite.case("hac/fit n=1000 d=4", || {
        HierarchicalArchimedeanCopula::fit(&data, &options).unwrap()
    });
}

fn factor_cases(suite: &mut Suite) {
    let model = catalog::factor_model();
    let points = obs(datagen::uniforms(1000, model.dim(), 0xFAC));
    suite.case("factor d=5/log_pdf n=1000", || {
        model.log_pdf(&points, &EvalOptions::default()).unwrap()
    });
    suite.case("factor d=5/sample n=5000", || {
        let mut rng = StdRng::seed_from_u64(7);
        model
            .sample(5000, &mut rng, &SampleOptions::default())
            .unwrap()
    });
    let mut rng = StdRng::seed_from_u64(42);
    let sample = model
        .sample(500, &mut rng, &SampleOptions::default())
        .unwrap();
    let data = obs(sample);
    let options = FactorFitOptions::default();
    suite.case("factor d=5/fit n=500", || {
        FactorCopula::fit(&data, &options).unwrap()
    });
}

fn vine_cases(suite: &mut Suite) {
    use PairCopulaFamily as F;
    let n = 10_000;
    for (name, model) in [
        ("vine/mixed_d5", catalog::mixed_d_vine()),
        ("vine/mixed_c5", catalog::mixed_c_vine()),
        ("vine/gaussian_d10", catalog::gaussian_d_vine(10)),
    ] {
        model_cases(suite, name, &model, n);
        let points = datagen::uniforms(n, model.dim(), 0x705E);
        let options = SampleOptions::default();
        suite.case(&format!("{name}/rosenblatt n={n}"), || {
            model.rosenblatt(points.view(), &options).unwrap()
        });
        suite.case(&format!("{name}/inverse_rosenblatt n={n}"), || {
            model.inverse_rosenblatt(points.view(), &options).unwrap()
        });
    }

    let data = obs(datagen::pseudo_obs(1000, 6, Dependence::HeavyTail, 0xF17));
    let default = VineFitOptions::default();
    suite.case("vine_fit/r default n=1000 d=6", || {
        VineCopula::fit_r_vine(&data, &default).unwrap()
    });
    suite.case("vine_fit/c default n=1000 d=6", || {
        VineCopula::fit_c_vine(&data, &default).unwrap()
    });
    suite.case("vine_fit/d default n=1000 d=6", || {
        VineCopula::fit_d_vine(&data, &default).unwrap()
    });
    let gaussian = VineFitOptions {
        family_set: vec![F::Gaussian],
        ..default.clone()
    };
    suite.case("vine_fit/r gaussian-only n=1000 d=6", || {
        VineCopula::fit_r_vine(&data, &gaussian).unwrap()
    });
    let extended = VineFitOptions {
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
    };
    suite.case("vine_fit/r all-parametric n=1000 d=6", || {
        VineCopula::fit_r_vine(&data, &extended).unwrap()
    });
    let tll = VineFitOptions {
        family_set: vec![F::Tll],
        ..default.clone()
    };
    suite.case("vine_fit/r tll n=1000 d=6", || {
        VineCopula::fit_r_vine(&data, &tll).unwrap()
    });
    let wide = obs(datagen::pseudo_obs(500, 15, Dependence::Gaussian, 0xF1A));
    suite.case("vine_fit/r default n=500 d=15", || {
        VineCopula::fit_r_vine(&wide, &default).unwrap()
    });
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value_of = |flag: &str| {
        args.iter()
            .position(|arg| arg == flag)
            .and_then(|index| args.get(index + 1))
            .cloned()
    };
    let quick = args.iter().any(|arg| arg == "--quick");
    let mut suite = Suite {
        filter: value_of("--filter"),
        budget: Duration::from_millis(if quick { 150 } else { 600 }),
        results: BTreeMap::new(),
    };

    pair_kernel_cases(&mut suite);
    pair_fit_cases(&mut suite);
    stats_cases(&mut suite);
    single_family_cases(&mut suite);
    hac_cases(&mut suite);
    factor_cases(&mut suite);
    vine_cases(&mut suite);

    if let Some(path) = value_of("--save") {
        let text = serde_json::to_string_pretty(&suite.results).expect("serialize timings");
        std::fs::write(&path, text + "\n").expect("write timings");
        println!("\nsaved {} timings to {path}", suite.results.len());
    }
    if let Some(path) = value_of("--compare") {
        let baseline: BTreeMap<String, f64> =
            serde_json::from_slice(&std::fs::read(&path).expect("read baseline"))
                .expect("parse baseline");
        println!(
            "\n{:<44} {:>12} {:>12} {:>9}",
            "case", "baseline", "now", "speed-up"
        );
        let mut log_sum = 0.0;
        let mut count = 0usize;
        for (name, &now) in &suite.results {
            let Some(&before) = baseline.get(name) else {
                continue;
            };
            let ratio = before / now;
            log_sum += ratio.ln();
            count += 1;
            println!(
                "{name:<44} {:>12} {:>12} {:>8.2}x",
                format_time(before),
                format_time(now),
                ratio
            );
        }
        if count > 0 {
            println!(
                "\ngeometric-mean speed-up over {count} cases: {:.2}x",
                (log_sum / count as f64).exp()
            );
        }
    }
}
