#!/usr/bin/env Rscript

userlib <- file.path(Sys.getenv("HOME"), "R", "library")
.libPaths(c(userlib, .libPaths()))
suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))

fixture_dir <- "fixtures/reference/r-copula/v1_1_3"
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)

metadata <- list(
  source_package = "copula",
  source_version = as.character(packageVersion("copula"))
)

write_fixture <- function(filename, payload) {
  write(
    toJSON(payload, auto_unbox = TRUE, digits = 16, pretty = TRUE, null = "null"),
    file = file.path(fixture_dir, filename)
  )
}

to_rows <- function(matrix_like) {
  unname(split(unname(matrix_like), row(matrix_like)))
}

pair_spec <- function(family, parameters = numeric(), rotation = "R0") {
  list(
    family = family,
    rotation = rotation,
    parameters = as.numeric(parameters)
  )
}

h12_numeric <- function(cop, u, v, eps = 1e-6) {
  v_lo <- pmax(v - eps, 1e-10)
  v_hi <- pmin(v + eps, 1 - 1e-10)
  (pCopula(cbind(u, v_hi), copula = cop) - pCopula(cbind(u, v_lo), copula = cop)) / (v_hi - v_lo)
}

h21_numeric <- function(cop, u, v, eps = 1e-6) {
  u_lo <- pmax(u - eps, 1e-10)
  u_hi <- pmin(u + eps, 1 - 1e-10)
  (pCopula(cbind(u_hi, v), copula = cop) - pCopula(cbind(u_lo, v), copula = cop)) / (u_hi - u_lo)
}

hinv12_numeric <- function(cop, p, v) {
  vapply(
    seq_along(p),
    function(idx) {
      uniroot(
        function(u) h12_numeric(cop, u, v[[idx]]) - p[[idx]],
        interval = c(1e-10, 1 - 1e-10),
        tol = 1e-9
      )$root
    },
    numeric(1)
  )
}

hinv21_numeric <- function(cop, u, p) {
  vapply(
    seq_along(p),
    function(idx) {
      uniroot(
        function(v) h21_numeric(cop, u[[idx]], v) - p[[idx]],
        interval = c(1e-10, 1 - 1e-10),
        tol = 1e-9
      )$root
    },
    numeric(1)
  )
}

pair_case <- function() {
  base_first <- pair_spec("gaussian", c(0.45))
  base_second <- pair_spec("clayton", c(2.0))
  shape_1 <- 0.35
  shape_2 <- 0.80
  cop <- khoudrajiCopula(
    copula1 = normalCopula(0.45, dim = 2, dispstr = "un"),
    copula2 = claytonCopula(2.0, dim = 2),
    shapes = c(shape_1, shape_2)
  )

  u1 <- c(0.17, 0.31, 0.62, 0.88)
  u2 <- c(0.23, 0.54, 0.41, 0.79)
  p <- c(0.27, 0.45, 0.73, 0.91)

  list(
    metadata = metadata,
    family = "khoudraji",
    rotation = "R0",
    base_copula_1 = base_first,
    base_copula_2 = base_second,
    shape_1 = shape_1,
    shape_2 = shape_2,
    u1 = as.list(u1),
    u2 = as.list(u2),
    p = as.list(p),
    expected_log_pdf = as.numeric(dCopula(cbind(u1, u2), copula = cop, log = TRUE)),
    expected_cond_first_given_second = as.numeric(h12_numeric(cop, u1, u2)),
    expected_cond_second_given_first = as.numeric(h21_numeric(cop, u1, u2)),
    expected_inv_first_given_second = as.numeric(hinv12_numeric(cop, p, u2)),
    expected_inv_second_given_first = as.numeric(hinv21_numeric(cop, u1, p))
  )
}

sample_case <- function() {
  base_first <- pair_spec("gaussian", c(0.45))
  base_second <- pair_spec("clayton", c(2.0))
  shape_1 <- 0.35
  shape_2 <- 0.80
  cop <- khoudrajiCopula(
    copula1 = normalCopula(0.45, dim = 2, dispstr = "un"),
    copula2 = claytonCopula(2.0, dim = 2),
    shapes = c(shape_1, shape_2)
  )

  set.seed(20260426)
  sample_matrix <- rCopula(10000, cop)

  list(
    metadata = metadata,
    family = "khoudraji",
    rotation = "R0",
    base_copula_1 = base_first,
    base_copula_2 = base_second,
    shape_1 = shape_1,
    shape_2 = shape_2,
    seed = 717171,
    sample_size = 10000,
    expected_mean = as.numeric(colMeans(sample_matrix)),
    expected_kendall_tau = to_rows(cor(sample_matrix, method = "kendall"))
  )
}

fit_case <- function() {
  fit_source <- khoudrajiCopula(
    copula2 = claytonCopula(2.5, dim = 2),
    shapes = c(0.45, 0.70)
  )

  set.seed(20260427)
  fit_input <- rCopula(96, fit_source)
  fit_model <- fitCopula(
    khoudrajiCopula(copula2 = claytonCopula(dim = 2)),
    data = fit_input,
    start = c(1.5, 0.4, 0.6),
    optim.method = "Nelder-Mead"
  )
  fitted <- as.numeric(coef(fit_model))

  list(
    metadata = metadata,
    family = "khoudraji",
    rotation = "R0",
    base_copula_1 = pair_spec("independence"),
    base_copula_2 = pair_spec("clayton"),
    input_pobs = to_rows(fit_input),
    expected_theta = fitted[[1]],
    expected_shape_1 = fitted[[2]],
    expected_shape_2 = fitted[[3]]
  )
}

write_fixture("khoudraji_pair_case01.json", pair_case())
write_fixture("khoudraji_sample_summary_case01.json", sample_case())
write_fixture("khoudraji_fit_case01.json", fit_case())
