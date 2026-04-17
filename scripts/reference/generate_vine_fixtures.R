#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(VineCopula))

fixture_dir <- "fixtures/reference/vinecopula/v2"
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)

metadata <- list(
  source_package = "VineCopula",
  source_version = as.character(packageVersion("VineCopula"))
)

write_fixture <- function(filename, payload) {
  write(
    toJSON(payload, auto_unbox = TRUE, digits = 16, pretty = TRUE),
    file = file.path(fixture_dir, filename)
  )
}

to_rows <- function(matrix_like) {
  unname(split(unname(matrix_like), row(matrix_like)))
}

rho <- matrix(
  c(
    1.0, 0.7, 0.3, 0.2,
    0.7, 1.0, 0.4, 0.1,
    0.3, 0.4, 1.0, 0.5,
    0.2, 0.1, 0.5, 1.0
  ),
  nrow = 4,
  byrow = TRUE
)
family <- rep(1, 6)
order <- 1:4

extract_params <- function(rvm) {
  d <- nrow(rvm$par)
  out <- c()
  for (row in d:2) {
    populated_cols <- seq_len(row - 1)
    for (col in populated_cols) {
      out <- c(out, rvm$par[row, col])
    }
  }
  out
}

set.seed(20260502)
log_pdf_input <- matrix(runif(16), ncol = 4)

c_rvm <- RVineCor2pcor(C2RVine(order, family, rep(0, length(family))), rho)
c_log_fixture <- list(
  metadata = metadata,
  structure = "C",
  order = as.list(0:3),
  correlation = to_rows(rho),
  pair_parameters = as.list(extract_params(c_rvm)),
  inputs = to_rows(log_pdf_input),
  expected_log_pdf = as.numeric(log(RVinePDF(log_pdf_input, c_rvm)))
)
write_fixture("gaussian_c_vine_log_pdf_d4_case01.json", c_log_fixture)

set.seed(20260503)
c_sample <- RVineSim(10000, c_rvm)
c_sample_fixture <- list(
  metadata = metadata,
  structure = "C",
  order = as.list(0:3),
  correlation = to_rows(rho),
  pair_parameters = as.list(extract_params(c_rvm)),
  seed = 919191,
  sample_size = 10000,
  expected_mean = as.numeric(colMeans(c_sample)),
  expected_kendall_tau = to_rows(cor(c_sample, method = "kendall"))
)
write_fixture("gaussian_c_vine_sample_summary_d4_case01.json", c_sample_fixture)

d_rvm <- RVineCor2pcor(D2RVine(order, family, rep(0, length(family))), rho)
d_log_fixture <- list(
  metadata = metadata,
  structure = "D",
  order = as.list(0:3),
  correlation = to_rows(rho),
  pair_parameters = as.list(extract_params(d_rvm)),
  inputs = to_rows(log_pdf_input),
  expected_log_pdf = as.numeric(log(RVinePDF(log_pdf_input, d_rvm)))
)
write_fixture("gaussian_d_vine_log_pdf_d4_case01.json", d_log_fixture)

set.seed(20260504)
d_sample <- RVineSim(10000, d_rvm)
d_sample_fixture <- list(
  metadata = metadata,
  structure = "D",
  order = as.list(0:3),
  correlation = to_rows(rho),
  pair_parameters = as.list(extract_params(d_rvm)),
  seed = 929292,
  sample_size = 10000,
  expected_mean = as.numeric(colMeans(d_sample)),
  expected_kendall_tau = to_rows(cor(d_sample, method = "kendall"))
)
write_fixture("gaussian_d_vine_sample_summary_d4_case01.json", d_sample_fixture)
