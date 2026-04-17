#!/usr/bin/env Rscript

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
    toJSON(payload, auto_unbox = TRUE, digits = 16, pretty = TRUE),
    file = file.path(fixture_dir, filename)
  )
}

to_rows <- function(matrix_like) {
  unname(split(unname(matrix_like), row(matrix_like)))
}

rho <- 0.55
nu <- 6
cop <- tCopula(param = rho, dim = 2, dispstr = "un", df = nu, df.fixed = TRUE)

set.seed(20260420)
log_pdf_input <- matrix(runif(16), ncol = 2)
log_pdf_fixture <- list(
  metadata = metadata,
  correlation = list(c(1.0, rho), c(rho, 1.0)),
  degrees_of_freedom = nu,
  inputs = to_rows(log_pdf_input),
  expected_log_pdf = as.numeric(dCopula(log_pdf_input, copula = cop, log = TRUE))
)
write_fixture("student_t_log_pdf_d2_case01.json", log_pdf_fixture)

set.seed(20260421)
fit_input <- rCopula(256, cop)
fit_model <- fitCopula(tCopula(dim = 2, dispstr = "un"), fit_input, method = "itau.mpl")
fit_coef <- as.numeric(coef(fit_model))
fit_fixture <- list(
  metadata = metadata,
  input_pobs = to_rows(fit_input),
  expected_correlation = list(c(1.0, fit_coef[1]), c(fit_coef[1], 1.0)),
  expected_degrees_of_freedom = fit_coef[2]
)
write_fixture("student_t_fit_d2_case01.json", fit_fixture)

set.seed(20260422)
sample_matrix <- rCopula(10000, cop)
sample_fixture <- list(
  metadata = metadata,
  correlation = list(c(1.0, rho), c(rho, 1.0)),
  degrees_of_freedom = nu,
  seed = 515151,
  sample_size = 10000,
  expected_mean = as.numeric(colMeans(sample_matrix)),
  expected_kendall_tau = to_rows(cor(sample_matrix, method = "kendall"))
)
write_fixture("student_t_sample_summary_d2_case01.json", sample_fixture)
