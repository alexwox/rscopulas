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

theta <- 2.5
cop <- claytonCopula(param = theta, dim = 2)

set.seed(20260423)
log_pdf_input <- matrix(runif(16), ncol = 2)
log_pdf_fixture <- list(
  metadata = metadata,
  theta = theta,
  inputs = to_rows(log_pdf_input),
  expected_log_pdf = as.numeric(dCopula(log_pdf_input, copula = cop, log = TRUE))
)
write_fixture("clayton_log_pdf_d2_case01.json", log_pdf_fixture)

set.seed(20260424)
fit_input <- rCopula(256, cop)
fit_model <- fitCopula(claytonCopula(dim = 2), fit_input, method = "itau")
fit_fixture <- list(
  metadata = metadata,
  input_pobs = to_rows(fit_input),
  expected_theta = as.numeric(coef(fit_model))[1]
)
write_fixture("clayton_fit_d2_case01.json", fit_fixture)

set.seed(20260425)
sample_matrix <- rCopula(10000, cop)
sample_fixture <- list(
  metadata = metadata,
  theta = theta,
  seed = 626262,
  sample_size = 10000,
  expected_mean = as.numeric(colMeans(sample_matrix)),
  expected_kendall_tau = to_rows(cor(sample_matrix, method = "kendall"))
)
write_fixture("clayton_sample_summary_d2_case01.json", sample_fixture)
