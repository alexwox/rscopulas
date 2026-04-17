#!/usr/bin/env Rscript

userlib <- file.path(Sys.getenv("HOME"), "R", "library")
.libPaths(c(userlib, .libPaths()))
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
    toJSON(payload, auto_unbox = TRUE, digits = 16, pretty = TRUE, null = "null"),
    file = file.path(fixture_dir, filename)
  )
}

to_rows <- function(matrix_like) {
  unname(split(unname(matrix_like), row(matrix_like)))
}

to_zero_based_rows <- function(matrix_like) {
  lowered <- ifelse(matrix_like == 0, 0, matrix_like - 1L)
  to_rows(lowered)
}

to_zero_based <- function(values) {
  as.list(unname(values - 1L))
}

rotation_name <- function(family_code) {
  switch(
    as.character(family_code),
    "13" = "R180",
    "14" = "R180",
    "23" = "R90",
    "24" = "R90",
    "33" = "R270",
    "34" = "R270",
    "R0"
  )
}

family_name <- function(family_code) {
  switch(
    as.character(family_code),
    "0" = "Independence",
    "1" = "Gaussian",
    "2" = "StudentT",
    "3" = "Clayton",
    "13" = "Clayton",
    "23" = "Clayton",
    "33" = "Clayton",
    "4" = "Gumbel",
    "14" = "Gumbel",
    "24" = "Gumbel",
    "34" = "Gumbel",
    "5" = "Frank",
    stop(sprintf("unsupported family code: %s", family_code))
  )
}

swap_family_code <- function(family_code) {
  switch(
    as.character(family_code),
    "23" = 33,
    "33" = 23,
    "24" = 34,
    "34" = 24,
    family_code
  )
}

edge_payload <- function(edge) {
  payload <- list(
    conditioned = to_zero_based(edge$conditioned),
    conditioning = to_zero_based(edge$conditioning),
    family = family_name(edge$family),
    rotation = rotation_name(edge$family)
  )

  if (edge$family == 0) {
    payload$params <- list()
  } else if (edge$family == 2) {
    payload$params <- list(unname(edge$par), unname(edge$par2))
  } else if (edge$family %in% c(23, 33, 24, 34)) {
    payload$params <- list(abs(unname(edge$par)))
  } else {
    payload$params <- list(unname(edge$par))
  }

  payload
}

trees_payload <- function(trees) {
  lapply(seq_along(trees), function(level) {
    list(
      level = level,
      edges = lapply(trees[[level]], edge_payload)
    )
  })
}

trees_to_rvine_parts <- function(trees, dim) {
  Matrix <- matrix(0, dim, dim)
  family <- matrix(0, dim, dim)
  par <- matrix(0, dim, dim)
  par2 <- matrix(0, dim, dim)
  remaining <- lapply(trees, function(tree) tree)

  for (k in seq_len(dim - 1)) {
    source_tree <- dim - k
    first <- remaining[[source_tree]][[1]]
    w <- first$conditioned[[1]]
    Matrix[k, k] <- w
    Matrix[k + 1, k] <- first$conditioned[[2]]
    family[k + 1, k] <- first$family
    par[k + 1, k] <- first$par
    par2[k + 1, k] <- first$par2

    if (k == dim - 1) {
      Matrix[k + 1, k + 1] <- first$conditioned[[2]]
      next
    }

    for (i in (k + 2):dim) {
      tree_idx <- dim - i + 1
      found_idx <- NA_integer_
      found_value <- NA_integer_
      found_family <- 0
      found_par <- 0
      found_par2 <- 0
      edges <- remaining[[tree_idx]]

      for (edge_idx in seq_along(edges)) {
        edge <- edges[[edge_idx]]
        if (edge$conditioned[[1]] == w) {
          found_idx <- edge_idx
          found_value <- edge$conditioned[[2]]
          found_family <- edge$family
          found_par <- edge$par
          found_par2 <- edge$par2
          break
        }
        if (edge$conditioned[[2]] == w) {
          found_idx <- edge_idx
          found_value <- edge$conditioned[[1]]
          found_family <- swap_family_code(edge$family)
          found_par <- edge$par
          found_par2 <- edge$par2
          break
        }
      }

      if (is.na(found_idx)) {
        stop("failed to convert trees into an RVine matrix")
      }

      Matrix[i, k] <- found_value
      family[i, k] <- found_family
      par[i, k] <- found_par
      par2[i, k] <- found_par2
      remaining[[tree_idx]] <- remaining[[tree_idx]][-found_idx]
    }
  }

  list(Matrix = Matrix, family = family, par = par, par2 = par2)
}

write_tree_fixture <- function(filename, structure, trees, inputs = NULL, sample_seed = NULL, sample_size = NULL) {
  dim <- length(trees) + 1L
  parts <- trees_to_rvine_parts(trees, dim)
  rvm <- RVineMatrix(
    Matrix = parts$Matrix,
    family = parts$family,
    par = parts$par,
    par2 = parts$par2
  )

  if (RVineMatrixCheck(rvm$Matrix) != 1) {
    stop("generated vine matrix is invalid")
  }

  if (!is.null(inputs)) {
    payload <- list(
      metadata = metadata,
      structure = structure,
      truncation_level = NULL,
      matrix = to_zero_based_rows(rvm$Matrix),
      trees = trees_payload(trees),
      inputs = to_rows(inputs),
      expected_log_pdf = as.numeric(log(RVinePDF(inputs, rvm)))
    )
    write_fixture(filename, payload)
  }

  if (!is.null(sample_seed) && !is.null(sample_size)) {
    set.seed(sample_seed)
    sample <- RVineSim(sample_size, rvm)
    payload <- list(
      metadata = metadata,
      structure = structure,
      truncation_level = NULL,
      matrix = to_zero_based_rows(rvm$Matrix),
      trees = trees_payload(trees),
      seed = sample_seed,
      sample_size = sample_size,
      expected_mean = as.numeric(colMeans(sample)),
      expected_kendall_tau = to_rows(cor(sample, method = "kendall"))
    )
    write_fixture(filename, payload)
  }
}

write_truncated_fixture <- function(filename, structure, trees, truncation_level, inputs = NULL, sample_seed = NULL, sample_size = NULL) {
  truncated <- trees
  for (level in seq_along(truncated)) {
    if (level > truncation_level) {
      for (edge_idx in seq_along(truncated[[level]])) {
        truncated[[level]][[edge_idx]]$family <- 0
        truncated[[level]][[edge_idx]]$par <- 0
        truncated[[level]][[edge_idx]]$par2 <- 0
      }
    }
  }

  dim <- length(truncated) + 1L
  parts <- trees_to_rvine_parts(truncated, dim)
  rvm <- RVineMatrix(
    Matrix = parts$Matrix,
    family = parts$family,
    par = parts$par,
    par2 = parts$par2
  )

  if (!is.null(inputs)) {
    payload <- list(
      metadata = metadata,
      structure = structure,
      truncation_level = truncation_level,
      matrix = to_zero_based_rows(rvm$Matrix),
      trees = trees_payload(truncated),
      inputs = to_rows(inputs),
      expected_log_pdf = as.numeric(log(RVinePDF(inputs, rvm)))
    )
    write_fixture(filename, payload)
  }

  if (!is.null(sample_seed) && !is.null(sample_size)) {
    set.seed(sample_seed)
    sample <- RVineSim(sample_size, rvm)
    payload <- list(
      metadata = metadata,
      structure = structure,
      truncation_level = truncation_level,
      matrix = to_zero_based_rows(rvm$Matrix),
      trees = trees_payload(truncated),
      seed = sample_seed,
      sample_size = sample_size,
      expected_mean = as.numeric(colMeans(sample)),
      expected_kendall_tau = to_rows(cor(sample, method = "kendall"))
    )
    write_fixture(filename, payload)
  }
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

mixed_r_trees <- list(
  list(
    list(conditioned = c(1L, 2L), conditioning = integer(), family = 1L, par = 0.65, par2 = 0),
    list(conditioned = c(2L, 3L), conditioning = integer(), family = 3L, par = 1.4, par2 = 0),
    list(conditioned = c(3L, 4L), conditioning = integer(), family = 5L, par = 3.2, par2 = 0),
    list(conditioned = c(3L, 5L), conditioning = integer(), family = 24L, par = -1.25, par2 = 0)
  ),
  list(
    list(conditioned = c(1L, 3L), conditioning = c(2L), family = 2L, par = 0.45, par2 = 5),
    list(conditioned = c(2L, 4L), conditioning = c(3L), family = 14L, par = 1.35, par2 = 0),
    list(conditioned = c(2L, 5L), conditioning = c(3L), family = 1L, par = -0.25, par2 = 0)
  ),
  list(
    list(conditioned = c(1L, 4L), conditioning = c(2L, 3L), family = 23L, par = -1.15, par2 = 0),
    list(conditioned = c(4L, 5L), conditioning = c(2L, 3L), family = 5L, par = 2.1, par2 = 0)
  ),
  list(
    list(conditioned = c(1L, 5L), conditioning = c(2L, 3L, 4L), family = 3L, par = 0.9, par2 = 0)
  )
)

set.seed(20260505)
r_input <- matrix(runif(25), ncol = 5)
write_tree_fixture(
  "mixed_r_vine_log_pdf_d5_case01.json",
  "R",
  mixed_r_trees,
  inputs = r_input
)
write_tree_fixture(
  "mixed_r_vine_sample_summary_d5_case01.json",
  "R",
  mixed_r_trees,
  sample_seed = 20260506,
  sample_size = 10000
)
set.seed(20260508)
mixed_r_fit_data <- RVineSim(120, RVineMatrix(
  Matrix = trees_to_rvine_parts(mixed_r_trees, 5)$Matrix,
  family = trees_to_rvine_parts(mixed_r_trees, 5)$family,
  par = trees_to_rvine_parts(mixed_r_trees, 5)$par,
  par2 = trees_to_rvine_parts(mixed_r_trees, 5)$par2
))
write_fixture(
  "mixed_r_vine_fit_data_d5_case01.json",
  list(
    metadata = metadata,
    structure = "R",
    trees = trees_payload(mixed_r_trees),
    data = to_rows(mixed_r_fit_data)
  )
)
write_truncated_fixture(
  "mixed_r_vine_trunc2_log_pdf_d5_case01.json",
  "R",
  mixed_r_trees,
  truncation_level = 2,
  inputs = r_input
)
write_truncated_fixture(
  "mixed_r_vine_trunc2_sample_summary_d5_case01.json",
  "R",
  mixed_r_trees,
  truncation_level = 2,
  sample_seed = 20260507,
  sample_size = 10000
)
