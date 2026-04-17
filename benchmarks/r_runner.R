#!/usr/bin/env Rscript

suppressPackageStartupMessages(library(copula))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(VineCopula))

args <- commandArgs(trailingOnly = TRUE)

arg_value <- function(flag) {
  index <- match(flag, args)
  if (is.na(index) || index == length(args)) {
    stop(sprintf("missing required argument %s", flag), call. = FALSE)
  }
  args[[index + 1L]]
}

read_json <- function(path) {
  fromJSON(path, simplifyVector = FALSE)
}

write_json <- function(path, payload) {
  dir.create(dirname(path), recursive = TRUE, showWarnings = FALSE)
  write(
    toJSON(payload, auto_unbox = TRUE, pretty = TRUE, null = "null"),
    file = path
  )
}

to_matrix <- function(rows) {
  if (length(rows) == 0) {
    matrix(numeric(), nrow = 0, ncol = 0)
  } else {
    matrix(unlist(rows), ncol = length(rows[[1L]]), byrow = TRUE)
  }
}

measure_iterations <- function(iterations, callback) {
  callback()
  start <- proc.time()[["elapsed"]]
  for (idx in seq_len(iterations)) {
    callback()
  }
  total_ns <- round((proc.time()[["elapsed"]] - start) * 1e9)
  list(
    iterations = iterations,
    mean_ns = total_ns / iterations,
    total_ns = total_ns
  )
}

make_result <- function(case, measurement, extra = list()) {
  c(
    list(
      case_id = case$id,
      category = case$category,
      family = case$family,
      operation = case$operation,
      structure = case$structure,
      fixture = case$fixture
    ),
    measurement,
    extra
  )
}

family_code <- function(family, rotation) {
  key <- paste0(family, ":", rotation)
  switch(
    key,
    "Independence:R0" = 0L,
    "Gaussian:R0" = 1L,
    "StudentT:R0" = 2L,
    "Clayton:R0" = 3L,
    "Clayton:R180" = 13L,
    "Clayton:R90" = 23L,
    "Clayton:R270" = 33L,
    "Gumbel:R0" = 4L,
    "Gumbel:R180" = 14L,
    "Gumbel:R90" = 24L,
    "Gumbel:R270" = 34L,
    "Frank:R0" = 5L,
    stop(sprintf("unsupported family/rotation combination: %s", key), call. = FALSE)
  )
}

swap_family_code <- function(code) {
  switch(
    as.character(code),
    "23" = 33L,
    "33" = 23L,
    "24" = 34L,
    "34" = 24L,
    code
  )
}

normalize_edge <- function(edge) {
  params <- unlist(edge$params)
  list(
    conditioned = as.integer(unlist(edge$conditioned)) + 1L,
    conditioning = as.integer(unlist(edge$conditioning)) + 1L,
    family = family_code(edge$family, edge$rotation),
    par = if (length(params) >= 1L) as.numeric(params[[1L]]) else 0.0,
    par2 = if (length(params) >= 2L) as.numeric(params[[2L]]) else 0.0
  )
}

trees_to_rvine_parts <- function(trees, dim) {
  Matrix <- matrix(0, dim, dim)
  family <- matrix(0, dim, dim)
  par <- matrix(0, dim, dim)
  par2 <- matrix(0, dim, dim)
  remaining <- lapply(trees, function(tree) tree)

  for (k in seq_len(dim - 1L)) {
    source_tree <- dim - k
    first <- remaining[[source_tree]][[1L]]
    w <- first$conditioned[[1L]]
    Matrix[k, k] <- w
    Matrix[k + 1L, k] <- first$conditioned[[2L]]
    family[k + 1L, k] <- first$family
    par[k + 1L, k] <- first$par
    par2[k + 1L, k] <- first$par2

    if (k == dim - 1L) {
      Matrix[k + 1L, k + 1L] <- first$conditioned[[2L]]
      next
    }

    for (i in (k + 2L):dim) {
      tree_idx <- dim - i + 1L
      edges <- remaining[[tree_idx]]
      found_idx <- NA_integer_
      found_value <- NA_integer_
      found_family <- 0L
      found_par <- 0.0
      found_par2 <- 0.0

      for (edge_idx in seq_along(edges)) {
        edge <- edges[[edge_idx]]
        if (edge$conditioned[[1L]] == w) {
          found_idx <- edge_idx
          found_value <- edge$conditioned[[2L]]
          found_family <- edge$family
          found_par <- edge$par
          found_par2 <- edge$par2
          break
        }
        if (edge$conditioned[[2L]] == w) {
          found_idx <- edge_idx
          found_value <- edge$conditioned[[1L]]
          found_family <- swap_family_code(edge$family)
          found_par <- edge$par
          found_par2 <- edge$par2
          break
        }
      }

      if (is.na(found_idx)) {
        stop("failed to reconstruct RVine matrix from benchmark fixture", call. = FALSE)
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

rvine_from_fixture <- function(fixture) {
  dim <- length(fixture$trees) + 1L
  trees <- lapply(fixture$trees, function(tree) {
    lapply(tree$edges, normalize_edge)
  })
  parts <- trees_to_rvine_parts(trees, dim)
  RVineMatrix(
    Matrix = parts$Matrix,
    family = parts$family,
    par = parts$par,
    par2 = parts$par2
  )
}

single_family_model <- function(case, fixture) {
  family <- case$family
  if (family == "gaussian") {
    rho <- as.numeric(fixture$correlation[[1L]][[2L]])
    return(normalCopula(param = rho, dim = 2L, dispstr = "un"))
  }
  if (family == "student_t") {
    rho <- as.numeric(fixture$correlation[[1L]][[2L]])
    return(tCopula(param = rho, dim = 2L, df = as.numeric(fixture$degrees_of_freedom), dispstr = "un"))
  }
  theta <- as.numeric(fixture$theta)
  if (family == "clayton") {
    return(claytonCopula(theta, dim = 2L))
  }
  if (family == "frank") {
    return(frankCopula(theta, dim = 2L))
  }
  if (family == "gumbel") {
    return(gumbelCopula(theta, dim = 2L))
  }
  stop(sprintf("unsupported family %s", family), call. = FALSE)
}

run_single_family <- function(case, fixture) {
  operation <- case$operation
  if (operation == "log_pdf") {
    model <- single_family_model(case, fixture)
    data <- to_matrix(fixture$inputs)
    measurement <- measure_iterations(case$iterations, function() dCopula(data, model, log = TRUE))
    return(make_result(case, measurement, list(dim = ncol(data), observations = nrow(data))))
  }
  if (operation == "fit") {
    data <- to_matrix(fixture$input_pobs)
    fit_call <- switch(
      case$family,
      "gaussian" = function() fitCopula(normalCopula(dim = 2L, dispstr = "un"), data, method = "itau"),
      "student_t" = function() fitCopula(tCopula(dim = 2L, dispstr = "un"), data, method = "itau.mpl"),
      "clayton" = function() fitCopula(claytonCopula(dim = 2L), data, method = "itau"),
      "frank" = function() fitCopula(frankCopula(dim = 2L), data, method = "itau"),
      "gumbel" = function() fitCopula(gumbelCopula(dim = 2L), data, method = "itau"),
      stop(sprintf("unsupported family %s", case$family), call. = FALSE)
    )
    measurement <- measure_iterations(case$iterations, fit_call)
    return(make_result(case, measurement, list(dim = ncol(data), observations = nrow(data))))
  }
  if (operation == "sample") {
    model <- single_family_model(case, fixture)
    sample_size <- as.integer(fixture$sample_size)
    seed <- as.integer(fixture$seed)
    measurement <- measure_iterations(case$iterations, function() {
      set.seed(seed)
      rCopula(sample_size, model)
    })
    return(make_result(case, measurement, list(dim = 2L, sample_size = sample_size)))
  }
  stop(sprintf("unsupported single-family operation %s", operation), call. = FALSE)
}

run_pair_kernels <- function(case, fixture) {
  family <- as.integer(fixture$family_code)
  par <- as.numeric(fixture$par)
  par2 <- as.numeric(fixture$par2)
  u1 <- as.numeric(unlist(fixture$u1))
  u2 <- as.numeric(unlist(fixture$u2))
  p <- as.numeric(unlist(fixture$p))
  measurement <- measure_iterations(case$iterations, function() {
    BiCopPDF(u1, u2, family = family, par = par, par2 = par2)
    BiCopHfunc1(u1, u2, family = family, par = par, par2 = par2)
    BiCopHfunc2(u1, u2, family = family, par = par, par2 = par2)
    BiCopHinv1(p, u2, family = family, par = par, par2 = par2)
    BiCopHinv2(u1, p, family = family, par = par, par2 = par2)
  })
  make_result(case, measurement, list(observations = length(u1)))
}

run_vine <- function(case, fixture) {
  operation <- case$operation
  if (operation == "log_pdf") {
    model <- rvine_from_fixture(fixture)
    data <- to_matrix(fixture$inputs)
    measurement <- measure_iterations(case$iterations, function() log(RVinePDF(data, model)))
    return(make_result(case, measurement, list(dim = ncol(data), observations = nrow(data))))
  }
  if (operation == "sample") {
    model <- rvine_from_fixture(fixture)
    sample_size <- as.integer(fixture$sample_size)
    seed <- as.integer(fixture$seed)
    measurement <- measure_iterations(case$iterations, function() {
      set.seed(seed)
      RVineSim(sample_size, model)
    })
    return(make_result(case, measurement, list(dim = ncol(model$Matrix), sample_size = sample_size)))
  }
  if (operation == "fit") {
    data <- to_matrix(fixture$data)
    familyset <- c(0L, 1L, 3L, 4L, 5L, 13L, 14L, 23L, 24L, 33L, 34L)
    measurement <- measure_iterations(case$iterations, function() {
      RVineStructureSelect(
        data,
        familyset = familyset,
        indeptest = FALSE,
        selectioncrit = "AIC",
        trunclevel = 2L
      )
    })
    return(make_result(case, measurement, list(dim = ncol(data), observations = nrow(data))))
  }
  stop(sprintf("unsupported vine operation %s", operation), call. = FALSE)
}

run_case <- function(case) {
  fixture <- read_json(case$fixture)
  if (case$category == "single_family") {
    return(run_single_family(case, fixture))
  }
  if (case$category == "pair_copula") {
    return(run_pair_kernels(case, fixture))
  }
  if (case$category == "vine") {
    return(run_vine(case, fixture))
  }
  stop(sprintf("unsupported benchmark category %s", case$category), call. = FALSE)
}

cases_path <- arg_value("--cases")
output_path <- arg_value("--output")
manifest <- read_json(cases_path)
cases <- Filter(function(case) "r" %in% case$implementations, manifest$cases)
results <- lapply(cases, run_case)
payload <- list(
  implementation = "r",
  runner = "benchmarks/r_runner.R",
  environment = list(
    platform = R.version$platform,
    r_version = R.version.string,
    copula_version = as.character(packageVersion("copula")),
    vinecopula_version = as.character(packageVersion("VineCopula"))
  ),
  results = results
)
write_json(output_path, payload)
