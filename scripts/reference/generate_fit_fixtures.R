#!/usr/bin/env Rscript
#
# Reference fixtures for the *fitters* (not just the kernels).
#
# For every pair-copula family (and every rotation of the Archimedean
# families) this script simulates n = 2000 pseudo-observations with
# `BiCopSim`, fits the same family with `BiCopEst(method = "mle", se = TRUE)`
# and stores the data together with R's maximum-likelihood estimate, its
# Hessian-based standard errors, the log-likelihood at that estimate and the
# implied Kendall tau. It also builds 4- and 5-dimensional C-vines with a
# fixed variable order, simulates with `RVineSim`, re-estimates the
# parameters with `RVineSeqEst` on the *same* structure/family matrix, and
# stores everything per edge.
#
# Output: fixtures/reference/fits/v1/*.json (run from the repository root).
#
# Conventions (rscopulas <-> VineCopula):
#   * rscopulas rotates the *arguments*: R90 evaluates the base copula at
#     (1 - u1, u2), R270 at (u1, 1 - u2), R180 at (1 - u1, 1 - u2). This is
#     exactly VineCopula's convention for the exchangeable families
#     (Clayton/Gumbel/Joe/BB1/BB6/BB7/BB8): codes +10 / +20 / +30 are
#     180 / 90 / 270 degrees, and 90/270 use *negative* parameters in R
#     while rscopulas keeps the base parameters positive.
#   * For the asymmetric Tawn families VineCopula's 90/270 codes correspond
#     to the *other* named family in rscopulas (verified numerically with
#     BiCopPDF): 124 == Tawn2 R90, 134 == Tawn2 R270, 224 == Tawn1 R90,
#     234 == Tawn1 R270 (0 and 180 degrees map 1:1). `rscopulas_family_code`
#     below encodes that mapping.
#   * A negative Frank parameter has no rscopulas representation today; the
#     fixture keeps the signed value (rotation R0) and the Rust test checks
#     the density through the identity c_{-theta}(u, v) = c_theta(1 - u, v).
#   * C-vine: `C2RVine(order)` puts `order[1]` in the bottom row of the
#     R-vine matrix (root of tree 1), `order[2]` in the row above (root of
#     tree 2), ... and the cell copula in (i, j) is C(u_M[i,j], u_M[j,j]),
#     i.e. (root, other). rscopulas' `fit_c_vine_with_order(order0)` uses
#     `order0[k]` (0-based) as the root of tree k + 1 with edges oriented
#     (root, other) as well, so `order0 = order - 1` and rotation codes map
#     1:1 per edge. Note that `order0[d - 1]` becomes `variable_order()[0]`
#     (the Rosenblatt anchor) on the rscopulas side.
#
# All data are rounded to 12 significant digits *before* fitting so that the
# JSON payload reproduces bit-identical doubles on the Rust side and the
# stored log-likelihoods refer to exactly the stored data.

userlib <- file.path(Sys.getenv("HOME"), "R", "library")
.libPaths(c(userlib, .libPaths()))
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(VineCopula))

fixture_dir <- "fixtures/reference/fits/v1"
dir.create(fixture_dir, recursive = TRUE, showWarnings = FALSE)

metadata <- list(
  source_package = "VineCopula",
  source_version = as.character(packageVersion("VineCopula")),
  r_version = R.version.string,
  generator = "scripts/reference/generate_fit_fixtures.R"
)

n_obs <- 2000L
clip <- 1e-10

write_fixture <- function(filename, payload) {
  write(
    toJSON(payload, auto_unbox = TRUE, digits = I(15), pretty = TRUE, null = "null"),
    file = file.path(fixture_dir, filename)
  )
}

to_rows <- function(matrix_like) {
  unname(split(unname(matrix_like), row(matrix_like)))
}

clamp_unit <- function(values) {
  pmin(pmax(values, clip), 1 - clip)
}

# --- family code mapping ------------------------------------------------------

base_family_name <- function(base_code) {
  switch(
    as.character(base_code),
    "1" = "Gaussian",
    "2" = "StudentT",
    "3" = "Clayton",
    "4" = "Gumbel",
    "5" = "Frank",
    "6" = "Joe",
    "7" = "Bb1",
    "8" = "Bb6",
    "9" = "Bb7",
    "10" = "Bb8",
    "104" = "Tawn1",
    "204" = "Tawn2",
    stop(sprintf("unsupported base family code %s", base_code))
  )
}

# Returns list(family = <rscopulas family>, rotation = <R0|R90|R180|R270>).
rscopulas_family_code <- function(code) {
  if (code %in% c(1, 2, 5)) {
    return(list(family = base_family_name(code), rotation = "R0"))
  }
  if (code %in% c(104, 204)) {
    return(list(family = base_family_name(code), rotation = "R0"))
  }
  if (code %in% c(114, 214)) {
    return(list(family = base_family_name(code - 10), rotation = "R180"))
  }
  # VineCopula's Tawn 90/270 codes correspond to the other named family in
  # rscopulas (see header comment).
  if (code == 124) return(list(family = "Tawn2", rotation = "R90"))
  if (code == 134) return(list(family = "Tawn2", rotation = "R270"))
  if (code == 224) return(list(family = "Tawn1", rotation = "R90"))
  if (code == 234) return(list(family = "Tawn1", rotation = "R270"))
  if (code %in% c(3, 4, 6, 7, 8, 9, 10)) {
    return(list(family = base_family_name(code), rotation = "R0"))
  }
  if (code %in% c(13, 14, 16, 17, 18, 19, 20)) {
    return(list(family = base_family_name(code - 10), rotation = "R180"))
  }
  if (code %in% c(23, 24, 26, 27, 28, 29, 30)) {
    return(list(family = base_family_name(code - 20), rotation = "R90"))
  }
  if (code %in% c(33, 34, 36, 37, 38, 39, 40)) {
    return(list(family = base_family_name(code - 30), rotation = "R270"))
  }
  stop(sprintf("unsupported family code %s", code))
}

n_params <- function(code) {
  if (code == 2) return(2L)
  if (code >= 100) return(2L) # Tawn
  base <- if (code >= 10) code %% 10 else code
  if (code %in% c(10, 20, 30, 40)) return(2L) # BB8 and rotations
  if (base %in% c(7, 8, 9)) return(2L)
  1L
}

# VineCopula parameters -> rscopulas parameters.
# Rotated (90/270) exchangeable families use negative parameters in R;
# rscopulas keeps the base parameters positive and stores the rotation.
# Frank keeps its sign (rscopulas has no rotation for Frank).
rscopulas_params <- function(code, par, par2) {
  mapped <- rscopulas_family_code(code)
  k <- n_params(code)
  values <- if (k == 2L) c(par, par2) else c(par)
  if (mapped$family == "Frank") {
    return(values)
  }
  if (mapped$rotation %in% c("R90", "R270")) {
    if (mapped$family %in% c("Tawn1", "Tawn2")) {
      # theta is negated by VineCopula for 90/270; the shape stays in [0, 1].
      return(c(abs(par), par2))
    }
    return(abs(values))
  }
  values
}

# --- pair-copula fits ---------------------------------------------------------

pair_case <- function(name, family, par, par2 = 0, seed) {
  set.seed(seed)
  raw <- BiCopSim(n_obs, family = family, par = par, par2 = par2)
  u <- signif(clamp_unit(raw), 12)
  u1 <- u[, 1]
  u2 <- u[, 2]

  fit <- BiCopEst(u1, u2, family = family, method = "mle", se = TRUE)
  loglik <- sum(log(BiCopPDF(u1, u2, family = family, par = fit$par, par2 = fit$par2)))
  se <- if (n_params(family) == 2L) c(fit$se, fit$se2) else c(fit$se)
  if (any(!is.finite(se))) {
    stop(sprintf("non-finite standard error for case %s", name))
  }
  mapped <- rscopulas_family_code(family)

  list(
    metadata = metadata,
    case = name,
    family_code = family,
    family = mapped$family,
    rotation = mapped$rotation,
    seed = seed,
    n = n_obs,
    true_par = par,
    true_par2 = par2,
    true_params = I(rscopulas_params(family, par, par2)),
    r_par = fit$par,
    r_par2 = fit$par2,
    r_params = I(rscopulas_params(family, fit$par, fit$par2)),
    r_se = I(se),
    r_loglik = loglik,
    r_loglik_reported = as.numeric(fit$logLik),
    r_tau = BiCopPar2Tau(family, fit$par, fit$par2),
    empirical_tau = cor(u1, u2, method = "kendall"),
    u1 = u1,
    u2 = u2
  )
}

pair_cases <- list(
  list(name = "gaussian", family = 1, par = 0.6),
  list(name = "gaussian_neg", family = 1, par = -0.45),
  list(name = "student_t", family = 2, par = 0.6, par2 = 5),
  list(name = "clayton", family = 3, par = 2.0),
  list(name = "clayton_rot180", family = 13, par = 2.0),
  list(name = "clayton_rot90", family = 23, par = -2.0),
  list(name = "clayton_rot270", family = 33, par = -2.0),
  list(name = "gumbel", family = 4, par = 1.8),
  list(name = "gumbel_rot180", family = 14, par = 1.8),
  list(name = "gumbel_rot90", family = 24, par = -1.8),
  list(name = "gumbel_rot270", family = 34, par = -1.8),
  list(name = "frank", family = 5, par = 4.0),
  list(name = "frank_neg", family = 5, par = -4.0),
  list(name = "joe", family = 6, par = 2.5),
  list(name = "joe_rot180", family = 16, par = 2.5),
  list(name = "joe_rot90", family = 26, par = -2.5),
  list(name = "joe_rot270", family = 36, par = -2.5),
  list(name = "bb1", family = 7, par = 1.0, par2 = 1.8),
  list(name = "bb1_rot180", family = 17, par = 1.0, par2 = 1.8),
  list(name = "bb1_rot90", family = 27, par = -1.0, par2 = -1.8),
  list(name = "bb1_rot270", family = 37, par = -1.0, par2 = -1.8),
  list(name = "bb6", family = 8, par = 1.8, par2 = 1.6),
  list(name = "bb6_rot180", family = 18, par = 1.8, par2 = 1.6),
  list(name = "bb6_rot90", family = 28, par = -1.8, par2 = -1.6),
  list(name = "bb6_rot270", family = 38, par = -1.8, par2 = -1.6),
  list(name = "bb7", family = 9, par = 1.5, par2 = 1.5),
  list(name = "bb7_rot180", family = 19, par = 1.5, par2 = 1.5),
  list(name = "bb7_rot90", family = 29, par = -1.5, par2 = -1.5),
  list(name = "bb7_rot270", family = 39, par = -1.5, par2 = -1.5),
  list(name = "bb8", family = 10, par = 2.0, par2 = 0.8),
  list(name = "bb8_rot180", family = 20, par = 2.0, par2 = 0.8),
  list(name = "bb8_rot90", family = 30, par = -2.0, par2 = -0.8),
  list(name = "bb8_rot270", family = 40, par = -2.0, par2 = -0.8),
  list(name = "tawn1", family = 104, par = 2.0, par2 = 0.6),
  list(name = "tawn1_rot180", family = 114, par = 2.0, par2 = 0.6),
  # VineCopula 124/134 == rscopulas Tawn2 R90/R270 (see header).
  list(name = "tawn1_rot90", family = 124, par = -2.0, par2 = 0.6),
  list(name = "tawn1_rot270", family = 134, par = -2.0, par2 = 0.6),
  list(name = "tawn2", family = 204, par = 2.0, par2 = 0.6),
  list(name = "tawn2_rot180", family = 214, par = 2.0, par2 = 0.6),
  # VineCopula 224/234 == rscopulas Tawn1 R90/R270 (see header).
  list(name = "tawn2_rot90", family = 224, par = -2.0, par2 = 0.6),
  list(name = "tawn2_rot270", family = 234, par = -2.0, par2 = 0.6)
)

for (idx in seq_along(pair_cases)) {
  case <- pair_cases[[idx]]
  par2 <- if (is.null(case$par2)) 0 else case$par2
  payload <- pair_case(case$name, case$family, case$par, par2, seed = 20260900L + idx)
  write_fixture(sprintf("pair_fit_%s.json", case$name), payload)
  cat(sprintf(
    "pair %-16s code=%4d  R: par=%9.5f par2=%8.5f loglik=%10.4f tau=%7.4f\n",
    case$name, case$family, payload$r_par, payload$r_par2, payload$r_loglik, payload$r_tau
  ))
}

# --- C-vine fits with a fixed order -------------------------------------------

# Per-tree dependence strength keeps every rotation unambiguous at n = 2000.
gaussian_rho_by_level <- list(
  c(0.65, -0.5, 0.55, 0.6),
  c(0.35, -0.4, 0.3),
  c(-0.25, 0.3),
  c(0.2)
)
clayton_theta_by_level <- list(
  c(2.2, 1.8, 2.5, 2.0),
  c(1.3, 1.5, 1.2),
  c(1.0, 0.9),
  c(0.8)
)

cvine_case <- function(name, order, kind, seed) {
  d <- length(order)
  template <- C2RVine(order, family = rep(1, d * (d - 1) / 2), par = rep(0.1, d * (d - 1) / 2))
  M <- template$Matrix
  family <- matrix(0, d, d)
  par <- matrix(0, d, d)
  par2 <- matrix(0, d, d)

  set.seed(seed)
  for (j in seq_len(d - 1)) {
    for (i in (j + 1):d) {
      level <- d - i + 1
      slot <- j
      if (kind == "gaussian") {
        family[i, j] <- 1
        par[i, j] <- gaussian_rho_by_level[[level]][slot]
      } else if (kind == "clayton") {
        rotation_code <- sample(c(3, 13, 23, 33), 1)
        theta <- clayton_theta_by_level[[level]][slot]
        family[i, j] <- rotation_code
        par[i, j] <- if (rotation_code %in% c(23, 33)) -theta else theta
      } else {
        stop("unknown kind")
      }
    }
  }
  rvm <- RVineMatrix(Matrix = M, family = family, par = par, par2 = par2)
  if (RVineMatrixCheck(rvm$Matrix) != 1) {
    stop("generated vine matrix is invalid")
  }

  set.seed(seed + 1L)
  data <- signif(clamp_unit(RVineSim(n_obs, rvm)), 12)
  fit <- RVineSeqEst(data, rvm, method = "mle", se = TRUE)
  loglik <- RVineLogLik(data, fit)$loglik

  edges <- list()
  for (j in seq_len(d - 1)) {
    for (i in (j + 1):d) {
      level <- d - i + 1
      code <- family[i, j]
      mapped <- rscopulas_family_code(code)
      conditioning <- if (i < d) as.integer(M[(i + 1):d, j] - 1L) else integer(0)
      edges[[length(edges) + 1]] <- list(
        tree = level,
        conditioned = I(as.integer(c(M[i, j], M[j, j]) - 1L)),
        conditioning = as.list(conditioning),
        family_code = code,
        family = mapped$family,
        rotation = mapped$rotation,
        true_par = par[i, j],
        true_params = I(rscopulas_params(code, par[i, j], par2[i, j])),
        r_par = fit$par[i, j],
        r_par2 = fit$par2[i, j],
        r_params = I(rscopulas_params(code, fit$par[i, j], fit$par2[i, j])),
        r_se = I(c(fit$se[i, j])),
        r_pair_loglik = fit$pair.logLik[i, j]
      )
    }
  }

  list(
    metadata = metadata,
    case = name,
    kind = kind,
    dim = d,
    n = n_obs,
    seed = seed,
    r_order = as.integer(order),
    order = as.integer(order - 1L),
    r_matrix = to_rows(M),
    r_family_matrix = to_rows(family),
    r_loglik = loglik,
    edges = edges,
    data = to_rows(data)
  )
}

cvine_cases <- list(
  list(name = "gaussian_d4", order = c(3, 1, 4, 2), kind = "gaussian"),
  list(name = "gaussian_d5", order = c(2, 5, 1, 4, 3), kind = "gaussian"),
  list(name = "clayton_d4", order = c(4, 2, 1, 3), kind = "clayton"),
  list(name = "clayton_d5", order = c(3, 1, 5, 2, 4), kind = "clayton")
)

for (idx in seq_along(cvine_cases)) {
  case <- cvine_cases[[idx]]
  payload <- cvine_case(case$name, case$order, case$kind, seed = 20261000L + 10L * idx)
  write_fixture(sprintf("cvine_fit_%s.json", case$name), payload)
  cat(sprintf("cvine %-12s d=%d loglik=%10.4f edges=%d\n", case$name, payload$dim, payload$r_loglik, length(payload$edges)))
}
