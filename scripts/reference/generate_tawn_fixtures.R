#!/usr/bin/env Rscript

# Generate reference fixtures for the Tawn1 and Tawn2 pair copulas
# (VineCopula family codes 104/114/124/134 and 204/214/224/234).
# Run from repo root.

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

u1 <- c(0.17, 0.31, 0.62, 0.88)
u2 <- c(0.23, 0.54, 0.41, 0.79)
p <- c(0.27, 0.45, 0.73, 0.91)

# Tawn1 / Tawn2: par = θ ∈ [1, 20], par2 = ψ ∈ [0, 1].
#
# Note on rotations: VineCopula's R90/R270 convention for Tawn swaps the
# type index (e.g. family 124 "Tawn1 R90" is actually Tawn2 evaluated at
# (1 - u, v) — see the experiment in scripts/reference/ for evidence). That
# is not a standard copula rotation; we only emit and cross-validate the R0
# and R180 variants here. Rotated-negative-dep Tawn is still usable in the
# fitter via the standard-rotation machinery — just without VineCopula
# parity at the fixture level.
cases <- list(
  list(name = "tawn1", family = 104, par = 2.0, par2 = 0.6),
  list(name = "tawn1_rot180", family = 114, par = 2.0, par2 = 0.6),
  list(name = "tawn2", family = 204, par = 2.0, par2 = 0.6),
  list(name = "tawn2_rot180", family = 214, par = 2.0, par2 = 0.6)
)

for (case in cases) {
  par_stored <- case$par
  fixture <- list(
    metadata = metadata,
    family = case$name,
    family_code = case$family,
    par = par_stored,
    par2 = case$par2,
    u1 = as.list(u1),
    u2 = as.list(u2),
    p = as.list(p),
    expected_log_pdf = as.numeric(log(BiCopPDF(u1, u2, family = case$family, par = case$par, par2 = case$par2))),
    expected_cond_first_given_second = as.numeric(BiCopHfunc2(u1, u2, family = case$family, par = case$par, par2 = case$par2)),
    expected_cond_second_given_first = as.numeric(BiCopHfunc1(u1, u2, family = case$family, par = case$par, par2 = case$par2)),
    expected_inv_first_given_second = as.numeric(BiCopHinv2(p, u2, family = case$family, par = case$par, par2 = case$par2)),
    expected_inv_second_given_first = as.numeric(BiCopHinv1(u1, p, family = case$family, par = case$par, par2 = case$par2))
  )
  write_fixture(sprintf("pair_%s_case01.json", case$name), fixture)
}
