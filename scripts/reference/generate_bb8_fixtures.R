#!/usr/bin/env Rscript

# Generate reference fixtures for the BB8 pair copula (VineCopula family 10).
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

# BB8: θ ≥ 1, δ ∈ (0, 1]. Rotations 30/40 expect par negated; δ stays in
# (-1, 0] for rotated families per VineCopula's bounds.
cases <- list(
  list(name = "bb8", family = 10, par = 2.0, par2 = 0.8),
  list(name = "bb8_rot90", family = 30, par = -2.0, par2 = -0.8),
  list(name = "bb8_rot180", family = 20, par = 2.0, par2 = 0.8),
  list(name = "bb8_rot270", family = 40, par = -2.0, par2 = -0.8)
)

for (case in cases) {
  par_stored <- if (case$family %in% c(30, 40)) abs(case$par) else case$par
  par2_stored <- if (case$family %in% c(30, 40)) abs(case$par2) else case$par2
  fixture <- list(
    metadata = metadata,
    family = case$name,
    family_code = case$family,
    par = par_stored,
    par2 = par2_stored,
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
