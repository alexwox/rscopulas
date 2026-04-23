#!/usr/bin/env Rscript

# Generate reference fixtures for the Joe pair copula (VineCopula family 6).
#
# Mirrors the structure of generate_paircopula_fixtures.R — emits one JSON per
# rotation case under fixtures/reference/vinecopula/v2/. Run from repo root:
#   Rscript scripts/reference/generate_joe_fixtures.R
# Requires R with the VineCopula package installed.

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

# VineCopula family codes for Joe:
#   6  → R0 (no rotation)
#   16 → R180
#   26 → R90  (expects negative par)
#   36 → R270 (expects negative par)
cases <- list(
  list(name = "joe", family = 6, par = 2.5, par2 = 0),
  list(name = "joe_rot90", family = 26, par = -2.5, par2 = 0),
  list(name = "joe_rot180", family = 16, par = 2.5, par2 = 0),
  list(name = "joe_rot270", family = 36, par = -2.5, par2 = 0)
)

for (case in cases) {
  fixture <- list(
    metadata = metadata,
    family = case$name,
    family_code = case$family,
    par = if (case$family %in% c(26, 36)) abs(case$par) else case$par,
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
