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

u1 <- c(0.17, 0.31, 0.62, 0.88)
u2 <- c(0.23, 0.54, 0.41, 0.79)
p <- c(0.27, 0.45, 0.73, 0.91)

cases <- list(
  list(name = "gaussian", family = 1, par = 0.7, par2 = 0),
  list(name = "student_t", family = 2, par = 0.6, par2 = 5),
  list(name = "clayton", family = 3, par = 2.0, par2 = 0),
  list(name = "frank", family = 5, par = 4.0, par2 = 0),
  list(name = "gumbel", family = 4, par = 1.8, par2 = 0),
  list(name = "clayton_rot90", family = 23, par = -2.0, par2 = 0),
  list(name = "clayton_rot180", family = 13, par = 2.0, par2 = 0),
  list(name = "clayton_rot270", family = 33, par = -2.0, par2 = 0),
  list(name = "gumbel_rot90", family = 24, par = -1.8, par2 = 0),
  list(name = "gumbel_rot180", family = 14, par = 1.8, par2 = 0),
  list(name = "gumbel_rot270", family = 34, par = -1.8, par2 = 0)
)

for (case in cases) {
  fixture <- list(
    metadata = metadata,
    family = case$name,
    family_code = case$family,
    par = if (case$family %in% c(23, 33, 24, 34)) abs(case$par) else case$par,
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
