# R reference fits for generate_evt_fixtures.py, which calls this script.
#
#   Rscript scripts/reference/generate_evt_fixtures.R <dir>
#
# <dir> holds one file per case, `gpd_<name>.txt` or `gev_<name>.txt`, with
# one observation per line. Results are printed as `kind name key value`
# lines. Needs only evd and extRemes (no JSON package).
suppressMessages({
  library(evd)
  library(extRemes)
})

dir <- commandArgs(trailingOnly = TRUE)[1]
emit <- function(kind, name, key, value) {
  cat(kind, name, key, format(value, digits = 17), "\n")
}
control <- list(reltol = 1e-14, maxit = 20000)

for (file in list.files(dir, pattern = "^gpd_.*\\.txt$")) {
  name <- sub("^gpd_(.*)\\.txt$", "\\1", file)
  y <- scan(file.path(dir, file), quiet = TRUE)
  fit <- fpot(y, threshold = 0, model = "gpd", std.err = TRUE, control = control)
  emit("gpd", name, "scale", fit$estimate["scale"])
  emit("gpd", name, "shape", fit$estimate["shape"])
  emit("gpd", name, "loglik", -fit$deviance / 2)
  emit("gpd", name, "se_scale", fit$std.err["scale"])
  emit("gpd", name, "se_shape", fit$std.err["shape"])
}

for (file in list.files(dir, pattern = "^gev_.*\\.txt$")) {
  name <- sub("^gev_(.*)\\.txt$", "\\1", file)
  x <- scan(file.path(dir, file), quiet = TRUE)
  fit <- fgev(x, std.err = TRUE, control = control)
  emit("gev", name, "location", fit$estimate["loc"])
  emit("gev", name, "scale", fit$estimate["scale"])
  emit("gev", name, "shape", fit$estimate["shape"])
  emit("gev", name, "loglik", -fit$deviance / 2)
  emit("gev", name, "se_location", fit$std.err["loc"])
  emit("gev", name, "se_scale", fit$std.err["scale"])
  emit("gev", name, "se_shape", fit$std.err["shape"])
  # extRemes: delta-method ("normal approximation") return-level interval.
  ext <- fevd(x, type = "GEV")
  for (period in c(10, 100, 1000)) {
    interval <- ci(ext, return.period = period, alpha = 0.05)
    emit("gev", name, paste0("level_", period), interval[2])
    emit("gev", name, paste0("level_se_", period),
         (interval[3] - interval[1]) / (2 * qnorm(0.975)))
  }
}
