# 31_honest_did_extensive.R
# Honest DiD (Rambachan & Roth 2023) sensitivity on the EXTENSIVE MARGIN, called twice by
# 47_slides_binary.py: for the Kenya riot event study (prefix honest_ke) and for the pooled
# lead-specific riot gap (prefix honest_gap). Reads the betahat and block-clustered vcov
# that 47 exports to output/panels/slides_bin/ and writes RM (Delta^RM) and SD (Delta^SD)
# sensitivity figures + the M-bar thresholds where the bound crosses zero
# (<prefix>_extensive_results.txt, read back by 47).

suppressMessages({
  library(HonestDiD)
  library(ggplot2)
  library(Matrix)
})

args <- commandArgs(trailingOnly = TRUE)
PANELS  <- args[1]
FIGDIRS <- strsplit(args[2], ",")[[1]]
# Optional third argument: top of the smoothness (Delta^SD) grid, in outcome units.
MMAX    <- if (length(args) >= 3) as.numeric(args[3]) else 0.1
# Optional fourth argument: input/output file prefix (default "honest_ke": the Kenya DD
# event study). "honest_gap" = pooled lead-specific gap event study (47_slides_binary.py),
# which is written as ONE two-panel figure fig_honest_did_gap.pdf.
PREFIX  <- if (length(args) >= 4) args[4] else "honest_ke"

beta_df <- read.csv(file.path(PANELS, paste0(PREFIX, "_betahat.csv")))
sigma   <- as.matrix(read.csv(file.path(PANELS, paste0(PREFIX, "_sigma.csv"))))
meta    <- readLines(file.path(PANELS, paste0(PREFIX, "_meta.txt")))
n_pre   <- as.integer(sub("n_pre=", "", meta[grep("n_pre", meta)]))
n_post  <- as.integer(sub("n_post=", "", meta[grep("n_post", meta)]))

betahat <- beta_df$betahat
stopifnot(length(betahat) == n_pre + n_post)
dimnames(sigma) <- NULL

# Conley spatial-HAC vcov can be slightly non-PSD; project to nearest PSD matrix
# so HonestDiD's optimisation is well-defined.
ev <- min(eigen(sigma, symmetric = TRUE, only.values = TRUE)$values)
if (ev < 0) {
  cat(sprintf("sigma min eigenvalue %.4f < 0; projecting to nearest PSD\n", ev))
  sigma <- as.matrix(nearPD(sigma, corr = FALSE, keepDiag = TRUE)$mat)
}

# Target the average post-treatment effect (matches "the Kenya riot coefficient")
lvec <- rep(1 / n_post, n_post)
Mbarvec <- seq(0, 0.6, by = 0.1)

# Relative magnitudes restriction (Delta^RM), C-LF method
rm <- createSensitivityResults_relativeMagnitudes(
  betahat = betahat, sigma = sigma,
  numPrePeriods = n_pre, numPostPeriods = n_post,
  Mbarvec = Mbarvec, l_vec = lvec
)

# Smoothness restriction (Delta^SD), FLCI method
Mvec <- seq(0, MMAX, length.out = 11)
sd <- createSensitivityResults(
  betahat = betahat, sigma = sigma,
  numPrePeriods = n_pre, numPostPeriods = n_post,
  method = "FLCI", Mvec = Mvec, l_vec = lvec
)

# Find where each lower bound crosses zero
cross_zero <- function(df, xcol) {
  df <- df[order(df[[xcol]]), ]
  below <- which(df$lb <= 0)
  if (length(below) == 0) return(NA)
  df[[xcol]][below[1]]
}
rm_lb0   <- rm$lb[which.min(rm$Mbar)]      # lb at Mbar=0
rm_cross <- cross_zero(rm, "Mbar")
# The 0.1 grid only brackets the crossing; refine it in steps of 0.01 inside that bracket
if (!is.na(rm_cross) && rm_cross > 0) {
  rm_fine <- createSensitivityResults_relativeMagnitudes(
    betahat = betahat, sigma = sigma,
    numPrePeriods = n_pre, numPostPeriods = n_post,
    Mbarvec = seq(rm_cross - 0.1 + 0.01, rm_cross, by = 0.01), l_vec = lvec
  )
  rm_cross <- cross_zero(rm_fine, "Mbar")
}
sd_lb0   <- sd$lb[which.min(sd$M)]
sd_cross <- cross_zero(sd, "M")

cat(sprintf("RM: lb at Mbar=0 is %.3f; crosses zero at Mbar = %.2f (0.01 grid)\n", rm_lb0, rm_cross))
cat(sprintf("SD: lb at M=0   is %.3f; crosses zero at M    ~ %.3f\n", sd_lb0, sd_cross))

# Save thresholds for macro generation
writeLines(c(
  sprintf("rm_lb0=%.3f", rm_lb0),
  sprintf("rm_cross=%.2f", rm_cross),
  sprintf("sd_lb0=%.3f", sd_lb0),
  sprintf("sd_cross=%.3f", sd_cross)
), file.path(PANELS, paste0(PREFIX, "_extensive_results.txt")))

# Figures
p_rm <- createSensitivityPlot_relativeMagnitudes(rm,
          originalResults = constructOriginalCS(betahat, sigma, n_pre, n_post,
                                                l_vec = lvec))
p_sd <- createSensitivityPlot(sd,
          originalResults = constructOriginalCS(betahat, sigma, n_pre, n_post,
                                                l_vec = lvec))

# Tufte styling: replace the default grey ggplot panel with a clean white
# theme so these match the matplotlib figures elsewhere in the paper.
clean_theme <- theme_bw(base_size = 12) +
  theme(panel.grid.minor = element_blank(),
        panel.border      = element_blank(),
        axis.line         = element_line(colour = "grey40"),
        legend.position   = "bottom",
        legend.title      = element_blank())
# House style: no rotated y-axis title; put it horizontally above the panel.
top_ylab <- function(p) {
  lab <- p$labels$y
  if (is.null(lab) || identical(lab, "")) return(p + labs(y = NULL))
  p + labs(y = NULL, subtitle = lab) +
    theme(plot.subtitle = element_text(size = rel(0.9), hjust = 0))
}
p_rm <- top_ylab(p_rm + clean_theme)
p_sd <- top_ylab(p_sd + clean_theme)

for (fd in FIGDIRS) {
  if (PREFIX == "honest_ke") {
    ggsave(file.path(fd, "fig_honest_did_kenya.pdf"), p_rm, width = 6, height = 4)
    ggsave(file.path(fd, "fig_honest_did_kenya_sd.pdf"), p_sd, width = 6, height = 4)
  } else {
    suppressMessages(library(patchwork))
    p2 <- (p_rm + labs(title = "Relative magnitudes")) +
          (p_sd + labs(title = "Smoothness"))
    ggsave(file.path(fd, paste0("fig_", sub("^honest_", "honest_did_", PREFIX), ".pdf")),
           p2, width = 11, height = 4.2)
  }
}
cat("Saved honest DiD figures (extensive margin)\n")
