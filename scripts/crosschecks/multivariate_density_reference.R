# Writes multi.json next to the working directory: reference log-densities from R copula.
suppressMessages({library(copula)})
num <- function(x) { s <- sprintf("%.17g", x); s[!is.finite(x)] <- "null"; s }
vec <- function(x) paste0("[", paste(num(x), collapse=","), "]")
mat <- function(m) paste0("[", paste(apply(m, 1, vec), collapse=","), "]")
js <- function(c) paste0('{"name":"',c$name,'","family":"',c$family,'","dim":',c$dim,',"par":',num(c$par),',"u":',mat(c$u),',"logpdf":',vec(c$logpdf), if (!is.null(c$corr)) paste0(',"corr":',mat(c$corr)) else "", "}")
set.seed(42)
out <- list()
pts <- function(d, n=60) { u <- matrix(runif(n*d), n, d); u[1:10,] <- u[1:10,]*0.02; u[11:20,] <- 1-u[11:20,]*0.02; u[21:25,] <- matrix(rep(runif(5),d),5,d)*0.999+0.0005+matrix(runif(5*d,-4e-4,4e-4),5,d); u }
add <- function(name, fam, d, par, cop, u, extra=list()) out[[length(out)+1]] <<- c(list(name=name, family=fam, dim=d, par=par, u=u, logpdf=dCopula(u, cop, log=TRUE)), extra)
for (d in c(2,3,5,8)) {
  u <- pts(d)
  for (th in c(0.3, 2, 9)) add("clayton","clayton",d,th,claytonCopula(th,dim=d),u)
  for (th in c(1.1, 2.5, 8)) add("gumbel","gumbel",d,th,gumbelCopula(th,dim=d),u)
  for (th in c(0.5, 4, 15)) add("frank","frank",d,th,frankCopula(th,dim=d),u)
  for (k in 1:2) {
    A <- matrix(rnorm(d*d), d, d); S <- cov2cor(A %*% t(A) + diag(d)*ifelse(k==1,1,0.05))
    add("gaussian","gaussian",d,0,normalCopula(P2p(S),dim=d,dispstr="un"),u,list(corr=S))
    for (nu in c(2.5, 7, 30)) add("student_t","student_t",d,nu,tCopula(P2p(S),dim=d,df=nu,dispstr="un"),u,list(corr=S))
  }
}
writeLines(paste0("[", paste(sapply(out, js), collapse=",\n"), "]"), "multi.json")
