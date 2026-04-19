# Benchmark Summary

Generated at: 2026-04-18T08:34:31.830586+00:00

## mixed_r_vine_fit

| Implementation | Mean | Total | Iterations | Relative to R |
| --- | ---: | ---: | ---: | ---: |
| python | 6.450ms | 322.475ms | 50 | 3.43x |
| rust | 7.091ms | 354.536ms | 50 | 3.12x |
| r | 22.140ms | 1.107s | 50 | - |

## mixed_r_vine_log_pdf

| Implementation | Mean | Total | Iterations | Relative to R |
| --- | ---: | ---: | ---: | ---: |
| rust | 22.858us | 11.429ms | 500 | 113.57x |
| python | 23.481us | 11.741ms | 500 | 110.56x |
| r | 2.596ms | 1.298s | 500 | - |

## mixed_r_vine_sample

| Implementation | Mean | Total | Iterations | Relative to R |
| --- | ---: | ---: | ---: | ---: |
| python | 27.939ms | 8.382s | 300 | 4.04x |
| rust | 28.445ms | 8.534s | 300 | 3.97x |
| r | 112.870ms | 33.861s | 300 | - |
