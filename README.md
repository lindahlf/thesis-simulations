# thesis-simulations

Simulations of various detection methods for sparse and weak mixtures in high-dimensional data. Part of my master thesis.

### The purpose of this repo is to collect all code needed for the simulations I need for the report.  

The functionalities contained in code in this repo are

- Implementations of some statistics used for detection in sparse and weak data including,
  - Higher Criticism, as popularized by Donoho and Jin (2004)
  - phi-divergence goodness of fit tests, introduced by Jager and Weller (2007) 
  - Recently proposed weighted Kolmogorov-Smirnoff procedures (2018)
- Specific tests of above statistics to demonstrate their relative merits. Examples of such tests are
  - How the scores are separated for between the null and alternative hypotheses.
  - How the sum of type I and II errors change as a function of sparsity or weakness.
  
In particular, the above functionalities will be used to investigate how these methods perform close to and on the so-called detection boundary (see Donoho and Jin (2004)).
