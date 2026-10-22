# Attribution and third-party notices

This example is original work by Nikhil Reddy Donapati, released under the MIT
license (see `LICENSE`), the same license OpenDP uses. The license can be
changed to whatever the OpenDP maintainers require when the example is placed
in their repository.

AI assistance (Claude, Anthropic) was used in writing and testing the code and
documentation; this is disclosed in this notice.

## Libraries used (not vendored; installed via `requirements.txt`)

| Library | License | Used for |
|---|---|---|
| OpenDP | MIT | measurements, composition, odometer |
| Opacus | Apache-2.0 | per-example gradients, clipping, noise (`DPOptimizer`), debug sampler |
| PyTorch | BSD-3-Clause | model and training |
| NumPy | BSD-3-Clause | synthetic data |
| SciPy | BSD-3-Clause | test-only: numerical integration reference |
| dp-accounting (Google) | Apache-2.0 | test and notebook cross-check only |

No source code is copied from any of these projects.

## Methods

* Sampled-Gaussian RDP: Mironov, Talwar & Zhang (2019), implemented
  independently.
* RDP → (ε, δ): Balle et al. (2020), Thm. 21; Canonne, Kamath & Steinke (2020).
* Adaptive RDP composition: Mironov (2017).
* Floating-point mitigation: Opacus `secure_mode`, after Holohan & Braghin (2021).

## Prior work by the author

Design lessons, not code, come from FedCRM-DP V2
(https://github.com/nikhilred2619/FedCRM-DP-V2-Reproducible-Research).
