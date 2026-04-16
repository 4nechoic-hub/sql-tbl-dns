# Data provenance and attribution

## Upstream source

This repository uses third-party research data from the **KTH Boundary Layer Data** portal maintained by Philipp Schlatter and collaborators.

Primary upstream source used here:

- **KTH Boundary Layer Data portal**
- **DNS release**: "new DNS Data, last update: 2012-05-27"
- **Reference**: Schlatter, P. and Orlu, R. (2010). *Assessment of direct numerical simulation data of turbulent boundary layers*. Journal of Fluid Mechanics, 659, 116-126. https://doi.org/10.1017/S0022112010003113

## What the upstream portal says

The portal describes the directory as containing integral quantities and velocity profiles for selected streamwise positions from DNS and LES of a zero-pressure-gradient turbulent boundary layer. It also states that the data is free to use and asks users to include proper references to the original publications.

Upstream contact listed on the portal:

- Philipp Schlatter
- pschlatt@mech.kth.se

## Files represented in this project

This project is built around the DNS profile and budget files for the Reynolds-number cases:

- Re_theta = 670
- Re_theta = 1000
- Re_theta = 1410
- Re_theta = 2000
- Re_theta = 2540
- Re_theta = 3030
- Re_theta = 3270
- Re_theta = 3630
- Re_theta = 3970
- Re_theta = 4060

Typical upstream file patterns:

- `vel_XXXX.prof`
- `bud_XXXX_dns_k.prof`
- `bud_XXXX_dns_uu.prof`
- `bud_XXXX_dns_vv.prof`
- `bud_XXXX_dns_ww.prof`
- `bud_XXXX_dns_uv.prof`

The 2012-05-27 update also notes inclusion of vorticity rms for all Reynolds numbers.

## Attribution guidance for this repository

Please keep the following distinction clear:

- **Repository code and pipeline implementation**: authored in this project
- **Raw DNS data**: created by the original KTH research effort and distributed through the KTH data portal

Recommended README acknowledgment:

> This repository uses turbulent boundary layer profile and budget data made available through the KTH Boundary Layer Data portal. Please cite the original publication by Schlatter and Orlu (2010) when using the upstream data.

## Recommended citation for upstream data

Schlatter, P. and Orlu, R. (2010). *Assessment of direct numerical simulation data of turbulent boundary layers*. Journal of Fluid Mechanics, 659, 116-126. https://doi.org/10.1017/S0022112010003113

## Suggested BibTeX

```bibtex
@article{schlatter_orlu_2010,
  author  = {Schlatter, Philipp and Orlu, Ramis},
  title   = {Assessment of direct numerical simulation data of turbulent boundary layers},
  journal = {Journal of Fluid Mechanics},
  volume  = {659},
  pages   = {116--126},
  year    = {2010},
  doi     = {10.1017/S0022112010003113}
}
```

## License separation

The repository may be MIT licensed for **code**, but that should not be interpreted as relicensing the upstream research data. Keep `LICENSE` for code and this file for data provenance and attribution.
