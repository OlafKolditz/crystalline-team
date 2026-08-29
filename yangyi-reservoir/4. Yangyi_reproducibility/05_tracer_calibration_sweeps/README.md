# Three-parameter joint sweep

Cases: 20 (16 Latin-hypercube points + 4 anchors).
Deep release time: 60 days.
Permeability is fixed. Aperture is scaled deterministically as A/A0=fpipe/(2/7).
Feedback tolerance: 0.1 ppb.

Best coarse case: anchor_80C.
- fpipe+storage: 33.3333%.
- implied pipe temperature: 80.00 C.
- alphaL: 5 m.
- Rdeep: 70.0000%.
- gross recovery: 274.812 kg (target 277.642 kg).
- ZK208 peak: 4.990 d, 464.674 ppb; FWHM=1.674 d; RMSE=167.468 ppb.
- ZK203 peak: 32.260 d, 151.213 ppb; FWHM=9.766 d; RMSE=38.033 ppb.
- maximum pressure RMSE: 0.00080 MPa.
- balanced score: 1.18192.

The balanced score is the unweighted sum of normalized full-curve RMSE, peak-time error, peak-height error, FWHM error, cumulative-mass error, and maximum pressure RMSE in MPa.
