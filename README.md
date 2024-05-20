# DBT Projection Interpolation

Codes used to train a Residual Refinement Interpolation Network (RRIN) using digital breast tomosynthesis (DBT) projections, and further synthesize projections based on the neighbour ones. The model was used in the published [paper](https://doi.org/10.1117/12.2625748):paper:

> Arthur C. Costa, Rodrigo B. Vimieiro, Lucas R. Borges, Bruno Barufaldi, Andrew D. A. Maidment, Marcelo A. C. Vieira, "Assessment of video frame interpolation network to generate digital breast tomosynthesis projections," Proc. SPIE 12286, 16th International Workshop on Breast Imaging (IWBI2022), 122861D (13 July 2022).

The scripts were adapted from the RRIN from Haopeng Li, Yuan Yuan, Qi Wang, IEEE Conference on Acoustics, Speech, and Signal Processing, ICASSP 2020 ([DOI](10.1109/ICASSP40776.2020.9053987)) [github](https://github.com/HopLee6/RRIN-PyTorch).

The Tunable-UNet principle was used based on the work of S. Kang, S. Uchida and B. K. Iwana, "Tunable U-Net: Controlling Image-to-Image Outputs Using a Tunable Scalar Value," in IEEE Access, vol. 9, pp. 103279-103290, 2021, ([DOI](10.1109/ACCESS.2021.3096530)).
 
