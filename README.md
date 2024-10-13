# hyPIRana

see also [hyPIRana release v.1.0.0 on BioPOLIMgit](https://github.com/BioPOLIM/hyPIRana/releases/tag/hyPIRana)
for updates see: [hyPIRana on BioPOLIMgit](https://github.com/BioPOLIM/hyPIRana)

Analysis software development for IR-PiFM / PiF-IR scan images and hyperspectral Scans 

This sofware is part of hyperspectral analysis methods used in the published article: "Nanoscale chemical characterization of secondary protein structure of F-Actin using mid-infrared photoinduced force microscopy (PiF-IR)" by Jesvin Joseph, Lukas Spantzel, Maryam Ali, Dijo Moonnukandathil Joseph, Sebastian Unger, Katharina Reglinski, Christoph Krafft, Anne-Dorothea Müller, Christian Eggeling, Rainer Heintzmann, Michael Börsch, Adrian T. Press, Daniela Täuber. Spectrochimica Acta part A: Molecular and Biomolecular Spectroscopy, 306, 123612, 2024. https://doi.org/10.1016/j.saa.2023.123612

Contributions in this code so far have been made by Sebastian Unger, Maryam Ali, René Lachmann, Rainer Heintzmann, Mohammad Soltaninezhad and Daniela Täuber. For details see headers of the code.

This code analyzes the hyperspectral data by:
- Plotting generated images: AFM Topography, integrated PiF-IR hyperspectral image, and AFM phase image.
- Plotting the mean spectrum on PiFM intensity.
- Principal Component Analysis PCA represented by loaded components plot, scatter plot, and individual component factor plots.

In the context of the article by Joseph et al. Spectrochimica Acta part A: Molecular and Biomolecular Sepctroscopy, 306, 123612, 2024, two PiF-IR hyperspectral data sets have been analysed using hyPIRana. The Raw data are available in this repository:
- single-fibrillar F-Actin
- crosslinked F-Actin
