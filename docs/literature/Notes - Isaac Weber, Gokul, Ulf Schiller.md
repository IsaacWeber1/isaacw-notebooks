## Jun 11, 2025 | [Isaac Weber, Gokul, Ulf Schiller](https://www.google.com/calendar/event?eid=MWxpZzhzdThramdpbWtha2V1b2hkNWpwNHUgdXNjaGlsbEB1ZGVsLmVkdQ)

Attendees: [Gokul Raman Arumugam Kumar](mailto:gokul@udel.edu) [Isaac Weber](mailto:isaacw@udel.edu) [Ulf Schiller](mailto:uschill@udel.edu)

Notes

* Works in NC in summer, potentially put in hourly work in fall  
* Isaac most interested in learning LBM collision operator  
* Learn transition probabilities using diffusion models

Action items

- [x] ~~prepare reading list for Isaac~~  
- [ ] Set up DARWIN account for Isaac  
- [ ] Next meeting: invite Isaac to join group meeting

**Ideas/Objectives**

* Develop surrogate models for lattice gas collision operator (transition probabilities)  
* use data from our fluctuating LB models to train diffusion models  
* potentially use Wagner’s data for MD lattice gas

**Reading list**

*Lattice Boltzmann Methods*

A tutorial review article on the broader context of the simulation methods we use and a textbook on lattice Boltzmann methods written by my colleagues Timm Krueger and Halim Kusumaatmaja.

1. Schiller, Ulf D., et al. “Mesoscopic Modelling and Simulation of Soft Matter.” *Soft Matter*, vol. 14, no. 1, Dec. 2017, pp. 9–26, [https://doi.org/10.1039/C7SM01711A](https://doi.org/10.1039/C7SM01711A).  
2. Krüger, Timm, et al. *The Lattice Boltzmann Method: Principles and Practice*. Springer International Publishing, 2017, [https://doi.org/10.1007/978-3-319-44649-3](https://doi.org/10.1007/978-3-319-44649-3).

*Fluctuating Lattice Boltzmann*

Some technical papers on the fluctuating lattice Boltzmann method. No worries if you don’t understand all the details.

3. Dünweg, Burkhard, et al. “Statistical Mechanics of the Fluctuating Lattice Boltzmann Equation.” *Phys. Rev. E*, vol. 76, 2007, p. 036704, [https://doi.org/10.1103/PhysRevE.76.036704](https://doi.org/10.1103/PhysRevE.76.036704).  
4. Gross, M., et al. “Thermal Fluctuations in the Lattice Boltzmann Method for Nonideal Fluids.” *Physical Review E*, vol. 82, no. 5, Nov. 2010, p. 056714, [https://doi.org/10.1103/PhysRevE.82.056714](https://doi.org/10.1103/PhysRevE.82.056714).  
5. Belardinelli, D., et al. “Fluctuating Multicomponent Lattice Boltzmann Model.” *Physical Review E*, vol. 91, no. 2, Feb. 2015, p. 023313, [https://doi.org/10.1103/PhysRevE.91.023313](https://doi.org/10.1103/PhysRevE.91.023313).

*Diffusion Models*

Some blog articles and papers on diffusion models. No worries if the technical details are overwhelming \- see if you can get the general ideas.

6. [https://lilianweng.github.io/posts/2021-07-11-diffusion-models/\#reverse-diffusion-process](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/#reverse-diffusion-process)  
7. [https://yang-song.net/blog/2021/score/](https://yang-song.net/blog/2021/score/)  
8. Dhariwal, Prafulla, and Alex Nichol. *Diffusion Models Beat GANs on Image Synthesis*. arXiv:2105.05233, arXiv, 1 June 2021, [https://doi.org/10.48550/arXiv.2105.05233](https://doi.org/10.48550/arXiv.2105.05233).  
9. Ho, Jonathan, et al. *Denoising Diffusion Probabilistic Models*. arXiv:2006.11239, arXiv, 16 Dec. 2020, [https://doi.org/10.48550/arXiv.2006.11239](https://doi.org/10.48550/arXiv.2006.11239).  
10. Sohl-Dickstein, Jascha, et al. *Deep Unsupervised Learning Using Nonequilibrium Thermodynamics*. arXiv:1503.03585, arXiv, 18 Nov. 2015, [https://doi.org/10.48550/arXiv.1503.03585](https://doi.org/10.48550/arXiv.1503.03585).  
11. Song, Jiaming, et al. *Denoising Diffusion Implicit Models*. arXiv:2010.02502, arXiv, 5 Oct. 2022, https://doi.org/10.48550/arXiv.2010.02502.  
12. Song, Yang, and Stefano Ermon. *Generative Modeling by Estimating Gradients of the Data Distribution*. arXiv:1907.05600, arXiv, 10 Oct. 2020, [https://doi.org/10.48550/arXiv.1907.05600](https://doi.org/10.48550/arXiv.1907.05600).  
13. Addition Refs: [https://github.com/diff-usion/Awesome-Diffusion-Models](https://github.com/diff-usion/Awesome-Diffusion-Models)  
14. Outlier and Explaining AI YouTube Videos  
15. [https://harshm121.medium.com/flow-matching-vs-diffusion-79578a16c510](https://harshm121.medium.com/flow-matching-vs-diffusion-79578a16c510)  
16. [https://github.com/harshm121/Diffusion-v-FlowMatching](https://github.com/harshm121/Diffusion-v-FlowMatching)

*Molecular dynamics lattice gas*

Some papers by my colleague Alexander Wagner at NDSU who mapped molecular dynamics simulations to a lattice representation. We might be able to collaborate with them and use their data for ML training.

17. Parsa, M. Reza, and Alexander J. Wagner. “Lattice Gas with Molecular Dynamics Collision Operator.” *Physical Review E*, vol. 96, no. 1, July 2017, p. 013314, [https://doi.org/10.1103/PhysRevE.96.013314](https://doi.org/10.1103/PhysRevE.96.013314).  
18. Pachalieva, Aleksandra, and Alexander J. Wagner. “Molecular Dynamics Lattice Gas Equilibrium Distribution Function for Lennard–Jones Particles.” *Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences*, vol. 379, no. 2208, Aug. 2021, p. 20200404, [https://doi.org/10.1098/rsta.2020.0404](https://doi.org/10.1098/rsta.2020.0404).  
19. Parsa, M. Reza, Aleksandra Pachalieva, et al. “Validity of the Molecular-Dynamics-Lattice-Gas Global Equilibrium Distribution Function.” *International Journal of Modern Physics C*, Oct. 2019, [https://doi.org/10.1142/S0129183119410079](https://doi.org/10.1142/S0129183119410079).  
20. Parsa, M. Reza, and Alexander J. Wagner. “Large Fluctuations in Nonideal Coarse-Grained Systems.” *Physical Review Letters*, vol. 124, no. 23, June 2020, p. 234501, [https://doi.org/10.1103/PhysRevLett.124.234501](https://doi.org/10.1103/PhysRevLett.124.234501).Pachalieva, Aleksandra, and Alexander J. Wagner. “Non-Gaussian Distribution of Displacements for Lennard-Jones Particles in Equilibrium.” *Physical Review E*, vol. 102, no. 5, Nov. 2020, p. 053310, [https://doi.org/10.1103/PhysRevE.102.053310](https://doi.org/10.1103/PhysRevE.102.053310).  
21. Parsa, M. Reza, Changho Kim, et al. “Nonuniqueness of Fluctuating Momentum in Coarse-Grained Systems.” *Physical Review E*, vol. 104, no. 1, July 2021, p. 015304, [https://doi.org/10.1103/PhysRevE.104.015304](https://doi.org/10.1103/PhysRevE.104.015304).

*(Advanced) Papers on Diffusion Models*

These papers discuss practical aspects of generative models. They are more advanced.

22. [https://arxiv.org/pdf/2503.18731](https://arxiv.org/pdf/2503.18731)  
23. [https://openreview.net/pdf?id=teE4pl9ftK](https://openreview.net/pdf?id=teE4pl9ftK)  
24. [https://openreview.net/pdf?id=0FbzC7B9xI](https://openreview.net/pdf?id=0FbzC7B9xI)  
    1. Isaac Link to Source Abstracts: [source\_analysis.docx](https://docs.google.com/document/d/1-cD1v7PvPQmBxTHTwV5vZ2AYbjxZqXrh/edit?usp=drive_link&ouid=112534752180151216024&rtpof=true&sd=true)

    

    

*8/6/2025 notes \[uschill\]*

* Literature search \- surrogate models for lattice Boltzmann

*These references may be more tangible to what we are going to do in practice:*

[https://research.tue.nl/files/201347816/0906980\_Prins\_J.H.M.\_MSc\_thesis\_MAP.pdf](https://research.tue.nl/files/201347816/0906980_Prins_J.H.M._MSc_thesis_MAP.pdf)  
[https://doi.org/10.1140/epje/s10189-023-00267-w](https://doi.org/10.1140/epje/s10189-023-00267-w)  
[https://arxiv.org/html/2412.08229v1](https://arxiv.org/html/2412.08229v1)  
[https://doi.org/10.1016/j.jcp.2022.111541](https://doi.org/10.1016/j.jcp.2022.111541)  
[https://www.sciencedirect.com/science/article/pii/S004578251930622X](https://www.sciencedirect.com/science/article/pii/S004578251930622X)  
[https://www.sciencedirect.com/science/article/abs/pii/S0045793020301985](https://www.sciencedirect.com/science/article/abs/pii/S0045793020301985)

* Toschi and Reith groups have done some work on this  
* also people at Sandia  
* \-\> check whether these are single component or multicomponent \- multicomponent might be novel

*Flow Matching*  
[https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)  
[https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html](https://mlg.eng.cam.ac.uk/blog/2024/01/20/flow-matching.html)  
[https://diffusionflow.github.io/](https://diffusionflow.github.io/) 