# Granular-box-filling
Simulation of the random filling of a box with grains of variable size

The list of functions and their purpose:

    show_box: function to show the box filled with grains.
  
    repart_grains: Function to calculate (and plot if asked) the distribution of the grains put into the box.
  
    show_trying: Function to show the evolution of the sucess/try ratio.
  
    granular_filling: Function to randomly fill a blank space with granular particules.
  
    dictio_range_ray: Function to create a dictionnarie of the relative positions of the bordure point of a dissk of ray n.
  
    compac_granular: A more compact method to fill a box with circular particules of random rays.
  

a1, b1, c1 = granular_filling(1000, [50, 400], 0.005, 'uniform', verbose=True)

a2, b2, c2 = granular_filling(1000, [50, 400], 0.005, 'lrqc', verbose=True)

a3, b3 = compac_granular(1000, [50, 400], verbose=True)

show_box(a1, b1)

![Exemple picture](img/randFill_s1000_rr_50_400_rp_0,005_uniform.png)

repart_grains(b1)

![Exemple picture](img/GrainsDistri_s1000_rr_50_400_rp_0,005_uniform.png)

show_trying(c1)

![Exemple picture](img/Trying_s1000_rr_50_400_rp_0,005_uniform.png)


show_box(a2, b2)

![Exemple picture](img/randFill_s1000_rr_50_400_rp_0,005_lrqc.png)

repart_grains(b2)

![Exemple picture](img/GrainsDistri_s1000_rr_50_400_rp_0,005_lrqc.png)

show_trying(c2)

![Exemple picture](img/Trying_s1000_rr_50_400_rp_0,005_lrqc.png)



show_box(a3, b3)

![Exemple picture](img/randFill_s1000_rr_50_400_rp_0,005_compact.png)

repart_grains(b3)

![Exemple picture](img/GrainsDistri_s1000_rr_50_400_compact.png)
