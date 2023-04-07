# Granular-box-filling
Simulation of the random filling of a box with grains of variable size

The list of functions and their purpose:

    ShowBox: function to show the box filled with grains.
  
    RepartGrains: Function to calculate (and plot if asked) the distribution of the grains put into the box.
  
    ShowTrying: Function to show the evolution of the sucess/try ratio.
  
    GranularFilling: Function to randomly fill a blank space with granular particules.
  
    DictioRangeRay: Function to create a dictionnarie of the relative positions of the bordure point of a dissk of ray n.
  
    CompacGranular: A more compact method to fill a box with circular particules of random rays.
  

a1, b1, c1 = GranularFilling(1000, [50, 400], 0.005, 'uniform', verbose=True)

a2, b2, c2 = GranularFilling(1000, [50, 400], 0.005, 'lrqc', verbose=True)

a3, b3 = CompacGranular(1000, [50, 400], verbose=True)

ShowBox(a1, b1)
![Exemple picture](randFill_s1000_rr_50_400_rp_0,005_uniform.png)

RepartGrains(b1)
![Exemple picture](GrainsDistri_s1000_rr_50_400_rp_0,005_uniform.png)

ShowTrying(c1)
![Exemple picture](Trying_s1000_rr_50_400_rp_0,005_uniform.png)


ShowBox(a2, b2)
![Exemple picture](randFill_s1000_rr_50_400_rp_0,005_lrqc.png)

RepartGrains(b2)
![Exemple picture](GrainsDistri_s1000_rr_50_400_rp_0,005_lrqc.png)

ShowTrying(c2)
![Exemple picture](Trying_s1000_rr_50_400_rp_0,005_lrqc.png)



ShowBox(a3, b3)
![Exemple picture](randFill_s1000_rr_50_400_rp_0,005_compact.png)

RepartGrains(b3)
![Exemple picture](GrainsDistri_s1000_rr_50_400_compact.png)
