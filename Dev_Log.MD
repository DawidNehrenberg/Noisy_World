# Dev log for Noisy World

#### v0.0.1
###### 20/11/2024

<p> Adding spline functions to alter the height of the base map to enable a larger range of heights.</p> 

<p> 
Height values are transformed by corresponding grid cell values along the x,y axis to the x value of the spline value array and prescribing the y value to the base map to create the new height map.
The new height values are stored in the Supporting_Data folder and the script expects a file named "Height_Spline.csv" to carry out the height transformation. This file can either be created manually or by using "Spline.py", found in Supporting_Functions, to generate the values for the height transformation.
</p>

<p>
I'm currently working on a feature filter function which aims to remove small lakes. 
This feature is almost functional, with it being able to remove the smallest lakes that are 1 cell large.
However, the process by which this is carried out also removes some ocean cells from the coast, but this isn't a prevalent bug so it can act as a feature to add some randomness to the coastline.
A fully functional transformation from identifying 1 cell lakes to filtering them out still has not been implemented, but this should be a simple implementation done via multiplying the 1 cell lake values by -1 to change the negative depth to a positive height.
No height smoothing will need to be carried out as the lakes generate in areas of low heights and the depth of the lake is roughly -50m which should smoothly blend into the terrain.
</p>