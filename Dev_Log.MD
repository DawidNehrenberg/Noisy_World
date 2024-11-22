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

###### 22/11/2024
<p>
Working on a river pathfinding algorithm to connect bodies of water.
Currently the algorithm, found in Developmental_Features/River_builder.py, essenitally works to expand the borders of bodies of water.
The end goal of this algorithm will be to connect all bodies of water to the ocean; the current approach to this problem is to find the cell with the lowest value in a 3x3 grid and set the height value to -1.
In order to not change the pre-exisitng water cells, their values are set to nan so that they are not included in the minimum calculation. The index of the lowest lying land cell next to a water cell is then recorded and passed onto the original area grid which is then placed back into the output array.
</p>

<p>
The next step in the development of river pathfinding algorithm would be to make it recrusive, so that it can extend rivers.
This could be achieved by doing an intial pass to "seed" the rivers, and then repeatedly carrying out the function until all rivers reach a point where the entire grid array is water cells.
Another improvement that could be made to the algorithm is passing it through the outline of the lakes.
An earlier iteration of the feature filter was capable of this, so reverting to that version and incorporating it into the river pathfinding algorithm could be used to ensure that there's only two main rivers; one to feed the lake and one to empty the lake.
This could be done by turning the lake outline into a list of indexes and selecting a random value that is seperated by an x amount of cells.
</p>