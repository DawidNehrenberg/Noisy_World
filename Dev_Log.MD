# Dev log for Noisy World

###### 20/11/2024 - v0.0.1

<p> Adding spline functions to alter the height of the base map to enable a larger range of heights.</p> 
<p> 
Height values are transformed by corresponding grid cell values along the x,y axis to the x value of the spline value array and prescribing the y value to the base map to create the new height map.
The new height values are stored in the Supporting_Data folder and the script expects a file named "Height_Spline.csv" to carry out the height transformation. This file can either be created manually or by using "Spline.py", found in Supporting_Functions, to generate the values for the height transformation.
</p>
