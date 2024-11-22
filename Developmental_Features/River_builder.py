#Libraries#####################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import polars as pl
import pandas as pd
from alive_progress import alive_bar
from alive_progress import config_handler
import time
#Functions#####################################################################
#colour palette
ocn_land_pal = mcolors.ListedColormap(["#71add9", "#7ab3df", "#85bae4",
                "#8ec2eb", "#97caf0", "#a2d3f7", "#addcfb", "#bae4ff",
                 "#BDE1F1", "#C7E6F5","#B3DAEC", "#c7edff","#d9f2fe", #water
                "#E3DAB3", "#add1a6", "#95c08c", "#a9c790",
                "#becd97", "#d2d8ad", "#e2e5b6", "#efecc1", "#e9e2b7",
                "#dfd7a4", "#d4cb9e", "#cbba83", "#c4a86b", "#ba995a",
                "#ab8852", "#ad9b7d", "#bbaf9b", "#cbc4b9", "#e1dfd9",
                "#f5f4f2"])
#bounds = [-4000, -3000, -2500, -2000, -1500, -1000, -750, -500, -250, -50,
          #0, 50, 75, 100, 200, 300, 450, 550, 650, 750, 850, 1000, 1100,
          #1500, 2000, 2250, 2500, 2750, 3000]
bounds = [-5000, -4000, -3000, -2000, -1000, -700, -500, -300, -200, -150, -50,
     0,
     100, 200, 300, 500, 700, 1000, 1500, 2000, 2500, 3000, 4000, 4500, 5000, 6000, 8000]
norm = mcolors.BoundaryNorm(bounds, ocn_land_pal.N)
#Progress bar
config_handler.set_global(length=40, bar = "fish")
#Perlin noise functions taken from iq.opengenus.org/perlin-noise###############
def gradient(c, x, y):
    vectors = np.array([[0,1], [0, -1], [1, 0], [-1, 0]])
    gradient_co = vectors[c % 4]
    return gradient_co[:, :, 0] * x + gradient_co[:, :, 1] * y

def fade(f):
    return 6 * f**5 - 15 * f**4 + 10 * f**3

def lin_interp(a, b, x):
    #linear interpolation (dot product)
    return a + x * (b - a)

def perlin_noise(x, y):
    #generates random values between 1-256
    rndm_val = np.arange(256, dtype = int)
    np.random.shuffle(rndm_val)  #shuffles the numbers
    #Creates a 2d array 
    rndm_table = np.stack([rndm_val, rndm_val]).flatten()
    #setting grid coordinate types
    xi, yi = x.astype(int), y.astype(int)
    #distance vector coordinates
    xg, yg = x - xi, y - yi
    #applying fade function to distance coordinates
    xf, yf = fade(xg), fade(yg)
    #gradient vectors for each corner
    n00 = gradient(rndm_table[rndm_table[xi] + yi], xg, yg)
    n01 = gradient(rndm_table[rndm_table[xi] + yi + 1], xg, yg - 1)
    n11 = gradient(rndm_table[rndm_table[xi + 1] + yi + 1], xg - 1, yg - 1)
    n10 = gradient(rndm_table[rndm_table[xi + 1] + yi], xg - 1, yg)
    #Applying linear interpolation, dot product to calculate average
    x1 = lin_interp(n00, n10, xf)
    x2 = lin_interp(n01, n11, xf)  
    return lin_interp(x1, x2, yf) 
#Base Functions################################################################
def base_mesh(start, end, x, y):
    #Create evenly spaced out numbers in a specified interval
    #This creates a df with 1° x 1° resolution
    x_array = np.linspace(1, end, 36, endpoint = False)
    y_array = np.linspace(1, end, 36, endpoint = False)
    # create grid using linear 1d arrays
    x, y = np.meshgrid(x_array, y_array)  
    return x, y

def rescale_z(matrix):
    matrix = matrix.to_numpy()
    #selecting values
    land_mask = (matrix >= 0)
    ocn_mask = (matrix <= 0)
    
    #Altering land surface to make it more "random"
    matrix[land_mask] = (matrix[land_mask] * 10)
    matrix[land_mask] = matrix[land_mask] + (np.sin(2 * matrix[land_mask])) + (np.sin(4 * matrix[land_mask]))+ (np.sin(8 * matrix[land_mask]) / 4)+ (np.sin(16 * matrix[land_mask])/32)
    
    #Altering ocean bed depth to make it more "random"
    matrix[ocn_mask] = (matrix[ocn_mask] * 10)
    matrix[ocn_mask] = matrix[ocn_mask] - ((np.sin(2 * matrix[ocn_mask])) + (np.sin(4 * matrix[ocn_mask]))+ (np.sin(8 * matrix[ocn_mask]) / 4)+ (np.sin(16 * matrix[ocn_mask])/32))
    return matrix

def create_dataframes(num_dfs):
    dataframes = []
    val = 2
    for i in range(num_dfs):
        x, y = base_mesh(1, val, 36, 36)
        val = (val * 2) + 5
        noise = perlin_noise(x, y)
        df = pl.DataFrame(noise)
        dataframes.append(df)
    return dataframes

def combine_dataframes(df_list):
    combined_df = df_list[0].clone()
    for df in df_list[1:]:
        combined_df += df
    return combined_df

def apply_spline(df):
    new_z = [[0] * 36 for _ in range(36)]
    values = pd.read_csv("Supporting_Data\\Height_Spline.csv")
    tolerance = 0.1 #appears to have control over how much land there is
    with alive_bar(36) as bar:
        for x in range(36):
            for y in range(36):
                z_value = df[x, y]
                for j in range(len(values)):
                    if abs(z_value - values["xfit"][j]) < tolerance:
                        new = values["yfit"][j]
                        new_z[x][y] = new
                        break
            bar()
    return new_z 

def smooth_2d(array, window_width):
    if window_width % 2 == 0:
        raise ValueError("Window width must be an odd number.")
    
    #Padding array so that it looks like it wraps around B)
    pad_width = window_width // 2
    padded_array = np.pad(array, pad_width, mode='wrap')

    # Create an output array of the same shape as the input
    smoothed_array = np.zeros_like(array)

    # Iterate over each element in the original array
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            # Extract the window
            window = padded_array[i:i + window_width, j:j + window_width]
            # Compute the mean of the window
            smoothed_array[i, j] = np.mean(window)
    
    return smoothed_array

def lake_filter(array) :
    output = np.copy(array)
    array = np.pad(array, 1, mode = "wrap")
    for x in range(36):
        for y in range(36):
            if (array[x,y] <= 1 and array[x + 1, y] == 1):
                output[x, y] = 2
            if (array[x,y] <= 1 and array[x- 1, y] == 1):
                output[x - 1, y] = 2
            if (array[x,y] <= 1 and array[x, y + 1] == 1):
                output[x, y] = 2
            if (array[x,y] <= 1 and array[x, y - 1] == 1):
                output[x, y - 1] = 2
    for x in range(36):
        for y in range(36):
            if output[x,y] == 1:
                output[x,y] = -1
    output = np.where(output >= 0, 1, -1)
    #Covers most edge cases, with the exception of a few which are weird...
    return output

#
def world_gen(octaves, seed):
    np.random.seed(seed)
    
    print("Creating your Noisy World!")
    with alive_bar(1) as bar:
        base_map = create_dataframes(octaves)
        base_map = combine_dataframes(base_map)

        height_mod = create_dataframes((octaves - 2))
        height_mod = combine_dataframes(height_mod)

        erosion_factor = create_dataframes((octaves))
        erosion_factor = combine_dataframes(erosion_factor)

        Continentalness = create_dataframes(octaves)
        Continentalness = combine_dataframes(Continentalness)
        bar()
    
    print("Reshaping your Noisy World!")
    with alive_bar(1) as bar:
        base_map = rescale_z(base_map)
        bar()
    
    print("Making your world taller!")
    final_world_map = apply_spline(base_map)
    final_world_map = (final_world_map) + (height_mod.to_numpy() * 500)
    
    print("Smoothing your world!")
    with alive_bar(1) as bar:
        final_world_map = smooth_2d(final_world_map, 11)
        bar()
    
    print("Eroding your world!")
    with alive_bar(1) as bar:
    #will need their own splines due to different ranges
        final_world_map = final_world_map + ((Continentalness.to_numpy() ** 2))
        final_world_map = final_world_map - ((erosion_factor.to_numpy() ** 2))
        #final_world_map = final_world_map + 35
        bar()
    print("Removing small lakes!")
    with alive_bar(1) as bar:
        binary_world = np.where(final_world_map > 0, 0, 1)
        b_world = lake_filter(binary_world)
        final_world_map = final_world_map * b_world
        bar()
    return final_world_map

#layers = int(input("How many Octaves? "))
seed = int(input("Set the seed "))
world = world_gen(octaves = 4, seed = seed)
#world = world_gen(octaves = 4, seed = 2004)
lat_ranges = np.arange(start = -90, stop = 90, step = 1)
lon_ranges = np.arange(-36, 36, 1)

x_coords = []
y_coords = []
z_values = []
#Moving df data to lists 
for x in range(36):
    for y in range(36):
        x_coords.append(lon_ranges[y])
        y_coords.append(lat_ranges[x])
        z_values.append(world[x][y])
#Creating new df
df_xyz = pl.DataFrame({
    "X": x_coords,
    "Y": y_coords,
    "Z": z_values
})

x_coords = np.array(df_xyz["X"])
y_coords = np.array(df_xyz["Y"])
z_values = np.array(df_xyz["Z"])
# Reshape X, Y, and Z into 2D arrays
x_grid = x_coords.reshape(36, 36)
y_grid = y_coords.reshape(36, 36)
z_grid = z_values.reshape(36, 36)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
scatter = plt.scatter(df_xyz["X"], df_xyz["Y"], c = df_xyz["Z"], cmap= ocn_land_pal, s = 250, marker = "s", norm = norm)
plt.colorbar(scatter, label = "Height")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")

def river_builder(array, lon, lat):
    
    output = np.copy(array)

    for x in range(lon):
        for y in range(lat):
            if array[x, y] <= 0:
                #defining ranges for array
                x_min = max(0, x-1)
                x_max = min(array.shape[0], x+2)
                y_min = max(0, y-1)
                y_max = min(array.shape[1], y + 2)
                
                #extracting values in area
                area = array[x_min : x_max, y_min : y_max].astype(float)
                area_temp = np.copy(area)
                area_temp[area_temp <= 0] = np.nan #sets all water bodies to nan so they're avoided
                
                #handling values
                ocean_check = np.all(np.isnan(area_temp))
                if ocean_check:
                    break #skips if in the middle of a lake/ocean
                else:
                    #lowest value in area
                    min_idx = np.unravel_index(np.nanargmin(area_temp), area_temp.shape) #finds index of lowest value in area
                    area[min_idx] = -1 #replaces lowest value in area with -1 
                    output[x_min : x_max, y_min : y_max] = area #applies change to world
    return output
                
                
test = river_builder(world, 36, 36)

x_coords = []
y_coords = []
z_values = []
#Moving df data to lists 
for x in range(36):
    for y in range(36):
        x_coords.append(lon_ranges[y])
        y_coords.append(lat_ranges[x])
        z_values.append(test[x][y])
#Creating new df
df_xyz_1 = pl.DataFrame({
    "X": x_coords,
    "Y": y_coords,
    "Z": z_values
})

plt.subplot(1, 2, 2)
scatter = plt.scatter(df_xyz_1["X"], df_xyz_1["Y"], c = df_xyz_1["Z"], cmap= ocn_land_pal, s = 250, marker = "s", norm = norm)
plt.colorbar(scatter, label = "Height")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.show()