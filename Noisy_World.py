#Libraries#####################################################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import polars as pl

#Functions#####################################################################
#colour palette
ocn_land_pal = mcolors.ListedColormap(["#71add9", "#7ab3df", "#85bae4",
                "#8ec2eb", "#97caf0", "#a2d3f7", "#addcfb", "#bae4ff",
                "#c7edff", "#d9f2fe", #water
                "#add1a6", "#95c08c", "#a9c790",
                "#becd97", "#d2d8ad", "#e2e5b6", "#efecc1", "#e9e2b7",
                "#dfd7a4", "#d4cb9e", "#cbba83", "#c4a86b", "#ba995a",
                "#ab8852", "#ad9b7d", "#bbaf9b", "#cbc4b9", "#e1dfd9",
                "#f5f4f2"])
bounds = [-4000, -3000, -2500, -2000, -1500, -1000, -750, -500, -250, -50,
          0, 50, 75, 100, 200, 300, 450, 550, 650, 750, 850, 1000, 1100,
          1500, 2000, 2250, 2500, 2750, 3000]
norm = mcolors.BoundaryNorm(bounds, ocn_land_pal.N)
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
    x_array = np.linspace(1, end, 360, endpoint = False)
    y_array = np.linspace(1, end, 180, endpoint = False)
    # create grid using linear 1d arrays
    x, y = np.meshgrid(x_array, y_array)  
    return x, y

def rescale_z(matrix):
    matrix = matrix.to_numpy()
    #selecting values
    land_mask = (matrix >= 0)
    ocn_mask = (matrix <= 0)
    #Altering land surface to make it more "random"
    matrix[land_mask] = (matrix[land_mask] * 100)
    #matrix[land_mask] = matrix[land_mask] + (np.sin(2 * matrix[land_mask]) * 10) + (np.sin(4 * matrix[land_mask]) * 5)
    matrix[land_mask] = matrix[land_mask] + (np.sin(2 * matrix[land_mask])) + (np.sin(4 * matrix[land_mask]))+ (np.sin(8 * matrix[land_mask]) / 4)+ (np.sin(16 * matrix[land_mask])/32)
    #Altering ocean bed depth to make it more "random"
    matrix[ocn_mask] = (matrix[ocn_mask] * 100)
    matrix[ocn_mask] = matrix[ocn_mask] - ((np.sin(2 * matrix[ocn_mask])) + (np.sin(4 * matrix[ocn_mask]))+ (np.sin(8 * matrix[ocn_mask]) / 4)+ (np.sin(16 * matrix[ocn_mask])/32))
    return matrix

def create_dataframes(num_dfs):
    dataframes = []
    val = 2
    for i in range(num_dfs):
        x, y = base_mesh(1, val, 360, 180)
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
#
def world_gen(octaves, seed):
    np.random.seed(seed)
    base_map = create_dataframes(octaves)
    base_map = combine_dataframes(base_map)
    
    height_mod = create_dataframes((octaves - 2))
    height_mod = combine_dataframes(height_mod)
    
    erosion_factor = create_dataframes((octaves))
    erosion_factor = combine_dataframes(erosion_factor)
    
    Continentalness = create_dataframes(octaves)
    Continentalness = combine_dataframes(Continentalness)
    
    base_map = rescale_z(base_map)
    final_world_map = base_map + ((height_mod.to_numpy()) * 1000)
    final_world_map = final_world_map + ((Continentalness.to_numpy() ** 2) * 100)
    #final_world_map = final_world_map - ((erosion_factor.to_numpy() ** 2) * 500)
    return final_world_map

layers = int(input("How many Octaves? "))
seed = int(input("Set the seed "))
world = world_gen(octaves = layers, seed = seed)

lat_ranges = np.arange(start = -90, stop = 90, step = 1)
lon_ranges = np.arange(-180, 180, 1)

x_coords = []
y_coords = []
z_values = []
#Moving df data to lists 
for x in range(180):
    for y in range(360):
        x_coords.append(lon_ranges[y])
        y_coords.append(lat_ranges[x])
        z_values.append(world[x, y])
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
x_grid = x_coords.reshape(180, 360)
y_grid = y_coords.reshape(180, 360)
z_grid = z_values.reshape(180, 360)

#plt.figure(figsize=(10, 8))
scatter = plt.scatter(df_xyz["X"], df_xyz["Y"], c = df_xyz["Z"], cmap= ocn_land_pal, s = 13, marker = "s", norm = norm)
plt.colorbar(scatter, label = "Height")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.show()
