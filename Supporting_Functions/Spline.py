import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import interpolate

y = [-8000, -6000, -5000, -4000, -3000, -2500, -2000, -1500, -1000, -850, -800, -700, -600, -550, -500, -300, -200, -50,
     0,
     10, 300, 350, 400, 500, 700, 1000, 1500, 2000, 3000, 4000, 4500, 5000, 6000, 7000, 8000, 9000]
n = 12 / len(y)
x = np.arange(-6, 6, n)
# python
tck = interpolate.splrep(x, y, s=0)
xfit = np.arange(-6, 6, 0.5)
yfit = interpolate.splev(xfit, tck, der=0)

plt.plot(x, y, "ro")
plt.plot(xfit, yfit, "b")
plt.plot(xfit, yfit)
plt.title("Spline interpolation In Python")
plt.show()

df = pd.DataFrame({
    "xfit" : xfit, 
    "yfit" : yfit})

df.to_csv("Supporting_Data\\Height_Spline.csv", index = False)