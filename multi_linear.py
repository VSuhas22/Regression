import numpy as np
from numpy import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
points=genfromtxt('data.csv',delimiter=",")
y_original=[]
x0_original=[]
x1_original=[]
x2_original=[]

for i in range(1,len(points)):
    y_original.append(points[i,0])
    x1_original.append(points[i,1])
    x2_original.append(points[i,2])
    x0_original.append(1)
Y=np.array(y_original)
X1=np.array([x0_original, x1_original, x2_original])
X=np.transpose(X1)
B=np.matmul(np.matmul(np.linalg.inv(np.matmul(X1,X)),X1),Y)
E= Y-np.matmul(X,B)
error=0
for i in range(0, len(points)-1):
    error+=E[i]**2
print("SE value for the data is: ",(float)(error/(len(points)-1)))
y_predict=[]
for i in range(0,len(points)-1):
    y_predict.append(B[0]+x1_original[i]*B[1]+x2_original[i]*B[2])



from mpl_toolkits.mplot3d import Axes3D
xx1, xx2 = np.meshgrid(np.linspace(min(x1_original),max(x1_original), 100), np.linspace(min(x2_original), max(x2_original), 100))
Z = B[0] + B[1] * xx1 + B[2] * xx2

fig = plt.figure(figsize=(12, 8))
ax = Axes3D(fig, azim=-115, elev=15)

surf = ax.plot_surface(xx1, xx2, Z, cmap=plt.cm.RdBu_r, alpha=0.6, linewidth=0)
ax.scatter(x1_original, x2_original, y_original, color='blue', alpha=1.0, facecolor='red')

plt.show()
