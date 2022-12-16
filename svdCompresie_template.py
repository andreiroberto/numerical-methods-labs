# Nume student: Stoica Andrei-Roberto
####################################
# DVS: Compresia de imagini alb negru
####################################
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])
Img =  mpimg.imread('imagine.png') 
fig1 =plt.figure(1)
plt.imshow(Img)

fig2 =plt.figure(2)
grayImg = rgb2gray(Img)
plt.imshow(grayImg, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)

plt.show()
U,sigma,V=np.linalg.svd(grayImg)
plt.semilogy(sigma) 
print(sigma>=1)
plt.show()
plt.title("Graficul procent informatie vs valori")
plt.ylabel('Informatie')
plt.xlabel('Valori')
plt.plot(np.cumsum(sigma) / np.sum(sigma))
plt.show()
rank= [800, 600, 300, 200, 100, 50,30,10]
j = 1
for i in rank:
    img= plt.figure(j)
    aprox_img = U[:,:i] @ np.diag(sigma[:i]) @ V[:i,:]
    plt.title(f'Rangul{i}')
    plt.imshow(aprox_img, cmap=plt.get_cmap('gray'), vmin=0, vmax=1)
    j=j+1
plt.show()