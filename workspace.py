import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
#%%
def pixelValueHistogram(I):
    #returns pixelCount[i,j]
    #where   pixelCount[i,j] is number of times the value i occurs in layer j
    pixelCount = np.zeros((255,3))
    for i in range(3):
        for j in range(255):
            print(i,j)
            pixelCount[j,i] = np.sum(np.where(I[:,:,i] == j))
    return pixelCount
#%%

#%%
#Map raw .png files can be downloaded from: http://tiles.il2missionplanner.com/stalingrad/stalingrad.png #etc for the four map names
mapDirectoryRoot = '/home/dpm314/coconut/maps/'
mapNames = ['moscow.png','stalingrad.png','kuban.png','rheinland.png']
mapFileNames = [mapDirectoryRoot + mapName for mapName in mapNames]
#%%
img = Image.open(mapFileNames[-1])
#plt.plot( img.histogram() )
#resh = img.resize((10000,10000), resample = PIL.Image.NEAREST)
img = img.crop((8000,8000,10000,10000))
plt.imshow(img)
#%%
r,g,b = img.split()

#%%
nr,ng,nb = np.array(r.getdata()), np.array(g.getdata()), np.array(b.getdata())
#%%
resh = img.resize((10000,10000), resample = PIL.Image.NEAREST)
#%%
plt.figure()
plt.plot( resh.histogram()); plt.yscale('log');
plt.figure()
plt.plot( img.histogram()); plt.yscale('log');

#%%
nparr = np.array( resh.getdata(), dtype = np.int16)
nparr = nparr.reshape(resh.size[1], resh.size[0],3)

#note: I[i,:] is east  -> west at latitude i
#      I[:,j] is north -> south at longitude j 
#%%

pixelInds = np.argsort( resh.histogram() )[::-1] #list in descending order of the most common 256 bit values 
#%%
#try to figure out what is what...
plt.figure(); 
plt.imshow(nparr);
#%%

k = np.where( (nparr == pixelInds[0]) | (nparr == 50) | (nparr == 124) | (nparr == 126),0,255) #this works without indexing into the color values because len(pixelInds) == len(resh.histogram) == 256


#%%
#useful stuff in Image.
# .composite() blends two images with a transparency mask
# .merge() combines bands (R,G,B) in to a multiband image
# .fromarray() make image from numpy array 
# .filter(filter)
# np_array_version = np.asarray(im)
#%%
import imageio
import copy
#%%
II = imageio.imread(mapFileNames[-1])
#%%
plt.close('all')
I = np.array(II[1000:3000,2800:5500,:], dtype = np.int16)
colorMap = {'clear' :np.array( [211,209,197], dtype = np.int16), 
            'forest':np.array( [189,186,162], dtype = np.int16),
            'water' :np.array( [161,186,197], dtype = np.int16),
            'city'  :np.array( [165,165,165], dtype = np.int16)
            }


#delta = np.abs((I - colorMap['water'])).sum(axis = 2)
key = 'city'
delta = np.zeros([I.shape[0],I.shape[1]])
for i in range(I.shape[0]):
    for j in range(I.shape[1]):
        delta[i,j] = np.sum(np.abs(I[i,j,:] - colorMap[key]))
plt.figure();plt.imshow(delta)




f = copy.copy(I)
threshold= 15

mask = np.where(delta <threshold, 0,1)

plt.figure(); plt.imshow(mask)

f[np.where(delta<threshold)[0],np.where(delta<threshold)[1],0] = 0
f[np.where(delta<threshold)[0],np.where(delta<threshold)[1],1] = 0
f[np.where(delta<threshold)[0],np.where(delta<threshold)[1],2] = 0
plt.imshow(f)
#%%














