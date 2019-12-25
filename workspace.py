import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ImageFilter
Image.MAX_IMAGE_PIXELS = None
#%%
def pixelsToCoordinates(pixelIndices, img, downsampleRate = 1):
    coordinates = np.zeros( pixelIndices.shape )
    coordinates[:,1] = img.size[1]  - pixelIndices[:,0]
    coordinates[:,0] = pixelIndices[:,1]
    coordinates = coordinates[::downsampleRate,:]

#%%
mapDirectoryRoot = '/home/dpm314/il2_map_analysis/maps/'
mapNames = ['moscow.png','stalingrad.png','kuban.png','rheinland.png']
mapFileNames = [mapDirectoryRoot + mapName for mapName in mapNames]
mapIndex = -1
imgRaw = Image.open(mapFileNames[mapIndex])
img = imgRaw.crop((0,0,5000,5000))
#resh = img.resize((10000,10000), resample = PIL.Image.NEAREST)
plt.imshow(img)
#Map raw .png files can be downloaded from: http://tiles.il2missionplanner.com/stalingrad/stalingrad.png #etc for the four map names
#mapDirectoryRoot = '/home/dpm314/coconut/maps/'

colorMap = {'forest':np.array( [189,186,162], dtype = np.int16),
            'water' :np.array( [161,186,192], dtype = np.int16),
            'city'  :np.array( [165,165,165], dtype = np.int16)
            }
thresholds = {'forest'  :np.int16(25),
              'water'   :np.int16(30),
              'city'    :np.int16(10)
            }
medianFilters = {'forest'  :ImageFilter.MedianFilter(size = 11),
                'water'   :ImageFilter.MedianFilter(size = 3),
                'city'    :ImageFilter.MedianFilter(size = 5)
            }


#%%
plt.close('all')
masks = {}
for key in colorMap.keys():
    print('Processing Masks for {} : {}'.format(mapNames[mapIndex], key))
    mask = np.int16(np.sum( np.abs( np.asarray(img) - colorMap[key]), axis = 2))

    mask = np.where(mask < thresholds[key], np.int16(0), mask )
    mask = Image.fromarray(mask)
    if medianFilters[key] is not None:
        mask = mask.filter(medianFilters[key])
    masks[key] = mask
    plt.figure()
    plt.imshow(masks[key])
#%%
locations = {}
coordinates = {}
for key in colorMap.keys():
    print('Locating for {} : {}'.format(mapNames[mapIndex], key))
    mask_as_array = np.array( masks[key].getdata(), dtype = np.int16 ).reshape( [img.size[0], img.size[1]] )
    locations[key] = np.argwhere( mask_as_array == 0 )
    coordinates[key] = pixelsToCoordinates( locations[key], masks[key], 1 )
#%%






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
key = 'forest'
#delta = np.zeros([I.shape[0],I.shape[1]])
#for i in range(I.shape[0]):
#    for j in range(I.shape[1]):
#        delta[i,j] = np.sum(np.abs(I[i,j,:] - colorMap[key]))
delta = np.sum(np.abs( I - colorMap[key]), axis = 2, dtype = np.int16)
plt.figure();plt.imshow(delta)


thresholds = {'clear' :np.int16(10), 
            'forest'  :np.int16(30), 
            'water'   :np.int16(10), 
            'city'    :np.int16(10)
            }

f = copy.copy(I)


mask = np.int16( np.where(delta <thresholds[key], 0,1))
#%%

#%%
#for k in range(3):
#    f[:,:,k] *= mask
#%%
medianFilter = PIL.ImageFilter.MedianFilter(size = 9)

filteredMask = PIL.Image.fromarray(mask).filter(medianFilter)
plt.figure()
plt.imshow(filteredMask)
for k in range(3):
    f[:,:,k] *= filteredMask
plt.figure()
plt.imshow(f)

#%%


plt.figure(); plt.imshow(mask)

f[np.where(delta<threshold)[0],np.where(delta<threshold)[1],0] = 0
f[np.where(delta<threshold)[0],np.where(delta<threshold)[1],1] = 0
f[np.where(delta<threshold)[0],np.where(delta<threshold)[1],2] = 0
plt.imshow(f)
#%%
deltaImage = PIL.Image.fromarray(delta)

#%% Apply median filter to remove small artifacts
from PIL import ImageFilter
medianFilter = PIL.ImageFilter.MedianFilter(size = 25)
deltaFiltered = deltaImage.filter(medianFilter)
plt.figure()
plt.imshow(deltaFiltered)
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

#%%
img = Image.open(mapFileNames[-1])
img = img.crop((8000,8000,10000,10000))


























