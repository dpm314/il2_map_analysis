#Map image .png files can be downloaded from: http://tiles.il2missionplanner.com/stalingrad/stalingrad.png #etc for the four map names
#mapDirectoryRoot = '/home/dpm314/coconut/maps/'

import numpy as np
import matplotlib.pyplot as plt
import PIL
from PIL import Image
from PIL import ImageFilter
Image.MAX_IMAGE_PIXELS = None
#%%###########################################################################
mapDirectoryRoot = '/home/dpm314/il2_map_analysis/maps/'
mapNames = ['moscow','stalingrad','kuban','rheinland']

colorMap = {'forest':np.array( [189,186,162], dtype = np.int16),
            'water' :np.array( [161,186,192], dtype = np.int16),
            'city'  :np.array( [165,165,165], dtype = np.int16)
            }
thresholds = {'forest'  :np.int16(25),
              'water'   :np.int16(30),
              'city'    :np.int16(10)
            }
medianFilters = {'forest' :ImageFilter.MedianFilter(size = 11),
                'water'   :ImageFilter.MedianFilter(size = 3 ),
                'city'    :ImageFilter.MedianFilter(size = 5 )
            }

medianFilters = {'forest'  :None,
                'water'   : None,
                'city'    : None
            }

mapDimensions = {
    'moscow'    : None, #todo: measure the other maps
    'stalingrad': None,
    'kuban'     : None,
    'rheinland' : (40*10*1000.0,40*10*1000.0) #40 squares, 10 km/square, 1000 meters/km
    }

def fixMapCoordinates(pixelIndices, img, downsampleRate = 1):
    coordinates = np.zeros( pixelIndices.shape )
    coordinates[:,1] = img.size[1]  - pixelIndices[:,0]
    coordinates[:,0] = pixelIndices[:,1]
    coordinates = coordinates[::downsampleRate,:]
    return coordinates

def writeCoordinatesToFile(coords,filename_base = 'coordinates_', path_base = '/home/dpm314/il2_map_analysis/data/'):
    for key in coords.keys():
        fname = path_base + '{}{}.csv'.format(filename_base, key)
        print("Writing Coordinates to file: {}".format(fname))
        np.savetxt(fname,coordinates[key],delimiter=',',fmt='%7i',header='test header')

def pixelsToMeters(pixel, mapDimension ):
    #normalize first then expand to meters
    x = np.float32( pixel[:,0] ) / np.max( pixel[:,0] )
    y = np.float32( pixel[:,1] ) / np.max( pixel[:,1] )
    x *= mapDimension[0]
    y *= mapDimension[1]
    x = np.int32(x) #round to nearest meter and save as ints
    y = np.int32(y) 
    return x,y

#%%############################################################################
mapFileNames = [mapDirectoryRoot + mapName + '.png' for mapName in mapNames]
mapIndex = -1
img = Image.open(mapFileNames[mapIndex])

img = img.resize( (img.size[0],
                   img.size[1],) )
plt.imshow(img)


#%%
plt.close('all')
masks = {}
img_as_array = np.asarray(img) 
for key in colorMap.keys():
    print('Generating masks for {} : {}'.format(mapNames[mapIndex], key))
    mask = np.int16(np.sum( np.abs( img_as_array - colorMap[key]), axis = 2))
    mask = np.where(mask < thresholds[key], np.int16(0), mask )
    mask = Image.fromarray(mask)
    if medianFilters[key] is not None:
        mask = mask.filter(medianFilters[key])
    masks[key] = mask
    #plt.figure()
    #plt.imshow(masks[key])
#%%
locations = {}
coordinates = {}
for key in colorMap.keys():
    print('Locating for {} : {}'.format(mapNames[mapIndex], key))
    #mask_as_array = np.array( masks[key].getdata(), dtype = np.int16 ).reshape( [masks[key].size[0], masks[key].size[1]] )
    mask_as_array  = np.asarray(masks[key], dtype = np.int16)
    locations[key] = np.argwhere( mask_as_array == 0 )
    locations[key] = fixMapCoordinates( locations[key], masks[key], 1 )
    coordinates[key] = pixelsToMeters( locations[key], mapDimensions[mapNames[mapIndex] ] )
    
#%%
for key in coordinates.keys():
    writeCoordinatesToFile(coordinates[key])

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


























