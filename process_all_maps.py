#Map image .png files can be downloaded from: http://tiles.il2missionplanner.com/stalingrad/stalingrad.png #etc for the four map names
#mapDirectoryRoot = '/home/dpm314/coconut/maps/' #mapDirectoryRoot = '/home/dpm314/il2_map_analysis/maps/'
import numpy as np
import matplotlib.pyplot as plt
import PIL
import os
from PIL import Image
from PIL import ImageFilter
Image.MAX_IMAGE_PIXELS = None
#%%################# Configuration ###################################
directoryRoot = '/home/dpm314/coconut/'
mapNames = ['moscow','stalingrad','kuban','rheinland']

colorMap = {'forest':np.array( [189,186,162], dtype = np.int16),
            'water' :np.array( [161,186,192], dtype = np.int16),
            'city'  :np.array( [165,165,165], dtype = np.int16)
            }
thresholds = {'forest'  :np.int16(25),
              'water'   :np.int16(20),
              'city'    :np.int16(1)
            }

mapFilters = {'forest'  :ImageFilter.ModeFilter( size = 5 ),#ImageFilter.ModeFilter( size = 3 ),
              'water'   :None,
              'city'    :ImageFilter.ModeFilter( size = 3 )
            }
maskFilters = {'forest' :None,
              'water'   :ImageFilter.MinFilter( size = 5 ),
              'city'    :None
            }
mapDimensions = {
    'moscow'    : (28*10*1000.0,28*10*1000.0),
    'stalingrad': (36*10*1000.0,23*10*1000.0),
    'kuban'     : (40*10*1000.0,32*10*1000.0),
    'rheinland' : (40*10*1000.0,40*10*1000.0) #40 squares, 10 km/square, 1000 meters/km
    }
#%%################# Configuration ###################################
directoryRoot = '/home/dpm314/coconut/'
mapNames = ['moscow','stalingrad','kuban','rheinland']

colorMap = {'forest':np.array( [173,174,162], dtype = np.int16),
            'water' :np.array( [161,186,192], dtype = np.int16),
            'city'  :np.array( [165,165,165], dtype = np.int16)
            }
thresholds = {'forest'  :np.int16(35),
              'water'   :np.int16(22),
              'city'    :np.int16(5)
            }

mapFilters = {'forest'  :ImageFilter.ModeFilter( size = 9 ),#ImageFilter.ModeFilter( size = 3 ),
              'water'   :None,
              'city'    :ImageFilter.ModeFilter( size = 11 )
            }
maskFilters = {'forest' :None,
              'water'   :ImageFilter.MinFilter( size = 5 ),
              'city'    :ImageFilter.MinFilter( size = 5)
            }
mapDimensions = {
    'moscow'    : (28*10*1000.0,28*10*1000.0),
    'stalingrad': (36*10*1000.0,23*10*1000.0),
    'kuban'     : (40*10*1000.0,32*10*1000.0),
    'rheinland' : (40*10*1000.0,40*10*1000.0) #40 squares, 10 km/square, 1000 meters/km
    }
#%%################# Utility Functions ###################################
def fixMapCoordinates(pixelIndices, imgSize):
    #converts screen coordinates to lat/long coordinate orientation
    # where X goes east to west, Y goes south to North
    coordinates = np.zeros( pixelIndices.shape )
    coordinates[:,1] = imgSize[1]  - pixelIndices[:,0]
    coordinates[:,0] = pixelIndices[:,1]
    return coordinates

def writeCoordinatesToFile(coords,filename_base = 'coordinates_', path_base = directoryRoot):
    for key in coords.keys():
        fname = path_base + 'data/' + '{}{}.csv'.format(filename_base, key)
        print("Writing Coordinates to file: {}".format(fname))
        np.savetxt(fname,np.transpose(coordinates[key]),delimiter=',',fmt='%1.9f')

'''
#not using this function anymore, normalize to 0.0 to 1.0 on lattitude and longitude coords
#   instead of using meters from southwest corner of the map
def pixelsToMeters(pixel, mapDimension, imgSize ):
    #normalize first then expand to meters

    #x = np.float32( pixel[:,0] ) / np.max( pixel[:,0] )
    #y = np.float32( pixel[:,1] ) / np.max( pixel[:,1] )
    x = np.float32( pixel[:,0] ) / img.size[0]
    y = np.float32( pixel[:,1] ) / img.size[1]

    x *= np.float32( mapDimension[0] )
    y *= np.float32( mapDimension[1] )
    return x,y
'''
def pixelsToNormalized(pixel, imgSize ):
    #normalize first then expand to meters
    x = np.float32( pixel[:,0] ) / np.float32( imgSize[0] )
    y = np.float32( pixel[:,1] ) / np.float32( imgSize[1] )
    return x,y
#%%################ Processing Code #########################################
if __name__ == '__main__':
    mapFileNames = [directoryRoot + 'maps/' + mapName + '.png' for mapName in mapNames]
    for mapIndex in range(len(mapFileNames)):
        print( '.... Processing Map: {}'.format(mapNames[mapIndex]))
        img = Image.open(mapFileNames[mapIndex])#.crop([2500,2500,4001,4001]) #for debug work on small subsection
        masks = {}
        diff = {}
        coordinates = {}
        for key in ['city','forest','water']: #loop in this order for hack fix to remove already-detected city from the forest detection.
            print('Generating masks for {} : {}'.format(mapNames[mapIndex], key))
            #Pre-process filter map image if needed
            if mapFilters[key] is not None:
                img_array = np.asarray(img.filter(mapFilters[key]))
            else:
                img_arrag = np.asarray(img)
            print('.')
            #Compute each pixels similarity to key (water, city or forest templates in colorMap)
            diff[key] = np.int16(np.sum( np.abs( img_array - colorMap[key]), axis = 2))
            #Create a mask from the difference, zero pixels below a threshold
            mask = np.where(diff[key] <= thresholds[key], np.int16(1), img.convert('I') ) #found
            mask = np.where(diff[key] > thresholds[key],  np.int16(0), mask ) #not_foundmake strictly binary
            #hack - multiply inverse of city mask to the forest map
            print('..')
            if(key == 'forest'):
                mask =  masks['city']  +  mask #will make the value '2' where it is both, in which case we don't want that detected. where mask == 1 willl fix this
            #Post-process filter mask if needed
            if maskFilters[key] is not None:
                #Convert back to Image and filter
                mask = Image.fromarray(mask).filter(maskFilters[key])
                #convert back to numpy array 
                mask = np.asarray(mask, dtype = np.int16)
            #optional plot the mask:
            #plt.figure() #plt.imshow(masks[key])
            #Find indices of all zeros in the mask (where key is)
            print('    Locating {}'.format(key))
            locations = np.argwhere( mask == 1 )
            #location is pixel coordinates
            locations = fixMapCoordinates( locations, img.size ) #fix screen coordinates to x/y lat/long coordinates
            #normalize from 0.0 to 1.0; increasing x is west to east, increasing y is south to north bound to 0.0..1.0
            coordinates[key] = pixelsToNormalized( locations, img.size )
            #store mask for debug & display
            masks[key] = mask
        #Write to .csv file
    dataFilePathBase = directoryRoot + 'data/' + mapNames[mapIndex] + '/'
    writeCoordinatesToFile(coordinates, path_base=dataFilePathBase)