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
#%%################# Utility Functions ###################################
def fixMapCoordinates(pixelIndices, imgSize):
    coordinates = np.zeros( pixelIndices.shape )
    coordinates[:,1] = imgSize[1]  - pixelIndices[:,0]
    coordinates[:,0] = pixelIndices[:,1]
    return coordinates
def writeCoordinatesToFile(coords,filename_base = 'coordinates_', path_base = directoryRoot):
    for key in coords.keys():
        if not path_base.endswith('/'):
            path_base += '/'
        fname = path_base + '{}{}.csv'.format(filename_base, key)
        os.makedirs(os.path.dirname(fname), exist_ok=True)
        print("Writing Coordinates to file: {}".format(fname))
        np.savetxt(fname,coordinates[key],delimiter=',',fmt='%7i')
def pixelsToMeters(pixel, mapDimension ):
    #normalize first then expand to meters
    x = np.float32( pixel[:,0] ) / np.max( pixel[:,0] )
    y = np.float32( pixel[:,1] ) / np.max( pixel[:,1] )
    x *= mapDimension[0]
    y *= mapDimension[1]
    x = np.int32(x) #round to nearest meter and save as ints
    y = np.int32(y) 
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
        for key in colorMap.keys():
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
            mask = np.where(diff[key] < thresholds[key], np.int16(0), img.convert('I') )
            print('..')
            #Post-process filter mask if needed
            if maskFilters[key] is not None:
                #Convert back to Image and filter
                mask = Image.fromarray(mask).filter(maskFilters[key])
                #convert back to numpy array 
                mask = np.asarray(mask, dtype = np.int16)
            #Find indices of all zeros in the mask (where key is)
            print('    Locating {}'.format(key))
            locations = np.argwhere( mask == 0 )
            locations = fixMapCoordinates( locations, img.size)
            #Convert to meters from the South-West corner (increasing X goes East and increasing Y goes North)
            coordinates[key] = pixelsToMeters( locations, mapDimensions[mapNames[mapIndex] ] )
            #store mask for debug & display
            masks[key] = mask
        #Write to .csv file
        dataFilePathBase = directoryRoot + 'data/' + mapNames[mapIndex] + '/'
        writeCoordinatesToFile(coordinates, path_base=dataFilePathBase)