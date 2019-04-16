# PACKAGE IMPORTATION
import numpy as np
import pandas as pd
import os
import re
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt as dt
from scipy.ndimage import binary_fill_holes as bfh
from skimage import io
from skimage.measure import label,regionprops
from skimage.morphology import disk,binary_closing,remove_small_holes,remove_small_objects,watershed
from skimage.morphology._flood_fill import flood_fill
from skimage.segmentation import clear_border
from skimage.feature import peak_local_max
from skimage.color import rgb2gray
from skimage.filters import sobel,threshold_local
from tqdm import tqdm


# PROCESS PREPARATION
# Variables and functions definition
data = {}
circle_mask = np.pad(disk(165), ((20, 20),(20, 20)), 'constant', constant_values=0)

def normal_hist(array):
    '''Normalise the greyscale intensity of the image.'''
    normalized=(array - array.min())/(array.max()-array.min())
    return np.round(normalized*255).astype('int')


def region_max_area(labeled):
    '''Returns the region with the maximum area from the labeled image.'''
    regions=regionprops(labeled)
    regions_areas = [region.area for region in regions]
    region_index = regions_areas.index(max(regions_areas))
    region = regions[region_index]
    return (region_index,region)


def im_proc(filename):
    '''Main function of the image processing.'''
    # Image reading and greyscale conversion
    im = 1 - rgb2gray(io.imread(filename))

    # Image greyscale normalisation
    im_n = normal_hist(im)[:,256:1792]

    # Center detection
    im_filled = bfh(sobel(im_n >= 216))
    im_labeled = label(im_filled)
    center = region_max_area(im_labeled)[1].centroid
    r,c=np.round(center).astype('int')

    # Image cropping
    im_center = im_n[r - 185:r + 186, c - 185:c + 186]

    # Gaussian thresholding
    im_gauss = im_center >= threshold_local(im_center, 21, method='gaussian', param=100,offset=0)

    # Artifacts removal
    im_gauss=remove_small_objects(im_gauss, min_size=64)
    im_gauss=remove_small_holes(im_gauss,min_size=64)

    # Border gap fill
    for pos in [(0,0),(0,370),(370,0),(370,370)]:
        im_gauss=flood_fill(im_gauss.astype('int'),pos,1)

    # Isolate spheroid from borders
    distance = dt(im_gauss)
    local_maxi = peak_local_max(distance, indices=False, footprint=np.ones((3, 3)),labels=im_gauss)
    markers = label(local_maxi)
    labels = watershed(-distance, markers, mask=im_gauss)
    im_cleared = remove_small_objects(clear_border(labels,buffer_size=5,bgval=0)*circle_mask != 0,min_size=64)

    # Select largest area region as spheroid
    im_cleared_labeled = label(im_cleared)
    region_index,region=region_max_area(im_cleared_labeled)
    im_final = binary_closing(im_cleared_labeled == region_index+1,selem=disk(4))

    # Spheroid properties
    spheroid = regionprops(im_final.astype('int'))[0]
    spheroid_maj_ax = round(spheroid.major_axis_length,3)
    spheroid_min_ax = round(spheroid.minor_axis_length,3)
    spheroid_area = spheroid.area
    spheroid_peri = round(spheroid.perimeter,3)

    im_gray = 255 - im_center

    return im_final, im_gray, spheroid_maj_ax, spheroid_min_ax, spheroid_peri, spheroid_area


# USER PROMPT
# Working directory setting
path=input('Please enter your spheroid images folder path.\n')
os.chdir(path)

# Folders creation
for newfolder in ['bw','gray','data']:
    try:
        os.makedirs(newfolder)
    except FileExistsError:
        pass

# Filenames are assumed to be in this format: a1_d1.jpg or a1_bone_d1.jpg.
# a1 = Spheroid position in SpheroFilm
# d1 = Number of day
# bone = Indication of spheroids undergoing osteogenic differentiation
nameformat=input("Enter 1 if the filenames are in this format: a1_d1.jpg.\n"
                 "Enter 2 if the filenames are in this format: a1_bone_d1.jpg.\n")

# Image file listing
if nameformat in ['1','2']:
    nameformat = int(nameformat)
    if nameformat == 1:
        r = re.compile("[a-zA-Z0-9]+_d\d+\..*")
        files = list(filter(r.match, os.listdir(os.getcwd())))
    elif nameformat == 2:
        r = re.compile("[a-zA-Z0-9]+_bone_d\d+\..*")
        files = list(filter(r.match, os.listdir(os.getcwd())))
else:
    print('Please enter a valid response!')
    raise ValueError

# Process trigger
input('Please press ENTER to start the process.\n')


# IMAGE PROCESSING AND OBJECT MEASUREMENT
# Batch image processing
for filename in tqdm(files):
    # Image processing
    im_final, im_gray, spheroid_maj_ax, spheroid_min_ax, spheroid_peri, spheroid_area = im_proc(filename)

    # Image saving
    plt.imsave(f'bw\\{filename}', im_final, cmap='gray')
    plt.imsave(f'gray\\{filename}', im_gray, cmap='gray')

    # Data concatenation
    try:
        pos = filename.split('_')[0]
        day = int(filename.split('.')[0].split('_')[nameformat][1:])
        data[(day,pos)] = {'maj_ax':spheroid_maj_ax, 'min_ax':spheroid_min_ax,
                           'peri':spheroid_peri, 'area':spheroid_area}
    except IndexError:
        pass


# DATA ANALYSIS
# Dataframe preparation
df = pd.DataFrame(data).T.sort_index()
df.index.names = ['day','pos']

# Pixel to micrometer conversion. 1 p = 0.8 um.
for prop in df.columns:
    if prop == 'area':
        df[prop] = (df[prop]*0.64).round(3)
    else:
        df[prop] = (df[prop]*0.8).round(3)

# Shape data calculation
df['gmd'] = np.sqrt(df['maj_ax']*df['min_ax']).round(3)  # Geometric Mean Diameter
df['sf'] = np.sqrt(4*df['area']*np.pi)/df['peri']  # Shape Factor
df['v_projected'] = (4*np.pi/3) * np.sqrt((df['area']/np.pi))**3  # Volume from Projected Area
df['v_corrected'] = df['sf']*df['v_projected']  # Volume from Shape Factor correction

# Mean and standard deviation calculation
df_mean = df.mean(level=0)
df_std = df.std(level=0)
df_stats = pd.concat([df_mean, df_std], 1, keys=['mean', 'std'])

# Data saving
df.to_csv('data\\data.csv')
df_stats.to_csv('data\\data_stats.csv')


# GRAPH PLOTTING
plt.style.use('seaborn-ticks')
prop_label = {'maj_ax':'Spheroid Major Axis ($\mu m$)',
              'min_ax':'Spheroid Minor Axis ($\mu m$)',
              'peri':'Spheroid Perimeter ($\mu m$)',
              'area':'Spheroid Area ($\mu m^2$)',
              'gmd':'Spheroid Geometric Mean Diameter ($\mu m$)',
              'sf':'Spheroid Shape Factor',
              'v_projected':'Spheroid Volume\nfrom Projected Area ($\mu m^3$)',
              'v_corrected':'Spheroid Volume\nfrom Shape Factor Correction ($\mu m^3$)'}
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(15,7))
for n,ax in enumerate(axes.ravel()):
    prop=df.columns[n]
    ax.errorbar(df_stats.index, df_stats['mean'][prop], df_stats['std'][prop], fmt='-o', label='Spheroids')
    ax.set_xlabel('Day Number')
    ax.set_ylabel(prop_label[prop])
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,5))
plt.setp(axes, xticks=df.index.levels[0])
plt.tight_layout()

# Graph saving
fig.savefig('data\\plot.jpg')

print(f"The results are saved in 'bw','gray','data' folders in {os.getcwd()}.")
