# SpheroFilm-analyzer
An image processing and data analysis tool for spheroids cultured in INCYTO SpheroFilm with 300 um diameter.
The script has been tested and utilised with microscopic images taken from phase contrast microscope at 4x magnification. The input images are to be at 2048 x 1536 in dimension. Sample images can be found in ".\sample". The output images (greyscale and binary) are cropped from input images at 371 x 371 in dimension. This size is sufficient to contain the spheroids from the sample images.  
In the event of input images with different dimension than the sample images, or the spheroid images were taken at different magnification, the following parameters can be changed as the user sees fit:

In **PROCESS PREPARATION > Variables and functions definition**  
``circle_mask = np.pad(disk(165), ((20, 20),(20, 20)), 'constant', constant_values=0)``  

In **im_proc > Image greyscale normalisation**  
``im_n = normal_hist(im)[:,256:1792]``  

In **im_proc > Image cropping**  
``im_center = im_n[r - 185:r + 186, c - 185:c + 186]``  

In **im_proc > Border gap fill**  
``for pos in [(0,0),(0,370),(370,0),(370,370)]:
  im_gauss_gap=flood_fill(im_gauss_gap.astype('int'),pos,1)``
  
