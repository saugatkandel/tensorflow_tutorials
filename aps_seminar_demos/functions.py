#Author - Saugat Kandel
# coding: utf-8


import numpy as np
import tensorflow as tf
from numpy.fft import fftshift, ifftshift, fftfreq, fft2, ifft2
import skimage, skimage.transform, skimage.data
from scipy import special



def propTF(beam_profile: np.ndarray, 
           step_size: float, 
           wavelength: float, 
           prop_dist: float):
    '''Propogation using the Transfer function method. 
    '''
    support_side_length = np.max(beam_profile.shape)
    M, N = np.shape(beam_profile)
    #k = 2 * np.pi / wavelength
    
    fx = fftfreq(M, d=step_size)
    fy = fftfreq(N, d=step_size)
    
    FX, FY = np.meshgrid(fx, fy)
    FX = fftshift(FX)
    FY = fftshift(FY)
    
    H = np.exp(-1j * np.pi * wavelength * prop_dist * (FX**2 + FY**2))
    H = fftshift(H)
    return H



def genViewIndices(det_pixels, positions, obj_shape):
    npix = det_pixels
    views_indices_all = []
    for py, px in positions:
        R, C = np.ogrid[py:npix + py, px:npix + px]
        view_single = (R % obj_shape[0]) * obj_shape[0] + (C % obj_shape[0])
        views_indices_all.append(view_single)
    return np.array(views_indices_all)



def tensor_clip(t: tf.Tensor, 
                  max_abs: float=1.,
                  min_abs: float=0.):
    
    absval = tf.abs(t)
    abs_clipped = tf.clip_by_value(absval, min_abs, max_abs)
    multiplier = tf.cast(abs_clipped / (absval), 'complex64')
    return t * multiplier



def genSpeckle(npix, window_size):
    real = np.random.randn(npix, npix)
    imag = np.random.randn(npix, npix) 
    ran = real + 1j *  imag
    
    
    window = np.zeros((npix, npix))
    indx1 = npix // 2 - window_size // 2
    indx2 = npix // 2 + window_size // 2
    window[indx1: indx2, indx1: indx2] = 1
    t = window * ran
    ft = np.fft.fftshift(np.fft.fft2(t, norm='ortho'))
    return np.abs(ft)



def getDiffractionIntensitiesAndPositions(obj, probe,prop_kernel, step_size, num_steps,  random_positions = True, poisson=True):
    diffs = []
    pos = []
    for i_row in range(num_steps):
        for i_col in range(num_steps):
            if random_positions:
                step_size_this = np.random.choice(np.arange(step_size - 20, step_size))
            else:
                step_size_this = step_size
            r1 = i_row  * step_size_this
            r2 = r1 + probe.shape[0]
            c1 = i_col * step_size_this
            c2 = c1 + probe.shape[1]
            exit_wave = probe * obj[r1:r2, c1:c2]
            diff = np.abs(ifftshift(ifft2(prop_kernel * fft2(exit_wave))))**2
            diffs.append(diff)
            pos.append([r1, c1])
    diffs = np.array(diffs)
    if poisson:
        diffs = np.random.poisson(diffs)
    positions = np.array(pos)
    return diffs, positions



def getImageData(obj_npix):
    arrdata = lambda x: skimage.img_as_float(skimage.color.rgb2gray(x))[::-1, ::-1]

    img_obj = skimage.data.immunohistochemistry()
    amp = skimage.transform.resize(arrdata(img_obj[:,:,0]), [obj_npix, obj_npix], 
                                   mode='wrap', preserve_range=True)
    phase = skimage.transform.resize(arrdata(img_obj[:,:,1]), [obj_npix, obj_npix],
                                     mode='wrap', preserve_range=True) * np.pi
    obj_true = amp * np.exp(1j * phase)
    return obj_true



def getAiryProbe(wavelength=0.142e-9, # 8.7 keV 
                 pixel_pitch=30e-9, # 55 micrometers 
                 npix=64,
                 n_photons=1e6,
                 beam_diam_pixels=20):
    """Calculates the beam profile given the final beam diameter. 
    
    Parameters:
    
    wavelength : 
    self-explanatory
    
    pixel_pitch : 
    object/probe pixel pitch. Usually calculated using the Nyquist theorem using the object-detector
    distance and the detector pixel pitch.
    
    n_pix:
    number of pixels along each axis in the probe view
    
    n_photons:
    self-explanatory
    
    beam_diam_pixels:
    the diameter (in pixels) of the first central lobe of the probe beam at the object (sample) plane.

    Assumption:
    - propagation distance (from aperture to sample) and initial aperture width are calculated 
    assuming a Fresnel number of 0.1
    """
    beam_width_dist = beam_diam_pixels * pixel_pitch
    radius = beam_width_dist / 2
    # Initial Aperture width
    w = 0.1 * 2 * np.pi * radius / (special.jn_zeros(1, 1))
    
    # Propagation dist from aperture to sample
    z = 0.1 * (2 * np.pi * radius)**2 / (special.jn_zeros(1, 1)**2 * wavelength)
    
    beam_center = npix // 2
    xvals = np.linspace( -beam_center * pixel_pitch, beam_center * pixel_pitch, npix)
    xx, yy = np.meshgrid(xvals, xvals)
    

    k = 2 * np.pi / wavelength
    lz = wavelength * z
    S = xx**2 + yy**2

    jinc_input = np.sqrt(S) * w / lz
    mask = (jinc_input != 0)
    jinc_term = np.pi * np.ones_like(jinc_input)
    jinc_term[mask] = special.j1(jinc_input[mask] * 2 * np.pi) / jinc_input[mask]

    # wavefield 
    term1 = np.exp(1j * k * z) / (1j * lz)
    term2 = np.exp(1j * k * S / (2 * z))
    term3 = w**2 * jinc_term
    field_vals = (term1 * term2 * term3).astype('complex64')

    scaling_factor = np.sqrt(n_photons / (np.abs(field_vals)**2).sum())
    field_vals = scaling_factor * field_vals
    return field_vals



def genViewIndices(det_pixels, positions, obj_shape):
    npix = det_pixels
    views_indices_all = []
    for py, px in positions:
        R, C = np.ogrid[py:npix + py, px:npix + px]
        view_single = (R % obj_shape[0]) * obj_shape[0] + (C % obj_shape[0])
        views_indices_all.append(view_single)
    return np.array(views_indices_all)



def batch_fftshift2d(tensor: tf.Tensor):
    # Shifts high frequency elements into the center of the filter
    indexes = len(tensor.get_shape()) - 1
    top, bottom = tf.split(tensor, 2, axis=indexes)
    tensor = tf.concat([bottom, top], indexes )
    left, right = tf.split(tensor, 2, axis=indexes - 1)
    tensor = tf.concat([right, left], indexes - 1 )
    return tensor

