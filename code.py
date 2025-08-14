# -*- coding: utf-8 -*-
"""

Tropical_Rainforest_Deforestation
Created on Tue Jul 22 00:16:35 2025

@author: Admin
"""



from sklearn.metrics import confusion_matrix, cohen_kappa_score
from sklearn.metrics import recall_score, classification_report
from sklearn.metrics import accuracy_score, jaccard_score
from sklearn.metrics import f1_score, precision_score 
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import normalize
from rasterio.features import geometry_mask
from rasterio.features import rasterize
from scipy.interpolate import griddata
from skimage.transform import resize
from rasterio.fill import fillnodata
from tensorflow.keras import layers
from scipy.fft import fft2, ifft2
from shapely.geometry import box
import matplotlib.pyplot as plt
from rasterio.mask import mask
from skimage.io import imread
from tensorflow import keras
from pathlib import Path
import geopandas as gpd
import tensorflow as tf
import numpy as np
import rasterio
import tifffile
import math
import json
import os


# VH_data_Nan= 'enter the path'
VH_data_train = 'enter the path'
# VV_data_train = 'enter the path of vv data for training'
mask_output_vh = 'enter the mask output path for vh'
# mask_output_vv = 'enter the mask output path for vv'
polygon = 'enter the path of polygon file'

#-------------------------------Fill_Nan_Value_using_Spatial_Interpolation------------------------------#
'''
def fill_nan_cubic(band):
    # band: 2D array with NaNs
    rows, cols = band.shape
    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
    # Mask of valid points
    valid_mask = ~np.isnan(band)
    # Points and values for interpolation
    points = np.column_stack((X[valid_mask], Y[valid_mask]))
    values = band[valid_mask]
    # Interpolate at all points
    band_filled = griddata(points, values, (X, Y), method='cubic')
    # If cubic fails (e.g. for edge pixels), fall back to nearest
    nan_mask = np.isnan(band_filled)
    if np.any(nan_mask):
        band_filled[nan_mask] = griddata(points, values, (X, Y), method='nearest')[nan_mask]
    return band_filled

with rasterio.open(VH_data_Nan) as n:
    profile = n.profile
    bands = n.read()  # shape: (30, 1000, 1000)
    filled_bands = np.empty_like(bands)
    for i in range(bands.shape[0]):
        print(f"Processing band {i+1}/{len(bands)}...")
        filled_bands[i] = fill_nan_cubic(bands[i])

# Save the filled raster
with rasterio.open(VH_data, "w", **profile) as dst:
    dst.write(filled_bands)

'''

#-------------------------------------clipping & Masking --------------------------#


# Load the GeoJSON polygon
gdf = gpd.read_file(polygon)


# Open the TIFF file
with rasterio.open(VH_data_train) as r:                                           #change
    r.read()
    if gdf.crs != r.crs:
        gdf = gdf.to_crs(r.crs)
    assert r.crs == gdf.crs
    
# #Check the CRS
# print("Raster CRS:", r.crs)
# print("GeoDataFrame CRS:", gdf.crs)

# Clip the raster using all polygons, crop=True to get only the clipped region
    out_image, out_transform = mask(r, gdf.geometry, crop=True)
    out_meta = r.meta.copy()

# Update metadata for the new dimensions
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform,
    "count": out_image.shape[0]
})



# Save the clipped stack
with rasterio.open(mask_output_vh, 'w', **out_meta) as dest:           #change
    dest.write(out_image)



#-----------------Prepearing_the_data------------------------#


change_polygons = gdf[gdf['id'] == 1]
unchange_polygons = gdf[gdf['id'] == 0]



with rasterio.open(mask_output_vh) as src:              #change
    transform = src.transform
    shape = (src.height, src.width)
    # Create masks: True = outside polygons, False = inside
    mask_change = geometry_mask(change_polygons.geometry, out_shape=shape, transform=transform, invert=True)
    mask_unchange = geometry_mask(unchange_polygons.geometry, out_shape=shape, transform=transform, invert=True)
    # Read all bands: shape (26, height, width)
    stack = src.read()



change_indices = np.where(mask_change)
unchange_indices = np.where(mask_unchange)


change_pixels = stack[:, change_indices[0], change_indices[1]].T
unchange_pixels = stack[:, unchange_indices[0], unchange_indices[1]].T

#change_pixel
change_pixels = change_pixels.reshape(-1, 1, 1, len(change_pixels[0]))
unchange_pixels = unchange_pixels.reshape(-1, 1, 1, len(unchange_pixels[0]))

#create_label
change_labels = np.ones((change_pixels.shape[0], 1, 1, 1), dtype=np.uint8)
unchange_labels = np.zeros((unchange_pixels.shape[0], 1, 1, 1), dtype=np.uint8)





#-----------------------Trainig_Validation_dataset-----------------#
# Split changed pixels

X_change_train, X_change_val, Y_change_train, Y_change_val = train_test_split(
    change_pixels, change_labels, test_size=0.2, random_state=42, shuffle=True
)


# Split unchanged pixels

X_unchange_train, X_unchange_val, Y_unchange_train, Y_unchange_val = train_test_split(
    unchange_pixels, unchange_labels, test_size=0.2, random_state=42, shuffle=True
)



# Combine for final training and validation sets
X_train = np.concatenate([X_change_train, X_unchange_train], axis = 0 )
Y_train = np.concatenate([Y_change_train, Y_unchange_train], axis =0)


X_val = np.concatenate([X_change_val, X_unchange_val], axis=0)
Y_val = np.concatenate([Y_change_val, Y_unchange_val], axis=0)


#----------------------------------------------Model--------------------------------------------#
import tensorflow as tf
from tensorflow.keras import layers, Model

def double_conv_block(x, n_filters):
    x = layers.Conv2D(n_filters, 1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    x = layers.Conv2D(n_filters, 1, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x

def build_deep_unet_vector_variable(input_shape=(1, 1, change_pixels.shape[3]), base_filters=16, depth=5, num_classes=1):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    skips = []

    # Encoder (5 blocks, no spatial pooling)
    for d in range(depth):
        x = double_conv_block(x, base_filters * (2 ** d))
        skips.append(x)

    # Bottleneck
    x = double_conv_block(x, base_filters * (2 ** depth))

    # Decoder (5 blocks, no upsampling, with skip connections)
    for d in reversed(range(depth)):
        x = layers.concatenate([x, skips[d]])
        x = double_conv_block(x, base_filters * (2 ** d))

    # Output layer
    activation = "sigmoid" if num_classes == 1 else "softmax"
    outputs = layers.Conv2D(num_classes, 1, padding="same", activation=activation)(x)

    return Model(inputs, outputs, name="Deep-Vector-U-Net-Variable")

# Example usage:
model_variable = build_deep_unet_vector_variable(input_shape=(1, 1, change_pixels.shape[3]), base_filters=8, depth=5, num_classes=1)
model_variable.summary()



#----------------------------------------------------Model_training---------------------------#

model = build_deep_unet_vector_variable()
model.compile(optimizer='adam',loss= 'binary_crossentropy',metrics = ['accuracy'])

history= model.fit(X_train,Y_train,
                   epochs = 20,
                   batch_size= 8,
                   validation_data = (X_val,Y_val)
    )

#---------------------------------------------------------Model_Efficiency--------------------------#
val_preds = model.predict(X_val) > 0.5
# from sklearn.metrics import accuracy_score, jaccard_score




# Flatten arrays for sklearn metrics
y_true = Y_val.flatten()
y_pred = val_preds.flatten()

# Calculate metrics
print("\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
# print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred))

print("Cohen Kappa Score:", cohen_kappa_score(y_true, y_pred))
print("F1 Score:", f1_score(y_true, y_pred))
print("Dice Coefficient:", f1_score(y_true, y_pred))  # Same as F1 for binary
print("IoU (Jaccard):", jaccard_score(y_true, y_pred))
print("Validation Accuracy:", accuracy_score(Y_val.flatten(), val_preds.flatten()))


#---------------------------------------Save_the_accuracy_parameter_fancy--------------------------------#
'''
# Compute metrics
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred)
kappa = cohen_kappa_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
iou = jaccard_score(y_true, y_pred)
acc = accuracy_score(Y_val.flatten(), val_preds.flatten())

# Prepare metrics text (shortened for clarity)
metrics_text = (
    f"Classification Report:\n{report}\n"
    f"Cohen Kappa Score: {kappa:.3f}\n"
    f"F1 Score: {f1:.3f}\n"
    f"Dice Coefficient: {f1:.3f}\n"
    f"IoU (Jaccard): {iou:.3f}\n"
    f"Validation Accuracy: {acc:.3f}"
)

# Create subplots: 1 row, 2 columns
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.2]})
fig.suptitle('H-Alpha_ Results', fontsize=32, y=1.03)

# Plot confusion matrix
im = ax1.imshow(cm, cmap='Blues')
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax1.text(j, i, cm[i, j], ha='center', va='center', color='black', fontsize=14, fontweight='bold')

ax1.set_xlabel('Predicted', fontsize=12)
ax1.set_ylabel('True', fontsize=12)
ax1.set_title('Confusion Matrix', fontsize=14)
ax1.set_xticks(np.arange(cm.shape[1]))
ax1.set_yticks(np.arange(cm.shape[0]))

# Remove axis from the second subplot and add metrics text
ax2.axis('off')
ax2.text(0, 1, metrics_text, fontsize=17, va='top', ha='left', family='monospace')

plt.tight_layout(rect=[0, 0, 1, 0.97])
# plt.savefig('optimized_classification_metrics.png', dpi=300)
plt.show()

'''


#-----------------------------Testing-----------------------#

VH_data = 'enter the testing path'

with rasterio.open(VH_data) as src:                                                      # Change
    general_data = src.read()
H, W = general_data.shape[1],general_data.shape[2]
X_general = general_data.transpose(1,2,0).reshape(-1, 1, 1, general_data.shape[0])
y_general_pred = model.predict(X_general)
y_general_mask = y_general_pred.reshape(H, W)
y_general_mask = np.nan_to_num(y_general_mask, nan=0)

#-----------------------Thresholding to get binary---------------#
from skimage.filters import threshold_otsu

thresh = threshold_otsu(y_general_mask)


#-------------border----------------#

import matplotlib.patches as patches

changes = (y_general_mask > thresh).astype(np.uint8)


fig, ax = plt.subplots()
im = ax.imshow(changes, cmap='Reds')                                                  #Change
ax.axis('off')
ax.set_title('Indore_Changes_H-Alpha(A)', pad = 20)                                           #change

# Get the dimensions of the image
height, width = changes.shape

# Add a rectangle patch as a border
rect = patches.Rectangle(
    (0, 0),                # (x, y) - bottom left corner
    width, height,         # width, height of the rectangle
    linewidth=2,           # thickness of the border
    edgecolor='black',       # border color
    facecolor='none'       # no fill
)
ax.add_patch(rect)
# plt.colorbar()
plt.show()


#----------------------------Save_georeferenced_mask_output------------------------#
'''This is the changed data after performing UNET'''
saved_image = 'enter the path to save the change map'

with rasterio.open(VH_data) as src:
    profile = src.profile.copy()
    profile.update(
        dtype=rasterio.uint8,  # or the appropriate dtype for your data
        count=1,
        height=changes.shape[0],
        width=changes.shape[1]
    )


with rasterio.open(saved_image, 'w', **profile) as dst:
    dst.write(changes,1)

#---------------------LULC_Data_Operation-------------------#
'''Basically LULC data gives the standerd high procced data for different land cover. So from there 
    we can get my desired area which coverd by desired object. and if we do some operation this perticular object area's
    pixel value will be "1" and remaining will be "0". So multiplying by saved_image and LULC data we can get the best
    result.
    
After Saving the Output We have to basically download the LULC Data of the respective year.
    this data will cover very big geographic area than our changes output.
    And also the pixel density of LULC cover and our saved output is different.
    1. If we study vegetation we have to seperate out only vegetation cover from the entire LULC cover
        for that open LULC cover in QGIS and find out the specific number whgic is denoted for vegetation,
        Then there is a python script named "Vegetation_extraction.py" run it and you will find the vegetation cover.
    
    2. We need to crop the LULC cover with matched with the saved image, otherwise multiplication will not possible.
        but due to the different pixel density Saved_image and LULC cropped image has different pixel count.
        so cropping and matching of pixel operation has been done in the python script - "LULC_Cropping.py" 
        
    3. Please take care the file locations where the file has been saved and what file is using for validation
        verification. '''


#---------------------------------Multiplying_with_LULC_vegetation cover-----------------------#
'''
saved_image = 'enter the path of saved change map'
LULC_vege_mask = 'enter the path of LULC masked path'


with rasterio.open(saved_image) as saved_image:
    mat1 = saved_image.read(1)

with rasterio.open(LULC_vege_mask) as LULC_vege:
    mat2 = LULC_vege.read(1)

effective_changes = np.multiply(mat1,mat2)                       # Perform Pixel by Pixel Classification..
# effective_changes = np.multiply(changes,mat2)                  #This is also correct..


fig, ax = plt.subplots()
im = ax.imshow(effective_changes, cmap='YlGnBu')                                                  #Change
ax.axis('off')
ax.set_title("LULC_Applied_Changes", pad = 20)                                           #change

# Get the dimensions of the image
height, width = changes.shape

# Add a rectangle patch as a border
rect = patches.Rectangle(
    (0, 0),                # (x, y) - bottom left corner
    width, height,         # width, height of the rectangle
    linewidth=2,           # thickness of the border
    edgecolor='black',       # border color
    facecolor='none'       # no fill
)
ax.add_patch(rect)

cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Change Intensity')

plt.show()

'''


#---------------------------------Saving_the model-------------------#
model_path = 'enter the path to save the model'
model.save(model_path) 
