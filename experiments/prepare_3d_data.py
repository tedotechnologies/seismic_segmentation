import os
import numpy as np
from tqdm import tqdm

cube_shape = (256, 256, 256)
amp_dtype = np.float32
mask_dtype = np.uint32


def load_cube_raw(filename, shape, dtype):
    data = np.fromfile(filename, dtype=dtype)
    if data.size != np.prod(shape):
        raise ValueError(f"Data size {data.size} does not match expected shape {shape}")
    return data.reshape(shape)


noise_dir = "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/noise"
karst_dir = "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/karst"


seismic_out_dir = "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/seismic"
label_out_dir   = "/home/dmatveev/workdir/rosneft_segmentation/data/paleokart/label"
os.makedirs(seismic_out_dir, exist_ok=True)
os.makedirs(label_out_dir, exist_ok=True)

noise_files = sorted(os.listdir(noise_dir))
karst_files = sorted(os.listdir(karst_dir))

num_slices = 10

for amp_filename, mask_filename in tqdm(zip(noise_files, karst_files)):
    amp_path = os.path.join(noise_dir, amp_filename)
    mask_path = os.path.join(karst_dir, mask_filename)
    
    cube_amp = load_cube_raw(amp_path, cube_shape, amp_dtype)
    cube_mask = load_cube_raw(mask_path, cube_shape, mask_dtype)
    
    total_slices = cube_shape[0]
    slice_indices = np.linspace(0, total_slices - 1, num=num_slices, dtype=int)
    
    base_name = os.path.splitext(amp_filename)[0]
    
    for idx in slice_indices:
        amp_slice = cube_amp[idx, :, :].astype(np.float32)
        mask_slice = cube_mask[idx, :, :].astype(np.float32)

        amp_slice_path = os.path.join(seismic_out_dir, f"{base_name}_slice_{idx:03d}.dat")
        mask_slice_path = os.path.join(label_out_dir, f"{base_name}_slice_{idx:03d}.dat")
        
        amp_slice.tofile(amp_slice_path)
        mask_slice.tofile(mask_slice_path)
        
        # print(f"Сохранён срез {idx} для куба {base_name}")
