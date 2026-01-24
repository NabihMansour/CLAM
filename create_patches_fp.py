# internal imports
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df
# other imports
import os
import numpy as np
import time
import argparse
import pdb
import pandas as pd
from tqdm import tqdm
import cv2

def stitching(file_path, wsi_object, downscale = 64):
	start = time.time()
	heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
	total_time = time.time() - start
	
	return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
	### Start Seg Timer
	start_time = time.time()
	# Use segmentation file
	if mask_file is not None:
		WSI_object.initSegmentation(mask_file)
	# Segment	
	else:
		WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

	### Stop Seg Timers
	seg_time_elapsed = time.time() - start_time   
	return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
	### Start Patch Timer
	start_time = time.time()

	# Patch
	file_path = WSI_object.process_contours(**kwargs)


	### Stop Patch Timer
	patch_time_elapsed = time.time() - start_time
	return file_path, patch_time_elapsed

# def filter_black_contours(wsi_object, contours, seg_level, threshold=15.5):
#     """
#     SATURATION FILTER (The "Color" Check).
#     Rejects contours that are grey/colorless (Markers), 
#     keeps contours that are pink/purple/brown (Tissue).
    
#     threshold (int): Saturation cutoff (0-255). 
#                      < 15 is usually Grey/Black/White. 
#                      > 20 is usually Tissue.
#     """
#     valid_contours = []

#     # 1. Get Thumbnail (Use 2048 for better resolution if possible)
#     slide = wsi_object.getOpenSlide()
#     w_full, h_full = slide.level_dimensions[0]
    
#     # We bump this to 2048 to help with the blurring issue
#     thumbnail = slide.get_thumbnail((2048, 2048))
#     w_thumb, h_thumb = thumbnail.size
#     thumb_np = np.array(thumbnail.convert('RGB'))
    
#     scale_x = w_thumb / w_full
#     scale_y = h_thumb / h_full

#     print(f"  Filtering {len(contours)} contours using Color Saturation...")

#     for i, contour in enumerate(contours):
#         x, y, w, h = cv2.boundingRect(contour)
        
# 		# check size of countour, if too big, keep it
#         contour_area = w * h
#         total_area = w_full * h_full
#         if contour_area > (total_area * 0.05):
#             valid_contours.append(contour)
#             continue

#         # Scale to thumbnail
#         x_s = int(x * scale_x); y_s = int(y * scale_y)
#         w_s = int(w * scale_x); h_s = int(h * scale_y)
        
#         # Safety checks
#         if w_s < 1 or h_s < 1: 
#             valid_contours.append(contour)
#             continue
            
#         # Crop patch
#         y_end = min(y_s + h_s, h_thumb)
#         x_end = min(x_s + w_s, w_thumb)
#         patch = thumb_np[y_s:y_end, x_s:x_end]
        
#         if patch.size == 0:
#             valid_contours.append(contour)
#             continue

#         # --- NEW LOGIC: SATURATION CHECK ---
        
#         # 1. Create Mask (To ignore white background)
#         # We need the contour relative to the patch
#         contour_thumb = (contour * [scale_x, scale_y]).astype(np.int32)
#         h_p, w_p, _ = patch.shape
#         mask = np.zeros((h_p, w_p), dtype=np.uint8)
#         cv2.drawContours(mask, [contour_thumb], -1, (255), thickness=cv2.FILLED, 
#                          offset=(-x_s, -y_s))
        
#         # 2. Convert to HSV Color Space
#         # H = Color, S = Amount of Color, V = Brightness
#         patch_hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        
#         # 3. Calculate Mean Saturation (Index 1) ONLY inside the mask
#         # cv2.mean returns (H, S, V, A)
#         mean_hsv = cv2.mean(patch_hsv, mask=mask)
#         mean_saturation = mean_hsv[1]
#         print(f"  > Contour ID {i} Mean Saturation: {mean_saturation:.1f}")
#         # 4. Decision
#         # Pink Tissue Saturation is usually > 40
#         # Black/Grey Marker Saturation is usually < 10
#         if mean_saturation > threshold:
#             valid_contours.append(contour)
#         else:
#             print(f"  > Dropped Grey Artifact ID {i} (Saturation: {mean_saturation:.1f})")

#     return valid_contours


def seg_and_patch(save_dir, patch_save_dir, mask_save_dir, stitch_save_dir, source = None, 
				  patch_size = 256, step_size = 256, 
				  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'},
				  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
				  vis_params = {'vis_level': -1, 'line_thickness': 500},
				  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
				  patch_level = 0,
				  use_default_params = False, 
				  seg = False, save_mask = True, 
				  stitch= False, 
				  patch = False, auto_skip=True, process_list = None, csv_path=None):
	

	if csv_path is not None:
		print(f"Loading slide paths from CSV: {csv_path}")
		path_df = pd.read_csv(csv_path)
		# Create a list of tuples (slide_id, path)
		slide_path_pairs = zip(path_df['slide_id'].values, path_df['full_path'].values)
		# Filter efficiently
		slides = [sid for sid, full_path in slide_path_pairs if os.path.isfile(full_path)]
	else:
		slides = sorted(os.listdir(source))
		slides = [slide for slide in slides if os.path.isfile(os.path.join(source, slide))]
		
	if process_list is None:
		df = initialize_df(slides, seg_params, filter_params, vis_params, patch_params)
	
	else:
		df = pd.read_csv(process_list)
		df = initialize_df(df, seg_params, filter_params, vis_params, patch_params)

	if csv_path is not None:
		df = pd.merge(df, path_df[['slide_id', 'full_path']], on='slide_id', how='left')

	mask = df['process'] == 1
	process_stack = df[mask]

	total = len(process_stack)

	legacy_support = 'a' in df.keys()
	if legacy_support:
		print('detected legacy segmentation csv file, legacy support enabled')
		df = df.assign(**{'a_t': np.full((len(df)), int(filter_params['a_t']), dtype=np.uint32),
		'a_h': np.full((len(df)), int(filter_params['a_h']), dtype=np.uint32),
		'max_n_holes': np.full((len(df)), int(filter_params['max_n_holes']), dtype=np.uint32),
		'line_thickness': np.full((len(df)), int(vis_params['line_thickness']), dtype=np.uint32),
		'contour_fn': np.full((len(df)), patch_params['contour_fn'])})

	seg_times = 0.
	patch_times = 0.
	stitch_times = 0.

	for i in tqdm(range(total)):
		df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
		idx = process_stack.index[i]
		slide = process_stack.loc[idx, 'slide_id']
		print("\n\nprogress: {:.2f}, {}/{}".format(i/total, i, total))
		print('processing {}'.format(slide))
		
		df.loc[idx, 'process'] = 0
		slide_id, _ = os.path.splitext(slide)

		if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
			print('{} already exist in destination location, skipped'.format(slide_id))
			df.loc[idx, 'status'] = 'already_exist'
			continue

		# Inialize WSI
		if csv_path is not None:
			full_path = process_stack.loc[idx, 'full_path']
		else:
			full_path = os.path.join(source, slide)


		WSI_object = WholeSlideImage(full_path)

		if use_default_params:
			current_vis_params = vis_params.copy()
			current_filter_params = filter_params.copy()
			current_seg_params = seg_params.copy()
			current_patch_params = patch_params.copy()
			
		else:
			current_vis_params = {}
			current_filter_params = {}
			current_seg_params = {}
			current_patch_params = {}


			for key in vis_params.keys():
				if legacy_support and key == 'vis_level':
					df.loc[idx, key] = -1
				current_vis_params.update({key: df.loc[idx, key]})

			for key in filter_params.keys():
				if legacy_support and key == 'a_t':
					old_area = df.loc[idx, 'a']
					seg_level = df.loc[idx, 'seg_level']
					scale = WSI_object.level_downsamples[seg_level]
					adjusted_area = int(old_area * (scale[0] * scale[1]) / (512 * 512))
					current_filter_params.update({key: adjusted_area})
					df.loc[idx, key] = adjusted_area
				current_filter_params.update({key: df.loc[idx, key]})

			for key in seg_params.keys():
				if legacy_support and key == 'seg_level':
					df.loc[idx, key] = -1
				current_seg_params.update({key: df.loc[idx, key]})

			for key in patch_params.keys():
				current_patch_params.update({key: df.loc[idx, key]})

		if current_vis_params['vis_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_vis_params['vis_level'] = 0
			
			else:	
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_vis_params['vis_level'] = best_level

		if current_seg_params['seg_level'] < 0:
			if len(WSI_object.level_dim) == 1:
				current_seg_params['seg_level'] = 0
			
			else:
				wsi = WSI_object.getOpenSlide()
				best_level = wsi.get_best_level_for_downsample(64)
				current_seg_params['seg_level'] = best_level

		keep_ids = str(current_seg_params['keep_ids'])
		if keep_ids != 'none' and len(keep_ids) > 0:
			str_ids = current_seg_params['keep_ids']
			current_seg_params['keep_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['keep_ids'] = []

		exclude_ids = str(current_seg_params['exclude_ids'])
		if exclude_ids != 'none' and len(exclude_ids) > 0:
			str_ids = current_seg_params['exclude_ids']
			current_seg_params['exclude_ids'] = np.array(str_ids.split(',')).astype(int)
		else:
			current_seg_params['exclude_ids'] = []

		w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
		if w * h > 1e8:
			print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
			df.loc[idx, 'status'] = 'failed_seg'
			continue

		df.loc[idx, 'vis_level'] = current_vis_params['vis_level']
		df.loc[idx, 'seg_level'] = current_seg_params['seg_level']


		seg_time_elapsed = -1
		if seg:
			WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 
			# print("Filtering dark artifacts...")
			# original_count = len(WSI_object.contours_tissue)

			# # Retrieve the seg_level used during segmentation
			# active_seg_level = current_seg_params['seg_level']

			# # Call the new optimized function
			# WSI_object.contours_tissue = filter_black_contours(
			# 	WSI_object, 
			# 	WSI_object.contours_tissue, 
        	# 	seg_level=active_seg_level, 
        	# 	threshold=15.5)
			# print(f"Dropped {original_count - len(WSI_object.contours_tissue)} contours.")

		if save_mask:
			mask = WSI_object.visWSI(**current_vis_params)
			mask_path = os.path.join(mask_save_dir, slide_id+'.jpg')
			mask.save(mask_path)

		patch_time_elapsed = -1 # Default time
		if patch:
			current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
										 'save_path': patch_save_dir})
			file_path, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
		
		stitch_time_elapsed = -1
		if stitch:
			file_path = os.path.join(patch_save_dir, slide_id+'.h5')
			if os.path.isfile(file_path):
				heatmap, stitch_time_elapsed = stitching(file_path, WSI_object, downscale=64)
				stitch_path = os.path.join(stitch_save_dir, slide_id+'.jpg')
				heatmap.save(stitch_path)

		print("segmentation took {} seconds".format(seg_time_elapsed))
		print("patching took {} seconds".format(patch_time_elapsed))
		print("stitching took {} seconds".format(stitch_time_elapsed))
		df.loc[idx, 'status'] = 'processed'

		seg_times += seg_time_elapsed
		patch_times += patch_time_elapsed
		stitch_times += stitch_time_elapsed

	seg_times /= total
	patch_times /= total
	stitch_times /= total

	df.to_csv(os.path.join(save_dir, 'process_list_autogen.csv'), index=False)
	print("average segmentation time in s per slide: {}".format(seg_times))
	print("average patching time in s per slide: {}".format(patch_times))
	print("average stiching time in s per slide: {}".format(stitch_times))
		
	return seg_times, patch_times

parser = argparse.ArgumentParser(description='seg and patch')
parser.add_argument('--source', type = str, default=None,
					help='path to folder containing raw wsi image files')
parser.add_argument('--step_size', type = int, default=256,
					help='step_size')
parser.add_argument('--patch_size', type = int, default=256,
					help='patch_size')
parser.add_argument('--patch', default=False, action='store_true')
parser.add_argument('--seg', default=False, action='store_true')
parser.add_argument('--stitch', default=False, action='store_true')
parser.add_argument('--no_auto_skip', default=True, action='store_false')
parser.add_argument('--save_dir', type = str,
					help='directory to save processed data')
parser.add_argument('--preset', default=None, type=str,
					help='predefined profile of default segmentation and filter parameters (.csv)')
parser.add_argument('--patch_level', type=int, default=0, 
					help='downsample level at which to patch')
parser.add_argument('--process_list',  type = str, default=None,
					help='name of list of images to process with parameters (.csv)')
parser.add_argument('--csv_path', type=str, default=None, help='Path to CSV with slide_id and full_path columns')

if __name__ == '__main__':
	args = parser.parse_args()

	if args.source is None and args.csv_path is None:
		parser.error("You must provide either --source OR --csv_path.")

	patch_save_dir = os.path.join(args.save_dir, 'patches')
	mask_save_dir = os.path.join(args.save_dir, 'masks')
	stitch_save_dir = os.path.join(args.save_dir, 'stitches')

	if args.process_list:
		process_list = os.path.join(args.save_dir, args.process_list)

	else:
		process_list = None

	if args.csv_path:
		print('Loading slide paths from CSV: ', args.csv_path)
	else:
		print('Loading slide paths from source directory: ', args.source)
	print('patch_save_dir: ', patch_save_dir)
	print('mask_save_dir: ', mask_save_dir)
	print('stitch_save_dir: ', stitch_save_dir)
	
	directories = {'save_dir': args.save_dir,
				   'patch_save_dir': patch_save_dir, 
				   'mask_save_dir' : mask_save_dir, 
				   'stitch_save_dir': stitch_save_dir}
	if args.source is not None:
		directories['source'] = args.source 

	for key, val in directories.items():
		print("{} : {}".format(key, val))
		if key not in ['source']:
			os.makedirs(val, exist_ok=True)

	seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,
				  'keep_ids': 'none', 'exclude_ids': 'none'}
	filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
	vis_params = {'vis_level': -1, 'line_thickness': 250}
	patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

	if args.preset:
		preset_df = pd.read_csv(os.path.join('presets', args.preset))
		for key in seg_params.keys():
			seg_params[key] = preset_df.loc[0, key]

		for key in filter_params.keys():
			filter_params[key] = preset_df.loc[0, key]

		for key in vis_params.keys():
			vis_params[key] = preset_df.loc[0, key]

		for key in patch_params.keys():
			patch_params[key] = preset_df.loc[0, key]
	
	parameters = {'seg_params': seg_params,
				  'filter_params': filter_params,
	 			  'patch_params': patch_params,
				  'vis_params': vis_params}

	print(parameters)

	seg_times, patch_times = seg_and_patch(**directories, **parameters,
											patch_size = args.patch_size, step_size=args.step_size, 
											seg = args.seg,  use_default_params=False, save_mask = True, 
											stitch= args.stitch,
											patch_level=args.patch_level, patch = args.patch,
											process_list = process_list, auto_skip=args.no_auto_skip, csv_path=args.csv_path)
