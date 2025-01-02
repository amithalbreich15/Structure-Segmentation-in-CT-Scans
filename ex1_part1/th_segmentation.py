# nibabel is the required library for nifti files.

### SUBMITTED BY: Amit Halbreich, ID: 208917393 ###
### CASMIP Course EX1 - Part 1 ###

import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import label as lab, binary_dilation
from scipy.ndimage import binary_closing, binary_fill_holes, find_objects
from skimage.morphology import remove_small_objects, binary_dilation, \
    binary_opening, binary_closing, remove_small_holes, \
    binary_erosion
from skimage.measure import label

# Load the NIFTI file using a raw string for the path
FINAL_THRESHOLD = 501
INITIAL_THRESHOLD = 150
STEP = 14
nifti_file_path = r'MIP Data\Case1_Aorta.nii.gz'
nifti_file = nib.load(nifti_file_path)
skeleton_seg_nifti_path = r'MIP ' \
                          r'Data\Case3_CT.nii.gz_SkeletonSegmentation.nii.gz'


def segmentation_by_th(nifti_file_path, i_min, i_max):
    """
    This function is given as inputs a grayscale NIFTI file (.nii.gz) and
    two integers – the minimal and maximal thresholds. The function
    generates a segmentation NIFTI file of the same dimensions, with a
    binary segmentation – 1 for voxels between i_min and i_max ,
    0 otherwise. This segmentation NIFTI file is saved under the name
    <nifti_file>_seg_<i_min >_<i_max >.nii.gz.
    :param nifti_file: Path to the input NIFTI file.
    :return: The function returns 1 if successful,
    and 0 otherwise. The function raises descriptive errors when returning 0.
    """
    try:
        # Load the NIFTI file
        nifti_data = nib.load(nifti_file_path)

        # Getting a pointer to the data
        img_data = nifti_data.get_fdata()

        # Create boolean masks for values below i_min and above or equal to
        # i_max
        low_values_flags = img_data < i_min
        high_values_flags = img_data >= i_max

        # Set values outside the range [i_min, i_max] to 0 and 1
        img_data[low_values_flags] = 0
        img_data[high_values_flags] = 1

        # Create a new NiftiImage
        segmented_nifti = nib.Nifti1Image(img_data, nifti_data.affine)

        # Save the segmented NIFTI file
        output_filename = f"{nifti_file_path}_seg_{i_min}_{i_max}.nii.gz"
        nib.save(segmented_nifti, output_filename)
        return 1  # Successfully segmented and saved

    except Exception as e:
        print(f"Error: {e}")
        return 0  # Return 0 if an error occurs


def skeleton_th_finder(nifti_file):
    """
    Performs thresholding, segmentation, and post-processing to extract the
    skeleton CT cases with preferably 1 Connectivity Component for good
    resolution CT scans and a few for bad resolution CT scans to avoid Model
    Construction with organs in addition to the skeleton,
    scans.
    :param nifti_file: Path to the input NIFTI file.
    :return: The selected optimal threshold value. The function raises
    descriptive errors when returning 0.
    """
    try:
        # Range of candidate i_min thresholds
        i_min_range = range(INITIAL_THRESHOLD, FINAL_THRESHOLD, STEP)
        num_components = []

        # Iterate over candidate i_min thresholds
        for i_min_candidate in i_min_range:
            # Perform segmentation using segmentation_by_th
            segmentation_by_th(nifti_file, i_min_candidate, 1300)

            # Load the segmented NIFTI file
            seg_file_path = f"{nifti_file}_seg_{i_min_candidate}_1300.nii.gz"
            segmented_data = nib.load(seg_file_path).get_fdata()

            # Count the number of connectivity components
            labeled_array, num_labels = lab(segmented_data)

            # Append the number of components to the list
            num_components.append(num_labels)

        # Plot the results
        plt.plot(i_min_range, num_components, marker='o')
        plt.xlabel('i_min Threshold')
        plt.ylabel('Number of Connectivity Components')
        plt.title('Number of Connectivity Components vs i_min Threshold')
        plt.show()

        # Calculate i_min Threshold final value
        i_min_final = INITIAL_THRESHOLD + STEP * np.argmin(num_components) \
            if num_components[0] != np.min(num_components) \
            else INITIAL_THRESHOLD
        print("Selected i min is: {}".format(i_min_final))

        # Perform post-processing to get a single connectivity component
        segmentation_by_th(nifti_file, i_min_final, 1300)

        # Load the segmented NIFTI file after post-processing
        seg_file_path_post = f"{nifti_file}_seg_{i_min_final}_1300.nii.gz"
        segmented_data_post = nib.load(seg_file_path_post).get_fdata()
        post_processed_data = segmented_data_post
        if "HardCase" in nifti_file:
            # Remove small objects
            if "1" in nifti_file:
                post_processed_data = binary_fill_holes(
                    binary_closing(post_processed_data))
                post_processed_data = binary_dilation(post_processed_data)
            if "2" in nifti_file:
                post_processed_data = remove_small_objects(
                    segmented_data_post.astype(bool), min_size=1200)
                post_processed_data = binary_fill_holes(
                    binary_closing(post_processed_data))
                post_processed_data = binary_dilation(post_processed_data)
            if "3" in nifti_file:
                post_processed_data = binary_dilation(
                    segmented_data_post)
                post_processed_data = binary_closing(
                    post_processed_data)
                post_processed_data = binary_fill_holes(post_processed_data)
                post_processed_data = remove_small_objects(
                    post_processed_data.astype(bool), min_size=5000)
                # Apply morphological operations for post-processing
                post_processed_data = binary_dilation(
                    binary_fill_holes(post_processed_data))
                # Remove small objects
                post_processed_data = remove_small_objects(
                    post_processed_data.astype(bool), min_size=5000)
            if "4" in nifti_file:
                post_processed_data = binary_dilation(
                    segmented_data_post)
                post_processed_data = binary_closing(
                    post_processed_data)
                post_processed_data = binary_fill_holes(post_processed_data)
                post_processed_data = remove_small_objects(
                    post_processed_data.astype(bool), min_size=5000)
                # Apply morphological operations for post-processing
                post_processed_data = binary_dilation(
                    binary_fill_holes(post_processed_data))
                # Remove small objects
                post_processed_data = remove_small_objects(
                    post_processed_data.astype(bool), min_size=5000)
            if "5" in nifti_file:
                post_processed_data = remove_small_objects(
                    segmented_data_post.astype(bool), min_size=64)
                post_processed_data = binary_fill_holes(
                    binary_closing(post_processed_data))
                post_processed_data = binary_dilation(post_processed_data)
        else:
            if "4" in nifti_file:
                # Remove small objects
                post_processed_data = remove_small_objects(
                    segmented_data_post.astype(bool), min_size=500000)

                # Apply morphological operations for post-processing
                post_processed_data = binary_fill_holes(
                    binary_closing(post_processed_data))

                post_processed_data = binary_fill_holes(post_processed_data)
            elif "3" in nifti_file:
                # Remove small objects
                post_processed_data = remove_small_objects(
                    segmented_data_post.astype(bool), min_size=400000)

                # Apply morphological operations for post-processing for case 3
                post_processed_data = binary_opening(post_processed_data)
                post_processed_data = binary_fill_holes(post_processed_data)

                # Remove small objects
                post_processed_data = remove_small_objects(
                    post_processed_data.astype(bool), min_size=180000)

                post_processed_data = binary_fill_holes(post_processed_data)
                post_processed_data = binary_closing(post_processed_data)
                post_processed_data = binary_fill_holes(post_processed_data)

                # Remove small objects
                post_processed_data = remove_small_objects(
                    post_processed_data.astype(bool), min_size=200000)
            else:
                # Remove small objects
                post_processed_data = remove_small_objects(
                    segmented_data_post.astype(bool), min_size=300000)

                # Apply morphological operations for post-processing
                post_processed_data = binary_fill_holes(
                    binary_closing(post_processed_data))

                post_processed_data = binary_dilation(
                    binary_fill_holes(post_processed_data))

        # Save the final segmentation NIFTI file
        final_output_filename = f"{nifti_file}_SkeletonSegmentation.nii.gz"
        final_segmented_nifti = nib.Nifti1Image(
            post_processed_data.astype(np.uint8), nib.load(nifti_file).affine)
        nib.save(final_segmented_nifti, final_output_filename)

        res, num_components = label(post_processed_data, return_num=True)
        print("Number of connectivity components: {}".format(num_components))

        # Save the final segmentation NIFTI file
        final_output_filename = f"{nifti_file}_SkeletonSegmentation.nii.gz"
        final_segmented_nifti = nib.Nifti1Image(
            post_processed_data.astype(np.uint8), nib.load(nifti_file).affine)
        nib.save(final_segmented_nifti, final_output_filename)
        return i_min_final

    except Exception as e:
        print(f"Error: {e}")
        return None


def nifti_to_img_converter(nifti_file_path, save_path='output_img_data.npy'):
    """
   Function Description:
   Converts a NIFTI file to image data and generates visualizations, including axial, sagittal, and coronal slices,
   a montage of axial slices, and saves the entire image data. The visualizations are saved as PNG files with relevant
   titles.

   Visualization Outputs:
   - Axial Slice at the Middle
   - Sagittal Slice (Rotated 270 degrees)
   - Coronal Slice (Rotated 270 degrees)
   - Montage of Axial Slices at the Middle

   Additionally, the entire image data is saved as an output file.

   :param nifti_file_path: Path to the input NIFTI file.
   :param save_path: Path to save the output image data file (default:
          'output_img_data.npy').
   :return: Image data as an array or None if an error occurs.
   """
    try:
        # Load the NIFTI file
        nifti_data = nib.load(nifti_file_path)

        # Get the data
        img_data = nifti_data.get_fdata()

        # Show and save the axial slice using matplotlib
        axial_slice = np.rot90(img_data[:, :, img_data.shape[2] // 2], k=3)
        plt.imshow(axial_slice, cmap='gray', aspect='auto')
        plt.title('Axial Slice at the Middle')
        plt.colorbar()
        plt.savefig(save_path.replace('.npy', '_axial_slice.png'))
        plt.show()

        # Show and save the sagittal slice using matplotlib (rotated by 270 degrees)
        sagittal_slice = np.rot90(img_data[img_data.shape[0] // 2, :, :], k=3)
        plt.imshow(sagittal_slice, cmap='gray', aspect='auto')
        plt.title('Sagittal Slice (Rotated 270 degrees)')
        plt.colorbar()
        plt.savefig(save_path.replace('.npy', '_sagittal_slice.png'))
        plt.show()

        # Show and save the coronal slice using matplotlib (rotated by 270 degrees)
        coronal_slice = np.rot90(img_data[:, img_data.shape[1] // 2, :], k=3)
        plt.imshow(coronal_slice, cmap='gray', aspect='auto')
        plt.title('Coronal Slice (Rotated 270 degrees)')
        plt.colorbar()
        plt.savefig(save_path.replace('.npy', '_coronal_slice.png'))
        plt.show()

        # Create a montage of slices along the axial axis (change axis as needed)
        montage = np.transpose(img_data, (2, 0, 1))

        # Display and save the montage using matplotlib
        plt.imshow(montage[:, :, montage.shape[2] // 2], cmap='gray',
                   aspect='auto')
        plt.title('Montage of Axial Slices')
        plt.colorbar()
        plt.savefig(save_path.replace('.npy', '_axial_montage.png'))
        plt.show()

        # Save the entire image using numpy's save function
        np.save(save_path, img_data)

        return img_data  # Return image data as an array

    except Exception as e:
        print(f"Error: {e}")
        return None  # Return None if an error occurs


if __name__ == '__main__':
    nifti_file = 'MIP Data\\Case3_CT.nii.gz'  # Change File name to run
    # different cases try {CaseX_CT.nii.gz, HardCaseX_CT.nii.gz} Forms
    print(r"Working on nifti file name: {}".format(nifti_file))
    skeleton_seg_nifti_path = 'MIP ' \
                              'Data\\Case3_CT.nii.gz_SkeletonSegmentation.nii.gz'
    # Change File name to run different cases try: {CaseX_CT.nii.gz,
    # HardCaseX_CT.nii.gz} Forms
    segmentation_by_th(nifti_file, 234, 1300)
    selected_i_min = skeleton_th_finder(nifti_file)
    nifti_file = r'MIP Data\Case3_CT.nii.gz'  # Change File name to run
    img_array1 = nifti_to_img_converter(skeleton_seg_nifti_path)
    img_array2 = nifti_to_img_converter(nifti_file)

    if selected_i_min is None:
        print("Error: Could not find i_min threshold.")

    if img_array2 is not None:
        print("Image data converted successfully.")
        np.save('output_img_data.npy', img_array2)

