# Import required modules
import cv2
import numpy as np


# plot signal waves
def plot_waves(sub_wave_matrix, wave_matrix, x_corr):
    import matplotlib.pyplot as plt

    plt.plot(sub_wave_matrix)
    plt.title("Sub-Wave Signal")
    plt.show()

    plt.plot(wave_matrix)
    plt.title("Wave Signal")
    plt.show()

    # normalize the cross correlation
    plt.plot(x_corr)
    plt.title("Cross-Correlation Plot")
    plt.show()


# get audio cross correlation
def find_audio_correlation(sub_wave_matrix, wave_matrix, plot=False):
    from scipy import signal

    sub_wave_matrix = sub_wave_matrix.flatten()
    wave_matrix = wave_matrix.flatten()
    x_corr = signal.correlate(sub_wave_matrix - np.mean(sub_wave_matrix), wave_matrix - np.mean(wave_matrix),
                              mode='valid') / (np.std(sub_wave_matrix) * np.std(wave_matrix) * len(wave_matrix))
    if plot:
        plot_waves(sub_wave_matrix, wave_matrix, x_corr)

    # getSound(sub_wave_matrix,44100)
    # getSound(wave_matrix,44100)
    return np.argmin(x_corr)


# convert audio
def stereo_to_mono_wave(path):
    import soundfile as sf

    wave, fs = sf.read(path, dtype='float32')
    wave = np.delete(wave, 1, 1)
    return fs, wave


# undistort the image
# image: cv2 image Object
# matrix: intrinsics matrix
# distortion: intrinsics distortion
def undistort_image(image, matrix, distortion):
    # Compute the undistorted image
    h, w = image.shape[:2]
    # Compute the newcamera intrinsic matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(matrix, distortion, (w, h), 0)
    # Undistort image
    image_new = cv2.undistort(image, matrix, distortion, None, new_camera_matrix)

    return image_new, roi


# return total n° of frames of a video
def get_video_total_frames(video_path):
    video = cv2.VideoCapture(video_path)
    tot_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))  # total number of frames
    video.release()
    return tot_frames


# convert svg to png
def svg_to_png(image_path):
    from svglib.svglib import svg2rlg
    from reportlab.graphics import renderPDF, renderPM

    image_path_new = image_path.replace('svg', 'png')

    drawing = svg2rlg(image_path)
    renderPM.drawToFile(drawing, image_path_new, fmt="PNG")

    return image_path_new


# enchantment of an image with morphological operations
def image_enchantment(image, params):
    kernel = np.ones((5, 5), np.uint8)

    for param in params:
        if param == 'erode':
            image = cv2.erode(image, kernel, iterations=3)
        if type == 'dilation':
            image = cv2.dilate(image, kernel, iterations=3)
        if type == 'opening':
            # erosion followed by dilation: removing noise
            # image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
            image = image_enchantment(image, ['erode'])
            image = image_enchantment(image, ['dilation'])
        if type == 'closing':
            # dilation followed by erosion: closing small holes inside the foreground objects
            # image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
            image = image_enchantment(image, ['dilation'])
            image = image_enchantment(image, ['erode'])
        if type == 'gradient':
            # difference between dilation and erosion of an image
            image = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
        if type == 'tophat':
            # difference between input image and opening of the image
            image = cv2.morphologyEx(image, cv2.MORPH_TOPHAT, kernel)
        if type == 'blackhat':
            # difference between the closing of the input image and input image
            image = cv2.morphologyEx(image, cv2.MORPH_BLACKHAT, kernel)
        if type == 'sharpen':
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            image = cv2.filter2D(image, -1, kernel)

    return image


def homography_transformation(img1, kp_image, img2, kp_image2, matches):
    query_pts = np.float32([kp_image[m.queryIdx]
                           .pt for m in matches]).reshape(-1, 1, 2)

    train_pts = np.float32([kp_image2[m.trainIdx]
                           .pt for m in matches]).reshape(-1, 1, 2)

    # Calculate Homography
    h, status = cv2.findHomography(query_pts, train_pts)
    # Warp source image to destination based on homography
    im_out = cv2.warpPerspective(img1, h, (img2.shape[1], img2.shape[0]))

    cv2.imshow("Transformed", im_out)
    cv2.waitKey(0)

    return im_out

# transform image into template image whenever possible
def transform_image(template_image, image):
    # Convert images to grayscale
    im1_gray = cv2.cvtColor(template_image, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find size of image1
    sz = template_image.shape

    # Define the motion model
    warp_mode = cv2.MOTION_HOMOGRAPHY

    # Define 2x3 or 3x3 matrices and initialize the matrix to identity
    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        warp_matrix = np.eye(3, 3, dtype=np.float32)
    else:
        warp_matrix = np.eye(2, 3, dtype=np.float32)

    # Specify the number of iterations.
    number_of_iterations = 5000

    # Specify the threshold of the increment
    # in the correlation coefficient between two iterations
    termination_eps = 1e-10

    # Define termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, number_of_iterations, termination_eps)

    # Run the ECC algorithm. The results are stored in warp_matrix.
    (cc, warp_matrix) = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, warp_mode, criteria)

    if warp_mode == cv2.MOTION_HOMOGRAPHY:
        # Use warpPerspective for Homography
        im2_aligned = cv2.warpPerspective(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    else:
        # Use warpAffine for Translation, Euclidean and Affine
        im2_aligned = cv2.warpAffine(image, warp_matrix, (sz[1], sz[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    return im2_aligned
