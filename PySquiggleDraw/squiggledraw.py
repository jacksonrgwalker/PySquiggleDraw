
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import block_reduce
from io import BytesIO
import cairo
import warnings


class SquiggleDrawer:

    def __init__(self) -> None:
        
        self.image = None
        self.aspect_ratio = None
        self.squiggles = None
        self.svgio = None
        
    def read_in_pixel_data(self, fname=None):
        """
        Sefely reads in the image pixel data a stores it as a numpy array. 
        """

        if fname is None: fname = self.image_file_path

        with Image.open(fname) as image:
            image_pixel_data = np.asarray(image, dtype="int32")

        self.aspect_ratio = image_pixel_data.shape[1] / image_pixel_data.shape[0]
        self.image = image_pixel_data

    def rgb_to_luminance(self):
        """
        Function to convert RGB values to luminance values. 
        """

        image_pixel_data = self.image

        image_lum_data = \
            0.2126 * image_pixel_data[..., 0] + \
            0.7152 * image_pixel_data[..., 1] + \
            0.0722 * image_pixel_data[..., 2] 

        image_lum_data = image_lum_data / 255

        image_lum_data = 1 - image_lum_data

        self.image = image_lum_data

    def increase_contrast(self, cutoff):
        """
        Increase contrast of image by applying a threshold.

        Inputs
        ------
        cutoff : int between 0 and 100, percentile threshold value where pixels below this value are set to 0, and pixels above this value are set to 1, and intermediate values are linearly stretched.
        """

        lum_data = self.image

        minval = np.percentile(lum_data, cutoff)
        maxval = np.percentile(lum_data, 100-cutoff)
        
        lum_data_contrasted = np.clip(lum_data, minval, maxval)
        lum_data_contrasted = (lum_data_contrasted - minval) / (maxval - minval)

        self.image = lum_data_contrasted

    def show_image(self, size=10):
        """
        Helper function to display image.

        Inputs
        ------
        size : int, size of image to display
        """

        plt.figure(figsize=(size * 2 * self.aspect_ratio, size))
        plt.imshow(self.image, cmap="Greys")
        plt.axis("off");

    @staticmethod
    def phase_reconciliation(wave_numbers):
        """
        Function to reconcile phase of consectuive waves so that they match up. 
        e.g. When we have a wave with 1.5 periods follow by a wave of 1 period, then we have to phase shift the second wave so that it matches up with the first wave.
        This has to be done itertivelty as far as I know, since whether the next wave needs to be shifted depends on if the previous wave is shifted or not.

        Visual:
        -------

        1.5 period wave  followed by  1 period wave       These don't match up:    So we phase shift the second wave by a half period: 
            _     _                       _                 _     _  _                 _     _     _                       
        / \   / \                     / \               / \   / \/ \               / \   / \   / \             
            \_/                           \_/               \_/       \_/              \_/   \_/                   

        Inputs
        ------
        wave_numbers : numpy array, wave numbers of each wave (1d array)

        Outputs
        -------
        phase_shift : numpy array of 1 if phase shift is needed at that index, 0 otherwise 
        """

        # WLOG, we don't need to shift the first wave
        phase_shift = [0]

        # looking to see if we need a phase shift for the wave after wave w
        for w in wave_numbers[:-1]:

            # initally, just do a phase shift if the wave number is odd 
            naive_adjust = w % 2

            # if we shifted the last one, do the opposite this time
            if phase_shift[-1]: 
                naive_adjust = 1 - naive_adjust 

            # record whether we shifted this one or not
            phase_shift.append(naive_adjust)

        return np.array(phase_shift)

    def prep_image(self, fname, contrast_cutoff, downsample_amount):
        """
        Orchestrates the entire process of reading in the image, converting it to luminance, increasing contrast, and downsampling.

        Inputs
        ------
        fname : str, path to image file
        contrast_cutoff : int between 0 and 100, percentile threshold value where pixels below this value are set to 0, and pixels above this value are set to 1, and intermediate values are linearly stretched.
        downsample_amount : int or 2-tuple in int, factor to downsample image by. If 2-tuple, then amount to downsample each axis respectively. Passed as the block_size arg to skimage.measure.block_reduce (https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.block_reduce)

        """

        self.read_in_pixel_data(fname)
        self.rgb_to_luminance()
        self.increase_contrast(cutoff = contrast_cutoff)
        self.image = block_reduce(self.image, block_size=downsample_amount, func=np.mean)

    def calculate_squiggles(self, 
                    max_wave_num = 5,
                    wave_num_threshold = 1.2,
                    max_amplitude= 1,
                    amp_threshold = .5,
                    resolution = 20,
                    ):
        """
        Calculates the squiggles in the image. Note that this function does not actually plot or render the squiggles, just figures out the equation of the squiggles and approximates them with with some resolution. 

        Inputs
        ------
        max_wave_num : int, maximum number of half periods for each pixel/downsampled block
        wave_num_threshold : float, threshold that determines the skewness of the luminance value -> wave number relationship
        max_amplitude : float, maximum amplitude of waves in for each pixel/downsampled block
        amp_threshold : float, threshold for amplitude that determines the skewness of the luminance value -> amplitude relationship
        resolution : int, number of points to plot for each pixel/downsampled block. The higher the number, the more accurate the squiggles will be, but the longer it will take to calculate.

        Assigns self.squiggles : numpy array, squiggles in the image
        """

        if self.image.shape[0] > 1000:
            warnings.warn(f"Warning: This image is very large. It may take a while to calculate the squiggles.\n There are {self.image.shape[0]} squiggle lines to calculate. Try increasing the vertical downsample_amount or decreasing the resolution to speed up the calculation.")

        if self.image.shape[1] > 10_000:
            warnings.warn(f"Warning: This image is very large. It may take a while to calculate the squiggles.\n There are {self.image.shape[1]} points per line to calculate. Try increasing the horizontal downsample_amount or decreasing the resolution to speed up the calculation.")


        wave_num = (self.image**(1/wave_num_threshold) * max_wave_num).astype(int)
        phase_shift = np.apply_along_axis(self.phase_reconciliation, 1, wave_num)

        amplitude = self.image**(1/amp_threshold) * max_amplitude 
        get_squiggle = lambda x: np.sin(x * wave_num * np.pi + (phase_shift * np.pi)) * amplitude

        x = np.linspace(0, 1, resolution, endpoint=False)
        squigggles_seperated = np.array(list(map(get_squiggle, x)))
        squiggles = squigggles_seperated.swapaxes(0, 1).reshape(squigggles_seperated.shape[1], -1, order='F')

        self.squiggles = squiggles

    def render_squiggles(self, 
                width_in_inches = 8.5,
                height_in_inches = 11,
                border_in_inches = .5,
                line_width = .2,
                rgb=(0, 0, 0),
                ):
        """
        Function to render the already calculated squiggles.

        Inputs
        ------
        width_in_inches : float, width of image in inches (this is the width of the canvas, not neccessarily the width of the drawing)
        height_in_inches : float, height of image in inches (this is the height of the canvas, not neccessarily the height of the drawing)
        border_in_inches : float, the minimum border around drawing in inches. The actual border can be larger depending on the aspect ratio of the image versus the width_in_inches, height_in_inches passed. 
        line_width : float, width of lines in image
        rgb : tuple, RGB color of lines

        """

        if self.squiggles is None:
            raise ValueError("Must calculate squiggles first! Call calculate_squiggles()")

        width_in_points  = width_in_inches * 72.
        height_in_points  = height_in_inches * 72.
        border_in_points = border_in_inches * 72.

        usuable_width = width_in_points - 2 * border_in_points 
        usuable_height = height_in_points - 2 * border_in_points

        if self.aspect_ratio < 1:

            drawing_height = usuable_height
            drawing_width = self.aspect_ratio * drawing_height

        else:

            drawing_width = usuable_width
            drawing_height = drawing_width / self.aspect_ratio

        effective_horizontal_border = (width_in_points - drawing_width) / 2
        effective_vertical_border = (height_in_points - drawing_height) / 2 

        line_spacing = drawing_height / self.squiggles.shape[0]
        vertical_offset = (np.ones(self.squiggles.shape) * line_spacing).cumsum(axis=0) - line_spacing
        vertical_offset += effective_vertical_border
        squiggles_offset = self.squiggles + vertical_offset

        x_vals = np.linspace(effective_horizontal_border, drawing_width+effective_horizontal_border, self.squiggles.shape[1])


        svgio = BytesIO()
        with cairo.SVGSurface(svgio,
                            width_in_points,
                            height_in_points) as surface:

            ctx = cairo.Context(surface)

            # move to start of first curve
            ctx.move_to(x_vals[0], squiggles_offset[0, 0])

            for line in squiggles_offset:

                start_x = effective_horizontal_border
                start_y = line[0] 
                
                ctx.move_to(start_x, start_y)

                for i in range(len(line)):

                    ctx.line_to(x_vals[i], line[i])

            ctx.set_line_width(line_width)
            ctx.set_source_rgb(*rgb)
            ctx.stroke()

        self.svgio = svgio
            
    def save_squiggle_image(self, fname):
        """
        Save the image to a file. 
        """

        if self.svgio is None: 
            raise ValueError("You must render the squiggles before saving them. Use render_squiggles()")

        with open(fname, "wb") as outfile:
            # Copy the BytesIO stream to the output file
            outfile.write(self.svgio.getbuffer())


    