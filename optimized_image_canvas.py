from PIL import Image, ImageTk

class OptimizedImageCanvas:
    def __init__(self, canvas):
        self.canvas = canvas
        self.image_cache = {}  # Cache for storing downsampled images
        self.max_cache_size = 5  # Maximum number of cached images
        self.current_scale = 1.0
        self.min_scale_for_full_res = 0.5  # Minimum scale at which to show full resolution

    def clear_cache(self):
        """Clear the image cache"""
        self.image_cache.clear()

    def get_downsampled_image(self, original_image, target_scale):
        """Get a downsampled version of the image appropriate for the current zoom level"""
        if target_scale >= self.min_scale_for_full_res:
            return original_image

        # Round scale to nearest 0.1 to prevent too many cached versions
        cache_scale = round(target_scale * 10) / 10

        if cache_scale in self.image_cache:
            return self.image_cache[cache_scale]

        # Calculate new dimensions
        new_width = int(original_image.width * cache_scale)
        new_height = int(original_image.height * cache_scale)

        # Create downsampled version
        downsampled = original_image.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Manage cache size
        if len(self.image_cache) >= self.max_cache_size:
            oldest_scale = list(self.image_cache.keys())[0]
            del self.image_cache[oldest_scale]

        self.image_cache[cache_scale] = downsampled
        return downsampled

    def show_image(self, original_image, imscale, bbox1, bbox2):
        """Show image on the Canvas with dynamic downsampling"""
        bbox = [min(bbox1[0], bbox2[0]), min(bbox1[1], bbox2[1]),
                max(bbox1[2], bbox2[2]), max(bbox1[3], bbox2[3])]

        # Get visible area coordinates
        x1 = max(bbox2[0] - bbox1[0], 0)
        y1 = max(bbox2[1] - bbox1[1], 0)
        x2 = min(bbox2[2], bbox1[2]) - bbox1[0]
        y2 = min(bbox2[3], bbox1[3]) - bbox1[1]

        if int(x2 - x1) > 0 and int(y2 - y1) > 0:
            # Get appropriate image based on scale
            display_image = self.get_downsampled_image(original_image, imscale)

            # Calculate source coordinates in the downsampled image
            scale_factor = display_image.width / original_image.width
            src_x1 = int(x1 / imscale * scale_factor)
            src_y1 = int(y1 / imscale * scale_factor)
            src_x2 = min(int(x2 / imscale * scale_factor), display_image.width)
            src_y2 = min(int(y2 / imscale * scale_factor), display_image.height)

            # Crop and resize the region
            image = display_image.crop((src_x1, src_y1, src_x2, src_y2))
            image = image.resize((int(x2 - x1), int(y2 - y1)), Image.Resampling.NEAREST)

            # Convert to PhotoImage and display
            imagetk = ImageTk.PhotoImage(image)
            imageid = self.canvas.create_image(
                max(bbox2[0], bbox1[0]),
                max(bbox2[1], bbox1[1]),
                anchor='nw',
                image=imagetk
            )
            self.canvas.lower(imageid)
            self.canvas.imagetk = imagetk 