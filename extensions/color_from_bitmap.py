#! /usr/bin/env python
# -*- coding: utf-8 -*-
"""
Grab color from underlying bitmap and apply it to path objects.
The color is averaged all over the path area. Erode parameter can be used
to shrink or expand (using negative value) this area.
If multiple bitmaps are selected, only one is considered.

version: 1.0

Known limitations:
- Only consider a single image.
- Only handle path objects: does not work on rectangles, circles, ... Make sure to convert objects to paths.
- Only works with straight paths: bezier area are approximated using straight lines.
- Dilatation (negative erosion) have weird corners.
- Ignore any clipping on the image.
- slow
"""


import sys
import base64
import inkex
import io
import os.path as path
from typing import List, Tuple, Optional
import PIL.Image
import PIL.ImageDraw
import PIL.ImagePath
import numpy as np
from inkex.elements import ShapeElement, Image
from inkex.localization import inkex_gettext as _


def srgb_to_linear(srgb):
    # Convert a single sRGB value to linear RGB
    if srgb <= 0.04045:
        return srgb / 12.92
    else:
        return ((srgb + 0.055) / 1.055) ** 2.4


class ColorFromBitmapException(Exception):
    pass


class ColorFromBitmap(inkex.EffectExtension):
    """Grab color from underlying bitmap and apply it to paths."""

    def add_arguments(self, pars):
        pars.add_argument("--tab", dest='tab',)
        pars.add_argument("--erode", type=int, default=0, help="erode path (using stroke)")
        pars.add_argument("--fill", choices=['unchanged', 'none', 'average'], default='average',
                          help="apply to fill color")
        pars.add_argument("--stroke", choices=['unchanged', 'none', 'average'], default='unchanged',
                          help="apply to stroke color")
        pars.add_argument("--scale_x_min", type=float, default=1.0,
                          help="scale factor for minimum intensity on horizontal axis.")
        pars.add_argument("--scale_x_max", type=float, default=1.0,
                          help="scale factor for maximum intensity on horizontal axis.")
        pars.add_argument("--scale_y_min", type=float, default=1.0,
                          help="scale factor for minimum intensity on vertical axis.")
        pars.add_argument("--scale_y_max", type=float, default=1.0,
                          help="scale factor for maximum intensity on vertical axis")
        pars.add_argument("--intensity", choices=['sRGB', 'linear'], default='sRGB',
                          help="which method used to compute light intensity. "
                               "sRGB takes raw RGB values as is.")
        pars.add_argument("--show_debug", type=inkex.Boolean, default=False,
                          help="also output debug images")
        pars.add_argument("--auto_select", type=inkex.Boolean, default=False,
                          help="auto select all if no selection.")

    def _load_image(self, image_node) -> PIL.Image:
        """
        Load image to PIL object from either the SVG file itself or an external file.
        inspired from: https://inkscape.org/~pakin/%E2%98%85pixels-to-objects   -- by Scott Pakin
        first try to load embedded image, if fails, try to load file.
        """
        # ignore sodipodi:absref
        image = None
        xlink = image_node.get('xlink:href')

        # first try embedded
        if image is None and xlink.startswith('data:'):
            try:
                _, dtype_data = xlink[5:].split(';', 1)
                dtype, data64 = dtype_data.split(',', 1)
            except ValueError:
                raise ColorFromBitmapException('failed to parse embedded image data')
            if dtype != 'base64':
                raise ColorFromBitmapException(f'embedded image is encoded as {dtype}, '
                                               f'but this plugin supports only base64')
            raw_data = base64.decodebytes(data64.encode('utf-8'))
            image = PIL.Image.open(io.BytesIO(raw_data))

        # if unsuccessful try to load as a file
        if image is None:
            image_filepath = self.absolute_href(xlink)  # expect a relative path in xlink, make it absolute.
            try:
                image = PIL.Image.open(image_filepath)
            except FileNotFoundError:
                raise ColorFromBitmapException(f'Image file not found : {image_filepath}')
            except PIL.UnidentifiedImageError:
                raise ColorFromBitmapException(f'Error while opening image file {image_filepath}')

        # none of the previous managed to load image
        if image is None:
            raise ColorFromBitmapException('unable to load image neither from file or base64.')

        return image

    def _get_image(self) -> (Image, PIL.Image):
        """
        Return both the svg image node and the image bitmap.
        """
        image_node = next(iter(self.svg.selection.get(Image)), None)
        if image_node is None:
            raise ColorFromBitmapException('unable to find image bitmap in selection.')
        image = self._load_image(image_node).convert('RGBA')
        return image_node, image

    def _get_shapes(self) -> List[ShapeElement]:
        """
        Return list of ShapeElement.
        """
        selection = self.svg.selection.get(ShapeElement)
        paths = [e for e in selection if e.TAG == 'path']
        return paths

    def effect(self):
        """
        the effect
        select one image and all path
        for all path:
         Raster/Render the path using polygon in binary mask image.
         Use outline to erode/dilate if needed.
         Then use that mask to select pixel of the image that fall into the path.
         Average those pixels and apply this computed color to the path.
        """
        try:
            if not self.svg.selection:
                self.msg(_('Nothing is selected: applying to all.'))
                if self.options.auto_select:
                    self.svg.selection.set(self.document.getroot())

            image_node, image = self._get_image()
            shapes = self._get_shapes()

            if not shapes:
                raise ColorFromBitmapException('No path selected.')

            matrix_image_node_from_world, matrix_image_from_image_node = np.identity(3), np.identity(3)
            matrix_image_node_from_world[0:2, :] = np.array((-image_node.composed_transform()).matrix)
            matrix_image_from_image_node[0:2, 2] = -np.array([image_node.left, image_node.top])
            matrix_pixel_from_u = np.diag([image.width / image_node.width, image.height / image_node.height, 1])
            matrix_pixel_from_world = matrix_pixel_from_u @ matrix_image_from_image_node @ matrix_image_node_from_world

            image_black = PIL.Image.new(mode="RGB", size=image.size, color=(0, 0, 0))
            image_debug = image.copy() if self.options.show_debug else None
            mask = PIL.Image.new("L", image.size, 0)
            mask_canvas = PIL.ImageDraw.Draw(mask)
            debug_canvas = PIL.ImageDraw.Draw(image_debug) if image_debug else None

            matrix_world_from_shape = np.identity(3)

            # eroding raster shape using stroke 2x width
            eroding_stroke_style = {
                'stroke-width': int(abs(self.options.erode) * 2),
                'stroke': 'black'
            }
            if self.options.erode < 0:
                # negative erode ==  dilating, uses white stroke
                eroding_stroke_style['stroke'] = 'white'

            scale_change = any(s != 1 for s in [self.options.scale_x_min,
                                                self.options.scale_x_max,
                                                self.options.scale_y_min,
                                                self.options.scale_y_max])
            scale_x_range = self.options.scale_x_max - self.options.scale_x_min
            scale_y_range = self.options.scale_y_max - self.options.scale_y_min

            for shape in shapes:
                # print(f'SHAPE::{shape.eid:10} :: {shape.TAG:10} // pos=[{shape_bb.center}] @ {shape_bb.size}')
                matrix_world_from_shape[0:2, :] = shape.composed_transform().matrix
                matrix_pixel_from_shape = matrix_pixel_from_world @ matrix_world_from_shape
                path = shape.path.to_superpath()
                # reset mask to None
                mask_canvas.rectangle(((0, 0), mask.size), fill='black')
                for i, subpath in enumerate(path):
                    # retrieve the path polygon coordinates
                    points_shape_homogen = np.ones((3, len(subpath)))
                    points_shape_homogen[0:2, :] = np.array(subpath)[:, 1, :].transpose()
                    # convert path coordinates to pixel coordinates into the image
                    points_shape_homogen = matrix_pixel_from_shape @ points_shape_homogen
                    # draw the path area into the mask
                    polygon = [(p[0], p[1])
                               for p in points_shape_homogen.transpose()]
                    mask_canvas.polygon(polygon, fill='#ffffff')
                    # handle erosion
                    if self.options.erode > 0:
                        # outline only draw inside the polygon
                        mask_canvas.polygon(polygon, fill=None, outline='#000000', width=self.options.erode)
                    # handle dilatation (!erosion)
                    if self.options.erode < 0:
                        # lines draw over the polygon, but corners are cut
                        mask_canvas.line(polygon, fill='#ffffff', width=abs(self.options.erode) * 2)
                    # mask.show()
                    # select the image pixels
                    image_composed = PIL.Image.composite(image, image_black, mask)
                    # image_composed.show()

                # early stop if no pixels to average
                mask_size = np.count_nonzero(np.asarray(mask))
                if mask_size == 0:
                    continue
                # compute color average
                color_average = np.sum(np.asarray(image_composed), axis=(0, 1)) / mask_size
                color_average = color_average.astype(int)
                # apply color to the path
                color_print = f'#{color_average[0]:02x}{color_average[1]:02x}{color_average[2]:02x}'
                if self.options.fill == 'average':
                    shape.style['fill'] = color_print
                if self.options.fill == 'none':
                    shape.style['fill'] = None
                if self.options.stroke == 'average':
                    shape.style['stroke'] = color_print
                if self.options.stroke == 'none':
                    shape.style['stroke'] = None

                if scale_change:
                    light_intensity = np.dot(color_average, np.array([0.299, 0.587, 0.114])) / 255
                    if self.options.intensity == 'linear':
                        # sRGB values are not proportional to light emission.
                        # activate linear option to have a linear light intensity value.
                        # Useful to make physical masks.
                        light_intensity = srgb_to_linear(light_intensity)
                    # relation between intensity and scale:
                    # 0..255 intensity value should match scale_min..scale_max value
                    scale_x = self.options.scale_x_min + light_intensity * scale_x_range
                    scale_y = self.options.scale_y_min + light_intensity * scale_y_range
                    bounding_box = shape.bounding_box().center
                    matrix_shape_from_center = np.identity(3)
                    matrix_shape_from_center[0:2, 2] = bounding_box
                    matrix_scale = np.diag([scale_x, scale_y, 1.0])
                    matrix_scale_from_shape = matrix_shape_from_center @ matrix_scale @ np.linalg.inv(matrix_shape_from_center)
                    scale_transform = inkex.Transform(matrix_scale_from_shape[0:2, :].tolist())
                    shape.update(transform=scale_transform)

                if debug_canvas:
                    stroke_color = None
                    if self.options.erode < 0:
                        stroke_color = color_print
                    debug_canvas.polygon(polygon,
                                         fill=color_print,
                                         outline=stroke_color,
                                         width=abs(self.options.erode)*2)

            if image_debug:
                image_debug.show()

            image.close()
            image_black.close()
            image_composed.close()
            if image_debug:
                image_debug.close()
        except ColorFromBitmapException as e:
            inkex.errormsg(e)
            sys.exit(1)
        except Exception as e:
            inkex.errormsg(str(e))
            sys.exit(-1)


if __name__ == "__main__":
    ColorFromBitmap().run()
