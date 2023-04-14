#! /usr/bin/env python
# -*- coding: utf-8 -*-
'''
Grab color from underlying bitmap and apply it to paths.
'''


import sys
import base64
import inkex
import io
from typing import List, Tuple, Optional
import PIL.Image
import PIL.ImageDraw
import PIL.ImagePath
import numpy as np
from inkex.elements import ShapeElement, Image
from inkex.localization import inkex_gettext as _


class ColorFromBitmap(inkex.EffectExtension):
    """Grab color from underlying bitmap and apply it to paths."""

    def add_arguments(self, pars):
        pars.add_argument("--tab")
        pars.add_argument("--debug", type=bool, default=0, help="also output debug images")
        pars.add_argument("--erode", type=int, default=0, help="erode path (using stroke)")

    @staticmethod
    def _read_image(img_elt) -> PIL.Image:
        '''
        Read an image from either the SVG file itself or an external file.
        remixed from: https://inkscape.org/~pakin/%E2%98%85pixels-to-objects
                        -- by Scott Pakin
        '''
        # Read the image from an external file.
        fname = img_elt.get('sodipodi:absref')
        if fname is not None:
            # Fully qualified filename.  Read it directly.
            return PIL.Image.open(fname)
        xlink = img_elt.get('xlink:href')
        if not xlink.startswith('data:'):
            # Unqualified filename.  Try reading it directly although there's a
            # good chance this will fail.
            return PIL.Image.open(fname)
        # Read an image embedded in the SVG file itself.
        try:
            mime, dtype_data = xlink[5:].split(';', 1)
            dtype, data64 = dtype_data.split(',', 1)
        except ValueError:
            raise ValueError('failed to parse embedded image data')
        if dtype != 'base64':
            raise ValueError(f'embedded image is encoded as {dtype}, but this plugin supports only base64')
        raw_data = base64.decodebytes(data64.encode('utf-8'))
        return PIL.Image.open(io.BytesIO(raw_data))

    def _get_image(self) -> (Image, PIL.Image):
        '''
        Return both the svg image node and the image bitmap.
        '''
        image_node = next(iter(self.svg.selection.get(Image)))
        if image_node is None:
            raise ValueError('unable to find image bitmap in selection.')
        image = self._read_image(image_node).convert('RGBA')
        return image_node, image

    def _get_shapes(self) -> List[ShapeElement]:
        """
        Return list of ShapeElement.
        """
        selection = self.svg.selection.get(ShapeElement)
        paths = [e for e in selection if e.TAG == 'path']
        return paths

    def effect(self):
        try:
            selection = self.svg.selection
            if not selection:
                self.msg(_('Nothing is selected: applying to all.'))
                self.svg.selection.set(self.document.getroot())

            image_node, image = self._get_image()
            shapes = self._get_shapes()
            if not shapes:
                raise ValueError("No path selected.")

            matrix_image_node_from_world, matrix_image_from_image_node = np.identity(3), np.identity(3)
            matrix_image_node_from_world[0:2, :] = np.array((-image_node.composed_transform()).matrix)
            matrix_image_from_image_node[0:2, 2] = -np.array([image_node.left, image_node.top])
            matrix_pixels_from_u = np.diag([image.width / image_node.width, image.height / image_node.height, 1])
            matrix_pixel_from_world = matrix_pixels_from_u @ matrix_image_from_image_node @ matrix_image_node_from_world

            image_black = PIL.Image.new(mode="RGB", size=image.size, color=(0, 0, 0))
            image_debug = image.copy() if self.options.debug else None
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

            for shape in shapes:
                # print(f'SHAPE::{shape.eid:10} :: {shape.TAG:10} // pos=[{shape_bb.center}] @ {shape_bb.size}')
                matrix_world_from_shape[0:2, :] = shape.composed_transform().matrix
                matrix_pixel_from_shape = matrix_pixel_from_world @ matrix_world_from_shape
                path = shape.path.to_superpath()
                mask_canvas.rectangle([(0, 0), mask.size], fill='black')
                for i, subpath in enumerate(path):
                    points_shape_homogen = np.ones((3, len(subpath)))
                    points_shape_homogen[0:2, :] = np.array(subpath)[:, 1, :].transpose()
                    points_shape_homogen = matrix_pixel_from_shape @ points_shape_homogen
                    polygon = [(p[0], p[1])
                               for p in points_shape_homogen.transpose()]
                    if self.options.erode != 0:
                        stroke_color = '#000000' if self.options.erode > 0 else '#ffffff'
                        mask_canvas.line(polygon, fill=stroke_color, width=abs(self.options.erode) * 2)
                    mask_canvas.polygon(polygon, fill="#ffffff")

                    image_composed = PIL.Image.composite(image, image_black, mask)
                    mask.show()
                    # image_composed.show()

                mask_size = np.count_nonzero(np.asarray(mask))
                if mask_size == 0:
                    continue

                color_average = np.sum(np.asarray(image_composed), axis=(0, 1)) / mask_size
                color_average = color_average.astype(int)
                color_print = f'#{color_average[0]:02x}{color_average[1]:02x}{color_average[2]:02x}'
                shape.style['fill'] = color_print
                # shape.style.update(eroding_stroke_style)
                if debug_canvas:
                    debug_canvas.polygon(polygon, fill=color_print)
                    # if self.options.erode > 0:
                    #     debug_canvas.line(polygon, fill="black", width=self.options.erode*2)

            if image_debug:
                image_debug.show()

            image.close()
            image_black.close()
            image_composed.close()
            if image_debug:
                image_debug.close()
        except ValueError as e:
            inkex.errormsg(e)
            sys.exit(-1)


if __name__ == "__main__":
    ColorFromBitmap().run()
