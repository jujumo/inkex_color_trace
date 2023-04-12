#!/usr/bin/env python
"""Randomise the selected item's colours using hsl colorspace"""

import sys
import base64
import inkex
import io
import PIL.Image
import PIL.ImageDraw
import PIL.ImagePath
import numpy as np
from inkex.transforms import Transform
from inkex.elements import ShapeElement, Image, Group, PathElement
from inkex.paths import Path, CubicSuperPath
from inkex.localization import inkex_gettext as _


class Trace(inkex.EffectExtension):
    """Randomize the colours of all objects"""

    def add_arguments(self, pars):
        pars.add_argument("--tab")
        pars.add_argument("--transparency", type=bool, default=0, help="transparency")

    @staticmethod
    def _read_image(img_elt):
        ''' Read an image from either the SVG file itself or an external file. '''
        # Read the image from an external file.
        fname = img_elt.get('sodipodi:absref')
        if fname != None:
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
            inkex.errormsg('failed to parse embedded image data')
            sys.exit(1)
        if dtype != 'base64':
            inkex.errormsg('embedded image is encoded as %s, but this plugin supports only base64' % dtype)
            sys.exit(1)
        raw_data = base64.decodebytes(data64.encode('utf-8'))
        return PIL.Image.open(io.BytesIO(raw_data))

    def _get_image(self):
        ''' Return the image. '''
        img_node = next(iter(self.svg.selection.get(Image)))
        img = self._read_image(img_node).convert('RGBA')
        return img_node, img

    def _get_objects(self):
        ''' Return list of objects. '''
        selection = self.svg.selection.get(ShapeElement)
        objs = [e for e in selection if e.TAG == 'path']

        # if img or len(objs) == 0:
        #     inkex.utils.errormsg(
        #         _('Trace requires that one image and at least one additional object be selected.'))
        return objs

    def effect(self):
        selection = self.svg.selection
        if not selection:
            self.msg(_('No selection'))
            self.svg.selection.set(self.document.getroot())

        shapes = self._get_objects()
        image_node, image = self._get_image()

        matrix_image_node_from_world, matrix_image_from_image_node = np.identity(3), np.identity(3)
        matrix_image_node_from_world[0:2, :] = np.array((-image_node.composed_transform()).matrix)
        matrix_image_from_image_node[0:2, 2] = -np.array([image_node.left, image_node.top])
        matrix_pixels_from_u = np.diag([image.width / image_node.width, image.height / image_node.height, 1])
        matrix_pixel_from_world = matrix_pixels_from_u @ matrix_image_from_image_node @ matrix_image_node_from_world

        image_black = PIL.Image.new(mode="RGB", size=image.size, color=(0, 0, 0))
        image_debug = image.copy()
        mask = PIL.Image.new("L", image.size, 0)
        mask_canvas = PIL.ImageDraw.Draw(mask)
        debug_canvas = PIL.ImageDraw.Draw(image_debug) if image_debug else None

        matrix_world_from_shape = np.identity(3)
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
                mask_canvas.polygon(polygon, fill="#ffffff")
                if debug_canvas:
                    debug_canvas.line(polygon, fill="red", width=10)
                    # debug_canvas.polygon(polygon, fill="#ffffff")
                image_composed = PIL.Image.composite(image, image_black, mask)
                # image_composed.show()

            mask_size = np.count_nonzero(np.asarray(mask))
            if mask_size != 0:
                color_average = np.sum(np.asarray(image_composed), axis=(0, 1)) / mask_size
                color_average = color_average.astype(int)
                color_print = f'#{color_average[0]:02x}{color_average[1]:02x}{color_average[2]:02x}'
                shape.style['fill'] = color_print
                shape.path = path

        if image_debug:
            image_debug.show()


if __name__ == "__main__":
    Trace().run()
