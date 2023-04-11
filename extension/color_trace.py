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
from inkex.elements import ShapeElement, Image, Group
from inkex.paths import Path, CubicSuperPath
from inkex.localization import inkex_gettext as _
import math

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
            self.svg.selection.set(self.document.getroot())

        objs = self._get_objects()
        img_node, image = self._get_image()
        image_bb = img_node.bounding_box()
        image_black = PIL.Image.new(mode="RGB", size=image.size, color=(0, 0, 0))

        mask = PIL.Image.new("L", image.size, 0)
        mask_canvas = PIL.ImageDraw.Draw(mask)
        scale = np.array([
            image.width / image_bb.width,
            image.height / image_bb.height
        ])
        translate = np.array([
            -image_bb.top,
            -image_bb.left
        ])

        # TODO: RASTER Path to image mask
        for shape in objs:
            shape.apply_transform()
            shape_bb = shape.bounding_box()
            path = shape.path.to_superpath()
            # print(f'SHAPE::{shape.eid:10} :: {shape.TAG:10} // pos=[{shape_bb.center}] @ {shape_bb.size}')
            mask_canvas.rectangle([(0, 0), mask.size], fill='black')
            for i, subpath in enumerate(path):
                xys = np.array(subpath)[:, 1, :]
                xys = (xys +translate) * scale
                polygon = [tuple(p) for p in xys]
                mask_canvas.polygon(polygon, fill="#ffffff", outline="black")
                image_composed = PIL.Image.composite(image, image_black, mask)
                # image_composed.show()
            color_average = np.sum(np.asarray(image_composed), axis=(0, 1)) / np.count_nonzero(np.asarray(mask))
            color_average = color_average.astype(int)
            color_print = f'#{color_average[0]:02x}{color_average[1]:02x}{color_average[2]:02x}'
            # print(color_print)
            shape.style['fill'] = color_print
            shape.path = path


if __name__ == "__main__":
    if True:
        input_file = r'/local/devel/inkex_color_trace/sample/drawing.svg'
        output_file = r'/local/devel/inkex_color_trace/sample/result.svg'
        Trace().run([input_file, '--output=' + output_file])
    else:
        Trace().run()
