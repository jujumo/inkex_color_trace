#!/usr/bin/env python
"""Randomise the selected item's colours using hsl colorspace"""

import sys
import base64
import inkex
import io
import PIL.Image
import numpy as np
from inkex.transforms import Transform
from inkex.elements import ShapeElement, Image, Group
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
            self.svg.selection.set(self.document.getroot())

        objs = self._get_objects()
        img_node, image = self._get_image()
        image_bb = img_node.bounding_box()

        # img_transform = Transform(scale=(1/self.svg.viewport_width, 1/self.svg.viewport_height))
            #translate=(-image_bb.left, -image_bb.top),
        # TODO: RASTER Path to image mask
        for shape in objs:
            shape_bb = shape.bounding_box()
            path = shape.path.to_superpath()
            print(f'SHAPE::{shape.eid:10} :: {shape.TAG:10} // pos=[{shape_bb.center}] @ {shape_bb.size}')
            for i, subpath in enumerate(path):
                ww = np.array(subpath)
                ww[:, 0, :] = ww[:, 1, :]
                ww[:, 2, :] = ww[:, 1, :]
                print(ww[:, 1, :])
                path[i] = ww.tolist()
            shape.style['fill'] = 'red'
            shape.path = path
            # print(f'SHAPE::{shape.eid:10} :: {shape.TAG:10} // pos=[{shape.x}]')
            # for p in e.path:
            #     print(p)
            # point = shape_bb.center
            # point2 = img_transform.apply_to_point(point)
            # print(f'{point} ==> {point2}')
            # trans = e.composed_transform()
            # point = trans.apply_to_point(point)
            # point = tuple(int(p) for p in point)
            # print(point)
            # color = image.getpixel(point)
            # color = f'#{color[0]:02x}{color[1]:02x}{color[1]:02x}'
            # print(color)
            # e.style['fill'] = color

        # image.show()


if __name__ == "__main__":
    if True:
        input_file = r'D:\devel\inkex_color_trace\sample\drawing.svg'
        output_file = r'D:\devel\inkex_color_trace\sample\result.svg'
        Trace().run([input_file, '--output=' + output_file])
    else:
        Trace().run()