import ctypes
import pyglet
from pywavefront import visualization
from pywavefront import Wavefront

class MeshViewer(pyglet.window.Window):
    def __init__(self, filename=None):
        self.rotation = 0.0

        super(MeshViewer, self).__init__(1024, 720, caption='Mesh Viewer', resizable=True)
        self.meshes = Wavefront(filename)
        pyglet.gl.glClearColor(1, 1, 1, 1)  # Note that these are values 0.0 - 1.0 and not (0-255).
        self.lightfv = ctypes.c_float * 4

    def on_resize(self, width, height):
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        pyglet.gl.gluPerspective(0.0, float(width) / height, 1.0, 100.0)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
        return True

    def on_draw(self):
        self.clear()
        pyglet.gl.glLoadIdentity()

        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_POSITION, self.lightfv(-40.0, 200.0, 100.0, 0.0))
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_AMBIENT, self.lightfv(0.2, 0.2, 0.2, 1.0))
        pyglet.gl.glLightfv(pyglet.gl.GL_LIGHT0, pyglet.gl.GL_DIFFUSE, self.lightfv(0.5, 0.5, 0.5, 1.0))
        pyglet.gl.glEnable(pyglet.gl.GL_LIGHT0)
        pyglet.gl.glEnable(pyglet.gl.GL_LIGHTING)

        pyglet.gl.glEnable(pyglet.gl.GL_COLOR_MATERIAL)
        pyglet.gl.glEnable(pyglet.gl.GL_DEPTH_TEST)
        pyglet.gl.glShadeModel(pyglet.gl.GL_SMOOTH)

        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)

        # pyglet.gl.glTranslated(0, .8, -20)
        pyglet.gl.glRotatef(-90, 0, 0, 1)
        pyglet.gl.glRotatef(self.rotation, 1, 0, 0)
        pyglet.gl.glRotatef(90, 0, 0, 1)
        pyglet.gl.glRotatef(0, 0, 1, 0)

        visualization.draw(self.meshes)

    def update(self, dt):
        self.rotation += 45 * dt
        if self.rotation > 720.0:
            self.rotation = 0.0

    def show_window(self):
        self.set_visible(visible=True)


if __name__ == "__main__":
    # window = pyglet.window.Window(1024, 720, caption='Demo', resizable=True)
    window = MeshViewer('.\\output\\0.obj')
    pyglet.clock.schedule(window.update)
    pyglet.app.run()
    window.show_window()
