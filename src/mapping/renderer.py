import multiprocessing as mp
import sys
import numpy as np


class SlamRenderer:
    """Launches an OpenGL 3D viewer in a separate process.

    Follows the same process+queue pattern as BagImageStreamer.
    The VisualOdometry pipeline pushes {pose, points} dicts into the queue;
    the renderer process drains them and renders trajectory + point cloud.
    """

    def __init__(self):
        self.queue = mp.Queue(maxsize=100)
        self.stop_event = mp.Event()
        self.process = mp.Process(target=_renderer_process_entry,
                                  args=(self.queue, self.stop_event))

    def get_queue(self):
        return self.queue

    def start(self):
        self.process.start()

    def stop(self):
        self.stop_event.set()
        self.process.join(timeout=3)
        if self.process.is_alive():
            self.process.terminate()


# ---------------------------------------------------------------------------
# Everything below runs in the child process — OpenGL is only imported here.
# ---------------------------------------------------------------------------

def _renderer_process_entry(queue, stop_event):
    from OpenGL.GL import (
        glClear, glClearColor, glEnable, glPointSize, glLineWidth,
        glBegin, glEnd, glVertex3f, glColor3f, glLoadIdentity,
        glMatrixMode, glViewport, glOrtho,
        GL_COLOR_BUFFER_BIT, GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST,
        GL_LINES, GL_LINE_STRIP, GL_POINTS,
        GL_PROJECTION, GL_MODELVIEW,
        glTranslatef, glRotatef, glScalef,
    )
    from OpenGL.GLUT import (
        glutInit, glutInitDisplayMode, glutInitWindowSize,
        glutCreateWindow, glutDisplayFunc, glutIdleFunc,
        glutReshapeFunc, glutKeyboardFunc, glutMouseFunc,
        glutMotionFunc, glutMainLoop, glutPostRedisplay,
        glutSwapBuffers, glutDestroyWindow, glutGetWindow,
        GLUT_DOUBLE, GLUT_RGB, GLUT_DEPTH,
        GLUT_LEFT_BUTTON, GLUT_RIGHT_BUTTON, GLUT_DOWN,
    )
    from OpenGL.GLU import gluPerspective

    # ---- state ----
    poses = []
    all_points = []
    cam_dist = 20.0
    cam_rot_x = 30.0
    cam_rot_y = 45.0
    mouse_last = [0, 0]
    mouse_button = [False, False]  # left, right
    cam_target = np.zeros(3)

    def drain_queue():
        nonlocal cam_target
        while not queue.empty():
            try:
                data = queue.get_nowait()
            except Exception:
                break
            poses.append(data['pose'].copy())
            pts = data['points']
            if len(pts) > 0:
                all_points.extend(pts.tolist())
            # auto-follow: track latest camera position
            cam_target = poses[-1][:3, 3].copy()

    def draw_axes(length=2.0):
        glLineWidth(2.0)
        glBegin(GL_LINES)
        # X - red
        glColor3f(1, 0, 0)
        glVertex3f(0, 0, 0); glVertex3f(length, 0, 0)
        # Y - green
        glColor3f(0, 1, 0)
        glVertex3f(0, 0, 0); glVertex3f(0, length, 0)
        # Z - blue
        glColor3f(0, 0, 1)
        glVertex3f(0, 0, 0); glVertex3f(0, 0, length)
        glEnd()

    def draw_frustum(pose, scale=0.5, color=(0.5, 0.5, 0.5)):
        """Draw a wireframe camera frustum at the given 4x4 pose."""
        # frustum corners in camera frame
        corners = np.array([
            [-1, -0.75, 1],
            [ 1, -0.75, 1],
            [ 1,  0.75, 1],
            [-1,  0.75, 1],
        ]) * scale
        origin = np.array([0, 0, 0])

        R = pose[:3, :3]
        t = pose[:3, 3]
        world_corners = (R @ corners.T).T + t
        world_origin = t

        glColor3f(*color)
        glBegin(GL_LINES)
        # rays from origin to corners
        for c in world_corners:
            glVertex3f(*world_origin)
            glVertex3f(*c)
        # rectangle connecting corners
        for i in range(4):
            j = (i + 1) % 4
            glVertex3f(*world_corners[i])
            glVertex3f(*world_corners[j])
        glEnd()

    def display():
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # orbit camera
        glTranslatef(0, 0, -cam_dist)
        glRotatef(cam_rot_x, 1, 0, 0)
        glRotatef(cam_rot_y, 0, 1, 0)
        glTranslatef(-cam_target[0], -cam_target[1], -cam_target[2])

        draw_axes()

        # trajectory
        if len(poses) > 1:
            glColor3f(0.2, 0.5, 1.0)
            glLineWidth(2.0)
            glBegin(GL_LINE_STRIP)
            for p in poses:
                pos = p[:3, 3]
                glVertex3f(*pos)
            glEnd()

        # frustums: every 10th historical (gray) + current (green)
        for i, p in enumerate(poses):
            if i == len(poses) - 1:
                draw_frustum(p, scale=0.5, color=(0.0, 1.0, 0.0))
            elif i % 10 == 0:
                draw_frustum(p, scale=0.3, color=(0.4, 0.4, 0.4))

        # point cloud
        if all_points:
            glPointSize(2.0)
            glColor3f(1.0, 1.0, 1.0)
            glBegin(GL_POINTS)
            for pt in all_points:
                glVertex3f(*pt)
            glEnd()

        glutSwapBuffers()

    def _quit():
        wid = glutGetWindow()
        if wid:
            glutDestroyWindow(wid)
        sys.exit(0)

    def idle():
        if stop_event.is_set():
            _quit()
            return
        drain_queue()
        glutPostRedisplay()

    def reshape(w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(60.0, w / h, 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW)

    def keyboard(key, x, y):
        nonlocal cam_dist
        if key in (b'q', b'\x1b'):  # q or ESC
            _quit()
        elif key == b'+' or key == b'=':
            cam_dist = max(1.0, cam_dist - 2.0)
        elif key == b'-':
            cam_dist += 2.0

    def mouse(button, state, x, y):
        mouse_last[0] = x
        mouse_last[1] = y
        if state == GLUT_DOWN:
            if button == GLUT_LEFT_BUTTON:
                mouse_button[0] = True
            elif button == GLUT_RIGHT_BUTTON:
                mouse_button[1] = True
        else:
            mouse_button[0] = False
            mouse_button[1] = False

    def motion(x, y):
        nonlocal cam_rot_x, cam_rot_y, cam_dist
        dx = x - mouse_last[0]
        dy = y - mouse_last[1]
        if mouse_button[0]:  # left drag → rotate
            cam_rot_y += dx * 0.5
            cam_rot_x += dy * 0.5
        if mouse_button[1]:  # right drag → zoom
            cam_dist = max(1.0, cam_dist + dy * 0.1)
        mouse_last[0] = x
        mouse_last[1] = y

    # ---- GLUT init ----
    glutInit()
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
    glutInitWindowSize(800, 600)
    glutCreateWindow(b"SLAM 3D Viewer")
    glClearColor(0.05, 0.05, 0.1, 1.0)
    glEnable(GL_DEPTH_TEST)

    glutDisplayFunc(display)
    glutIdleFunc(idle)
    glutReshapeFunc(reshape)
    glutKeyboardFunc(keyboard)
    glutMouseFunc(mouse)
    glutMotionFunc(motion)

    glutMainLoop()
