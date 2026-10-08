"""Recognising a hand-drawn digit -- a linear-algebra demo made with ManimGL.
 
    manimgl digit_scene.py Presentation -f        # Scene 1 (no hidden layers), then Scene 2 (two hidden layers)
    manimgl digit_scene.py LinearScene -f         # Scene 1 only
    manimgl digit_scene.py HiddenLayersScene -f   # Scene 2 only (you draw a digit first)
 
    mouse   draw on the 28x28 canvas (right button erases)      Enter   go
    C       clear the canvas                                    N       new digit
    Space   next step (right arrow works too)                   Esc / Ctrl+Q   quit
 
Scene 1 (no hidden layers), one Space per step:
    draw -> divide into squares -> flatten into x (784) -> the squares become the 784 neurons
    -> W x + b written out with real numbers -> 784 dimensions to 10 dimensions
    -> the network -> the 10 weight maps (one per Space).
Scene 2 (two hidden layers): the same digit through 784 -> 32 -> 16 -> 10.
 
All the maths lives in network.py (plain NumPy).  This file only draws it.
"""
import sys
import tempfile
from pathlib import Path
 
sys.path.insert(0, str(Path(__file__).resolve().parent))      # so `import network` works from any folder
 
from manimlib import *
from PIL import Image
from pyglet.window import key as KEY
from pyglet.window import mouse as MOUSE
 
from network import H1, H2, centre_shift, hidden_forward, linear_scores, load_weights, shift_image, softmax
 
# ----------------------------------------------------------------------------- look & layout
CELL = 0.21                                  # one drawing pixel on screen (28 x 0.21 = 5.9 units tall)
BRUSH_CORE, BRUSH_EDGE = 0.8, 1.9            # brush radius in pixels: solid inside CORE, fading out by EDGE
SMALL_CELL = 0.058                           # one pixel once the image sits in the top-left corner
IMAGE_POS = np.array([-6.0, 2.85, 0.0])
IN_X, OUT_X = -4.0, 4.1                      # x of the input column / output column
HIDDEN_X = (-1.3, 1.4)                       # x of the two hidden columns (Scene 2)
STEP, R = 0.4, 0.15                          # gap between neurons, neuron radius
POS, NEG = GREEN_C, RED_C                    # weights: green = positive, red = negative
 
 
def gray(v):
    return rgb_to_hex([float(np.clip(v, 0.0, 1.0))] * 3)
 
 
def neuron(x, y, v=0.0, r=R):
    """A circle whose brightness is its activation (0 = black, 1 = white)."""
    return Circle(radius=r).set_stroke(WHITE, 1.5).set_fill(gray(v), 1.0).move_to([x, y, 0])
 
 
def elided_ys():
    """y of the 16 neurons drawn for a long layer: 8 on top, a gap for the dots, 8 below."""
    top = 3.3 - STEP * np.arange(8)
    return np.concatenate([top, -top[::-1]])
 
 
def vdots(x, y=0.0, gap=0.18, r=0.03):
    return VGroup(*[Dot([x, y + k * gap, 0], radius=r).set_fill(GREY_B) for k in (1, 0, -1)])
 
 
def hdots(x, y=0.0, gap=0.18, r=0.03):
    return VGroup(*[Dot([x + k * gap, y, 0], radius=r).set_fill(GREY_B) for k in (-1, 0, 1)])
 
 
def ddots(x, y=0.0, gap=0.18, r=0.03):
    return VGroup(*[Dot([x + k * gap, y - k * gap, 0], radius=r).set_fill(GREY_B) for k in (-1, 0, 1)])
 
 
def sub_label(base, sub, size=30):
    """x with a small subscript (no LaTeX needed)."""
    b = Text(base, font_size=size)
    s = Text(sub, font_size=max(int(size * 0.6), 12))
    s.next_to(b, RIGHT, buff=0.02).shift(DOWN * 0.07 * size / 30)
    return VGroup(b, s)
 
 
def bracket(height, left=True, tick=0.13):
    s = 1 if left else -1
    pts = [[s * tick, height / 2, 0], [0, height / 2, 0], [0, -height / 2, 0], [s * tick, -height / 2, 0]]
    return VMobject().set_points_as_corners(pts).set_stroke(WHITE, 2.5).set_fill(opacity=0)
 
 
def curly_brace(height, r=0.16):
    """A '{' built from arcs (ManimGL's own Brace needs LaTeX, this does not)."""
    h = height / 2
    parts = [
        Arc(PI / 2, PI / 2, radius=r, arc_center=[2 * r, h - r, 0]),
        Line([r, h - r, 0], [r, r, 0]),
        Arc(0, -PI / 2, radius=r, arc_center=[0, r, 0]),
        Arc(PI / 2, -PI / 2, radius=r, arc_center=[0, -r, 0]),
        Line([r, -r, 0], [r, -h + r, 0]),
        Arc(PI, PI / 2, radius=r, arc_center=[2 * r, -h + r, 0]),
    ]
    return VGroup(*parts).set_stroke(WHITE, 2).set_fill(opacity=0)
 
 
class Mat(VGroup):
    """A bracketed block of entries.  Each entry is text, (text, colour), or 'h' / 'v' / 'd' for
    horizontal / vertical / diagonal dots.  cells[r][c] is the mobject in row r, column c."""
 
    def __init__(self, rows, col_w=0.85, row_h=0.56, size=26, **kwargs):
        n_r, n_c = len(rows), len(rows[0])
        cells, parts = [], []
        for r, row in enumerate(rows):
            line = []
            for c, item in enumerate(row):
                cx, cy = (c - (n_c - 1) / 2) * col_w, ((n_r - 1) / 2 - r) * row_h
                if item == "h":
                    m = hdots(cx, cy, 0.13, 0.028)
                elif item == "v":
                    m = vdots(cx, cy, 0.13, 0.028)
                elif item == "d":
                    m = ddots(cx, cy, 0.11, 0.028)
                else:
                    text, color = item if isinstance(item, tuple) else (item, WHITE)
                    m = Text(text, font_size=size).set_color(color).move_to([cx, cy, 0])
                line.append(m)
                parts.append(m)
            cells.append(line)
        h, wx = n_r * row_h + 0.12, n_c * col_w / 2 + 0.08
        super().__init__(*parts, bracket(h).shift([-wx, 0, 0]), bracket(h, False).shift([wx, 0, 0]), **kwargs)
        self.cells, self.n_r, self.row_h = cells, n_r, row_h
 
    def row_y(self, r):
        return self.get_center()[1] + ((self.n_r - 1) / 2 - r) * self.row_h
 
 
def num(v, fmt="%.2f"):
    """A number as (text, colour): green if positive, red if negative, grey if zero."""
    s = fmt % v
    if float(s) == 0:
        return fmt % 0.0, GREY_B
    return s, (POS if v > 0 else NEG)
 
 
def display_rows(pred):
    """Digits whose rows are written out (0, 1, the predicted one, 9); None = a row of dots between them."""
    keep = sorted({0, 1, pred, 9})
    rows = []
    for i, d in enumerate(keep):
        if i and d != keep[i - 1] + 1:
            rows.append(None)
        rows.append(d)
    return rows
 
 
def pixel_window(x):
    """Three neighbouring pixels across the edge of the stroke, for the numbers shown in the matrices."""
    img = x.reshape(28, 28)
    for r in sorted(range(4, 24), key=lambda r: abs(r - 14)):          # rows near the middle first
        for c in range(1, 27):
            if img[r, c - 1] < 0.3 and img[r, c] > 0.6:
                return [r * 28 + c - 1, r * 28 + c, r * 28 + c + 1]
    return [14 * 28 + 13, 14 * 28 + 14, 14 * 28 + 15]
 
 
def pick_input_pixels(x, n_mid=10):
    """The 16 input neurons that get drawn: x1, x2, x3, ten 'ink' pixels, x782, x783, x784."""
    ink = np.where(x > 0.25)[0]
    if len(ink) >= n_mid:
        mid = ink[np.linspace(0, len(ink) - 1, n_mid).round().astype(int)]
    else:                                                       # a very faint drawing: pad from the middle rows
        rest = np.setdiff1d(np.arange(4 * 28, 24 * 28), ink)
        mid = np.concatenate([ink, rest[np.linspace(0, len(rest) - 1, n_mid - len(ink)).round().astype(int)]])
    return np.concatenate([[0, 1, 2], np.sort(mid), [781, 782, 783]]).astype(int)
 
 
# ----------------------------------------------------------------------------- the 28 x 28 canvas
class PixelGrid(VGroup):
    """28 x 28 square pixels.  `.values` (28 x 28, 0..1) is exactly what the networks read."""
 
    def __init__(self, cell=CELL, **kwargs):
        squares = []
        for r in range(28):
            for c in range(28):
                sq = Square(cell).set_fill(BLACK, 1.0).set_stroke("#2B2B2B", 1.0)
                sq.move_to([(c - 13.5) * cell, (13.5 - r) * cell, 0])
                squares.append(sq)
        super().__init__(*squares, **kwargs)
        self.squares = squares
        self.values = np.zeros((28, 28))
 
    def set_value(self, r, c, v):
        v = float(np.clip(v, 0.0, 1.0))
        self.values[r, c] = v
        self.squares[r * 28 + c].set_fill(gray(v), 1.0)
 
    def load(self, values):
        for r, c in zip(*np.nonzero(values != self.values)):
            self.set_value(r, c, values[r, c])
 
    def stamp(self, u, v, erase=False):
        """One brush dab at pixel coordinates (u across, v down); a soft round brush ~3 pixels wide."""
        r0, r1 = max(0, int(np.floor(v - BRUSH_EDGE))), min(27, int(np.ceil(v + BRUSH_EDGE)))
        c0, c1 = max(0, int(np.floor(u - BRUSH_EDGE))), min(27, int(np.ceil(u + BRUSH_EDGE)))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                a = np.clip((BRUSH_EDGE - np.hypot(c + 0.5 - u, r + 0.5 - v)) / (BRUSH_EDGE - BRUSH_CORE), 0, 1)
                if a <= 0:
                    continue
                old = self.values[r, c]
                new = max(old - 1.5 * a, 0.0) if erase else max(old, a)
                if new != old:
                    self.set_value(r, c, new)
 
    def stroke(self, p0, p1, erase=False):
        """Dab along the segment p0 -> p1 so fast mouse moves leave no gaps."""
        n = max(1, int(np.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 0.3))
        for t in np.linspace(0.0, 1.0, n + 1):
            self.stamp(p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), erase)
 
 
class NewDigit(Exception):
    """Raised when N is pressed: back to an empty canvas."""
 
 
# ----------------------------------------------------------------------------- the scenes
class DigitScene(Scene):
 
    def setup(self):
        self.camera.background_rgba = [0.0, 0.0, 0.0, 1.0]       # pure black, like the 3b1b video
        self.weights = load_weights()
        self.pixels = np.zeros((28, 28))            # the drawing, values 0..1 = the raw 28 x 28 input
        self.canvas_active = False
        self.erasing = False
        self.last = None
        self.go = False
        self.new_digit = False
        self.in_idx = None                          # which 16 of the 784 input neurons are drawn
 
    # ------------------------------------------------------------------ mouse & keyboard
    def to_pixel(self, point):
        """Scene coordinates -> (across, down) in canvas-pixel units (the canvas is centred on the screen)."""
        return np.array([point[0] / CELL + 14.0, 14.0 - point[1] / CELL])
 
    def on_mouse_press(self, point, button, mods):
        if self.canvas_active:
            self.erasing = button == MOUSE.RIGHT
            self.last = self.to_pixel(point)
            self.grid.stamp(*self.last, erase=self.erasing)
 
    def on_mouse_drag(self, point, d_point, buttons, modifiers):
        if self.canvas_active:
            p = self.to_pixel(point)
            self.grid.stroke(self.last if self.last is not None else p, p, erase=self.erasing)
            self.last = p
 
    def on_mouse_release(self, point, button, mods):
        self.last = None
 
    def on_mouse_motion(self, point, d_point):      # (the default would pan / rotate the camera)
        pass
 
    def on_mouse_scroll(self, point, offset, x_pixel_offset, y_pixel_offset):
        pass
 
    def on_key_press(self, symbol, modifiers):
        if symbol in (KEY.ENTER, KEY.NUM_ENTER):
            if self.canvas_active and self.grid.values.sum() > 6:      # need some ink
                self.go = True
        elif symbol == KEY.C and not modifiers & (KEY.MOD_CTRL | KEY.MOD_COMMAND):
            if self.canvas_active:
                self.grid.load(np.zeros((28, 28)))
        elif symbol == KEY.N:
            self.new_digit = True
            self.hold_on_wait = False
        elif symbol in (KEY.SPACE, KEY.RIGHT):
            self.hold_on_wait = False
        elif symbol == KEY.Q and modifiers & (KEY.MOD_CTRL | KEY.MOD_COMMAND):
            self.quit_interaction = True
 
    def next_step(self):
        """Hold the picture until Space is pressed (N = start over with a new digit)."""
        if self.window is None:                     # rendering to a file: nobody to wait for
            self.wait(0.5)
            return
        self.new_digit = False
        self.hold_on_wait = True
        self.hold_loop()
        if self.new_digit:
            raise NewDigit()
 
    # ------------------------------------------------------------------ the talk
    def play_through(self, *scenes):
        """Draw a digit, then play the scenes in order (Space between them).  N starts over."""
        while True:
            try:
                self.draw_digit()
                self.centre_digit()
                self.next_step()
                for scene in scenes:
                    scene()
                    self.next_step()
            except NewDigit:
                pass
            self.clear()
            if self.window is None:
                break
 
    # ------------------------------------------------------------------ step 1: draw
    def draw_digit(self):
        self.clear()
        self.in_idx = None
        self.grid = PixelGrid()
        self.border = Rectangle(28 * CELL, 28 * CELL).set_stroke(BLUE_B, 3).set_fill(opacity=0)
        self.guide = Rectangle(20 * CELL, 20 * CELL).set_stroke(WHITE, 2, opacity=0.35).set_fill(opacity=0)
        self.add(self.grid, self.guide, self.border)
        self.canvas_active, self.go, self.last = True, False, None
        print("\nDraw a digit with the mouse (right button erases).  Enter = go,  C = clear.")
        self.wait_for_drawing()
        self.canvas_active = False
        self.pixels = self.grid.values.copy()
 
    def wait_for_drawing(self):
        if self.window is None:
            return
        while not self.go:                          # events (mouse / keys) are handled inside update_frame
            self.update_frame(1 / 60)               # redraw at up to 60 fps so the brush feels smooth
 
    def centre_digit(self):
        """Slide the drawing (whole pixels) so its centre of mass is in the middle, like MNIST digits."""
        dy, dx = centre_shift(self.pixels)
        if (dy, dx) == (0, 0):
            return
        ink = VGroup(*[self.grid.squares[r * 28 + c].copy() for r, c in zip(*np.nonzero(self.pixels > 0.03))])
        self.grid.load(np.zeros((28, 28)))
        self.add(ink)
        self.play(ink.animate.shift([dx * CELL, -dy * CELL, 0]), run_time=0.9)
        self.pixels = shift_image(self.pixels, dy, dx)
        self.grid.load(self.pixels)
        self.remove(ink)
 
    # ------------------------------------------------------------------ building blocks
    def input_layer(self, x):
        """The 784 input neurons: 16 drawn (+ dots), a brace and the number 784."""
        if self.in_idx is None:
            self.in_idx = pick_input_pixels(x)
        neurons = VGroup(*[neuron(IN_X, y, x[i]) for y, i in zip(elided_ys(), self.in_idx)])
        brace = curly_brace(7.2).move_to([IN_X - 0.56, 0, 0])
        label = Text("784", font_size=40).next_to(brace, LEFT, buff=0.15)
        return neurons, vdots(IN_X), brace, label
 
    def input_side(self, x):
        """Everything left of the network: the small picture, '28×28', the 784 neurons, brace and label."""
        small = PixelGrid(SMALL_CELL)
        small.load(self.pixels)
        small.set_stroke(width=0)
        small.move_to(IMAGE_POS)
        frame = Rectangle(28 * SMALL_CELL, 28 * SMALL_CELL).set_stroke(BLUE_B, 3).set_fill(opacity=0)
        frame.move_to(IMAGE_POS)
        tag = Text("28×28", font_size=22).set_color(GREY_B).next_to(frame, DOWN, buff=0.12)
        neurons, in_dots, brace, label = self.input_layer(x)
        return small, frame, tag, neurons, in_dots, brace, label
 
    def output_layer(self):
        neurons = VGroup(*[neuron(OUT_X, y, 0.0, r=0.17) for y in 2.25 - 0.5 * np.arange(10)])
        labels = VGroup(*[Text(str(i), font_size=30).next_to(neurons[i], RIGHT, buff=0.2) for i in range(10)])
        box = Rectangle(1.35, 5.5).set_stroke(YELLOW, 2.5).set_fill(opacity=0).move_to([OUT_X + 0.28, 0, 0])
        return neurons, labels, box
 
    def hidden_layer(self, x, n_total):
        """16 drawn neurons: all of them if the layer has 16, else 8 + dots + 8 (like the input)."""
        if n_total > 16:
            neurons = VGroup(*[neuron(x, y) for y in elided_ys()])
            return neurons, vdots(x), list(range(8)) + list(range(n_total - 8, n_total))
        return VGroup(*[neuron(x, y) for y in 3.0 - STEP * np.arange(16)]), VGroup(), list(range(16))
 
    def make_edges(self, src, dst, W):
        """One line per (src, dst) pair.  W[j, i] = weight from src neuron i to dst neuron j.
        Green = positive weight, red = negative, more opaque = larger |weight|."""
        a = np.abs(W)
        top = np.percentile(a, 95) + 1e-9
        edges = VGroup()
        for j, d in enumerate(dst):
            for i, s in enumerate(src):
                e = Line(s.get_right(), d.get_left())
                e.set_stroke(POS if W[j, i] > 0 else NEG, width=1.6, opacity=0.9 * float(min(a[j, i] / top, 1.0)))
                edges.add(e)
        return edges
 
    def reveal_edges(self, edges, run_time=2.0):
        self.add(edges)
        self.bring_to_back(edges)                   # lines go behind the neurons
        self.play(ShowCreation(edges, lag_ratio=0.01), run_time=run_time)
 
    def light_up(self, neurons, values, run_time=1.2, extra=()):
        self.play(*[n.animate.set_fill(gray(v), 1.0) for n, v in zip(neurons, values)], *extra, run_time=run_time)
 
    def show_prediction(self, out_neurons, out_labels, probs):
        k = int(np.argmax(probs))
        big = Text(str(k), font_size=150).set_color(YELLOW).move_to([6.15, 0.45, 0])
        conf = Text("%d%%" % int(100 * probs[k]), font_size=34).set_color(GREY_B).next_to(big, DOWN, buff=0.25)
        self.play(out_neurons[k].animate.set_stroke(YELLOW, 4),
                  out_labels[k].animate.set_color(YELLOW).scale(1.4),
                  FadeIn(big), FadeIn(conf), run_time=1.0)
 
    # ------------------------------------------------------------------ Scene 1: no hidden layers
    def linear_scene(self):
        x = self.pixels.reshape(784)
        z = linear_scores(self.weights, x)          # z = W x + b : ten numbers, one per digit
        self.flatten_chapter(x)
        self.next_step()
        self.matrix_chapter(x, z)
        self.next_step()
        self.dims_chapter()
        self.next_step()
        self.network_chapter(x, softmax(z))
        self.next_step()
        self.weights_chapter()
 
    # ---- chapter 1: 28 x 28 image  ->  flatten  ->  x = [x1, x2, ..., x784]  ->  784 neurons
    def flatten_chapter(self, x):
        # (a) divide the picture into its 784 squares
        label = Text("28 × 28", font_size=40).next_to(self.border, UP, buff=0.25)
        gapped = self.grid.copy()
        for sq in gapped.submobjects:
            sq.scale(0.78).set_stroke("#707070", 1.2)
        self.play(FadeOut(self.guide), FadeIn(label), Transform(self.grid, gapped), run_time=1.0)
        self.next_step()
 
        # (b) flatten: the squares pour, row by row, into the column vector x
        pic = VGroup(self.grid, self.border)
        self.play(FadeOut(label), pic.animate.scale(0.6).move_to([-4.6, 0.0, 0.0]), run_time=1.0)
        pic_label = Text("28 × 28", font_size=32).next_to(self.border, UP, buff=0.2)
        arrow = Arrow([-2.5, 0, 0], [-0.7, 0, 0], buff=0)
        arrow_text = Text("flatten", font_size=34).next_to(arrow, UP, buff=0.15)
 
        vx = 2.35                                                       # x of the column of entries
        slot = {0: [vx, 1.8, 0], 1: [vx, 0.9, 0], 2: [vx, 0.0, 0], 783: [vx, -1.8, 0]}
        dots_at = [vx, -0.9, 0]
        x_eq = Text("x =", font_size=44).move_to([vx - 1.55, 0, 0])
        lb, rb = bracket(4.8).shift([vx - 0.55, 0, 0]), bracket(4.8, False).shift([vx + 1.65, 0, 0])
        outlines = VGroup(*[Square(0.5).set_stroke(GREY_D, 1.5).set_fill(opacity=0).move_to(p) for p in slot.values()])
        names = VGroup(*[sub_label("x", s, 34).move_to([vx + 0.45, p[1], 0], aligned_edge=LEFT)
                         for s, p in zip(("1", "2", "3", "784"), slot.values())])
        vec_dots = vdots(*dots_at[:2])
        dims = Text("784 × 1", font_size=26).set_color(GREY_B).move_to([vx + 0.55, -2.85, 0])
        self.play(*[FadeIn(m) for m in (arrow, arrow_text, pic_label, x_eq, lb, rb, outlines, names, vec_dots, dims)],
                  run_time=0.8)
 
        tiles = [sq.copy() for sq in self.grid.submobjects]             # copies sitting exactly on the squares
        targets = []
        for i, t in enumerate(tiles):
            if i in slot:                                               # x1, x2, x3 and x784 land in their slots
                tgt = Square(0.5).set_stroke(WHITE, 1.5).set_fill(gray(x[i]), 1.0).move_to(slot[i])
            else:                                                       # the others pour into the dots
                tgt = Square(0.04).set_stroke(opacity=0).set_fill(gray(x[i]), 0.0).move_to(dots_at)
            targets.append(tgt)
        stream = VGroup(*tiles)
        self.add(stream)
        self.play(Transform(stream, VGroup(*targets), lag_ratio=0.0013), run_time=4.0)
        landed = [stream.submobjects[i] for i in (0, 1, 2, 783)]
        self.remove(stream)
        self.add(*landed)
        self.next_step()
 
        # (c) the entries of x are the 784 neurons of the input layer
        small, frame, tag, neurons, in_dots, brace, label784 = self.input_side(x)
        pours = [ReplacementTransform(Dot(dots_at, radius=0.03).set_fill(GREY_B), neurons[k]) for k in range(3, 15)]
        fades = [FadeOut(m) for m in (arrow, arrow_text, pic_label, x_eq, lb, rb, outlines, names, dims)]
        self.play(*fades,
                  Transform(self.grid, small), Transform(self.border, frame),
                  *[ReplacementTransform(landed[j], neurons[j]) for j in (0, 1, 2)],
                  ReplacementTransform(landed[3], neurons[15]),
                  ReplacementTransform(vec_dots, in_dots),
                  LaggedStart(*pours, lag_ratio=0.06),
                  run_time=2.4)
        self.play(FadeIn(brace), FadeIn(label784), FadeIn(tag), run_time=0.7)
 
    # ---- chapter 2: z = W x + b, written out with real numbers
    def matrix_chapter(self, x, z):
        self.clear()
        W, b = self.weights["L_W"], self.weights["L_b"]
        pred = int(np.argmax(z))
        rows, cols = display_rows(pred), pixel_window(x)
 
        W_rows, b_rows, z_rows = [], [], []
        for d in rows:
            if d is None:
                W_rows.append(["d", "v", "v", "v", "d"])
                b_rows.append(["v"])
                z_rows.append(["v"])
            else:
                W_rows.append(["h"] + [num(W[d, c]) for c in cols] + ["h"])
                b_rows.append([num(b[d])])
                z_rows.append([("%.1f" % z[d], YELLOW if d == pred else WHITE)])
        x_rows = [["v"]] + [[("%.2f" % x[c], WHITE if x[c] > 0.005 else GREY_B)] for c in cols] + [["v"]]
 
        W_m, x_m, b_m, z_m = Mat(W_rows, 0.82), Mat(x_rows, 0.95), Mat(b_rows, 0.95), Mat(z_rows, 1.0)
        times, plus, eq = (Text(s, font_size=44) for s in ("×", "+", "="))
        VGroup(W_m, times, x_m, plus, b_m, eq, z_m).arrange(RIGHT, buff=0.35).center().shift(UP * 0.35)
 
        def caption(mat, name, dims):
            n = Text(name, font_size=40).next_to(mat, UP, buff=0.3)
            d = Text(dims, font_size=24).set_color(GREY_B).next_to(mat, DOWN, buff=0.3)
            return VGroup(n, d)
 
        W_cap, x_cap, b_cap, z_cap = (caption(W_m, "W", "10 × 784"), caption(x_m, "x", "784 × 1"),
                                      caption(b_m, "b", "10 × 1"), caption(z_m, "z", "10 × 1"))
        digit_labels = VGroup(*[Text(str(d), font_size=22).set_color(YELLOW if d == pred else GREY_C)
                                .move_to([W_m.get_left()[0] - 0.3, W_m.row_y(r), 0])
                                for r, d in enumerate(rows) if d is not None])
 
        self.play(FadeIn(W_m), FadeIn(W_cap), FadeIn(digit_labels), run_time=0.8)
        self.play(FadeIn(times), FadeIn(x_m), FadeIn(x_cap), run_time=0.8)
        self.play(FadeIn(plus), FadeIn(b_m), FadeIn(b_cap), run_time=0.8)
        self.next_step()
 
        # one row of W against x, plus b, gives one entry of z
        for line in z_m.cells:
            for m in line:
                m.set_opacity(0)
        self.play(FadeIn(eq), FadeIn(z_m), FadeIn(z_cap), run_time=0.6)
        hl_w = Rectangle(W_m.get_width() + 0.1, 0.48).set_stroke(YELLOW, 2.5).set_fill(opacity=0)
        hl_x = Rectangle(x_m.get_width() + 0.1, 3 * x_m.row_h + 0.06).set_stroke(YELLOW, 2.5).set_fill(opacity=0)
        hl_w.move_to([W_m.get_center()[0], W_m.row_y(0), 0])
        hl_x.move_to([x_m.get_center()[0], x_m.row_y(2), 0])
        self.play(FadeIn(hl_w), FadeIn(hl_x), run_time=0.4)
        for r, d in enumerate(rows):
            if d is None:
                continue
            self.play(hl_w.animate.move_to([W_m.get_center()[0], W_m.row_y(r), 0]),
                      z_m.cells[r][0].animate.set_opacity(1), run_time=0.7)
        self.play(FadeOut(hl_w), FadeOut(hl_x),
                  *[m.animate.set_opacity(1) for r, d in enumerate(rows) if d is None for m in z_m.cells[r]],
                  run_time=0.5)
        self.play(FadeIn(Text("z = Wx + b", font_size=52).move_to([0, -3.15, 0])), run_time=0.7)
 
    # ---- chapter 3: 784 dimensions -> matrix transformation -> 10 dimensions
    def dims_chapter(self):
        self.clear()
        t = [Text("784 dimensions", font_size=54),
             Text("matrix transformation", font_size=54).set_color(YELLOW),
             Text("10 dimensions", font_size=54)]
        VGroup(*t).arrange(DOWN, buff=1.2).center()
        a1 = Arrow(t[0].get_bottom(), t[1].get_top(), buff=0.2)
        a2 = Arrow(t[1].get_bottom(), t[2].get_top(), buff=0.2)
        self.play(FadeIn(t[0]), run_time=0.6)
        self.play(FadeIn(a1), FadeIn(t[1]), run_time=0.7)
        self.play(FadeIn(a2), FadeIn(t[2]), run_time=0.7)
 
    # ---- chapter 4: the same thing as a network
    def network_chapter(self, x, probs):
        self.clear()
        small, frame, tag, in_neurons, in_dots, brace, label = self.input_side(x)
        out_neurons, out_labels, out_box = self.output_layer()
        self.add(small, frame, tag, in_neurons, in_dots, brace, label)
        self.play(FadeIn(out_box), FadeIn(out_neurons), FadeIn(out_labels))
        self.reveal_edges(self.make_edges(in_neurons, out_neurons, self.weights["L_W"][:, self.in_idx]), 2.5)
        self.next_step()
        soft = Text("softmax", font_size=24).set_color(GREY_B).move_to([OUT_X + 0.28, 3.05, 0])
        self.light_up(out_neurons, probs, 1.5, extra=(FadeIn(soft),))
        self.show_prediction(out_neurons, out_labels, probs)
 
    # ---- chapter 5: what each output neuron looks for = one row of W drawn as a 28 x 28 picture
    def weights_chapter(self):
        self.clear()
        W = self.weights["L_W"]
        top = np.percentile(np.abs(W), 99.5)                    # one scale for all ten maps
        folder = Path(tempfile.mkdtemp())
        maps = []
        for d in range(10):
            r, c = divmod(d, 5)
            cx, cy = (c - 2) * 2.7, 1.3 - r * 3.1
            v = np.clip(W[d].reshape(28, 28) / top, -1, 1)
            rgb = np.stack([np.clip(-v, 0, 1), np.clip(v, 0, 1), np.zeros_like(v)], axis=-1)     # red -, green +
            path = folder / ("w%d.png" % d)
            Image.fromarray((rgb * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST).save(path)
            img = ImageMobject(str(path), height=2.1).move_to([cx, cy, 0])
            frame = Rectangle(2.22, 2.22).set_stroke(WHITE, 2).set_fill(opacity=0).move_to([cx, cy, 0])
            tag = Text(str(d), font_size=34).next_to(frame, UP, buff=0.12)
            maps.append((img, frame, tag))
        legend = VGroup(Square(0.22).set_fill("#00FF00", 1).set_stroke(width=0), Text("+", font_size=28),
                        Square(0.22).set_fill("#FF0000", 1).set_stroke(width=0), Text("−", font_size=28))
        legend.arrange(RIGHT, buff=0.2).move_to([0, -3.65, 0])
 
        for d, (img, frame, tag) in enumerate(maps):            # one more map on every Space
            frame.set_stroke(YELLOW, 3.5)
            anims = [FadeIn(img), FadeIn(frame), FadeIn(tag)]
            if d:
                anims.append(maps[d - 1][1].animate.set_stroke(WHITE, 2))
            else:
                anims.append(FadeIn(legend))
            self.play(*anims, run_time=0.6)
            if d < 9:
                self.next_step()
 
    # ------------------------------------------------------------------ Scene 2: two hidden layers
    def hidden_scene(self):
        self.clear()                                # a clean cut: nothing from the previous scene stays
        x = self.pixels.reshape(784)
        a1, a2, probs = hidden_forward(self.weights, x)
        w = self.weights
 
        # the picture and its 784 neurons, as at the end of the flatten chapter
        small, frame, tag, in_neurons, in_dots, brace, label = self.input_side(x)
        self.play(FadeIn(VGroup(small, frame)), FadeIn(tag), FadeIn(in_neurons), FadeIn(in_dots),
                  FadeIn(brace), FadeIn(label))
        self.next_step()
 
        # first hidden layer: 32 neurons (16 drawn)
        h1, h1_dots, h1_idx = self.hidden_layer(HIDDEN_X[0], H1)
        n1 = Text(str(H1), font_size=26).set_color(GREY_B).move_to([HIDDEN_X[0], -3.78, 0])
        self.play(FadeIn(h1), FadeIn(h1_dots), FadeIn(n1))
        self.reveal_edges(self.make_edges(in_neurons, h1, w["W1"][np.ix_(h1_idx, self.in_idx)]), 2.0)
        self.next_step()
 
        # second hidden layer: 16 neurons
        h2, _, _ = self.hidden_layer(HIDDEN_X[1], H2)
        n2 = Text(str(H2), font_size=26).set_color(GREY_B).move_to([HIDDEN_X[1], -3.78, 0])
        self.play(FadeIn(h2), FadeIn(n2))
        self.reveal_edges(self.make_edges(h1, h2, w["W2"][:, h1_idx]), 2.0)
        self.next_step()
 
        # output layer
        out_neurons, out_labels, out_box = self.output_layer()
        self.play(FadeIn(out_box), FadeIn(out_neurons), FadeIn(out_labels))
        self.reveal_edges(self.make_edges(h2, out_neurons, w["W3"]), 1.5)
        self.next_step()
 
        # the activations ripple through, layer by layer
        self.light_up(h1, a1[h1_idx], 1.2)
        self.wait(0.3)
        self.light_up(h2, a2, 1.2)
        self.wait(0.3)
        self.light_up(out_neurons, probs, 1.2)
        self.show_prediction(out_neurons, out_labels, probs)
 
 
class Presentation(DigitScene):
    """The whole talk: Scene 1 (linear, no hidden layers) then Scene 2 (two hidden layers)."""
 
    def construct(self):
        self.play_through(self.linear_scene, self.hidden_scene)
 
 
class LinearScene(DigitScene):
    """Scene 1 on its own."""
 
    def construct(self):
        self.play_through(self.linear_scene)
 
 
class HiddenLayersScene(DigitScene):
    """Scene 2 on its own (you draw a digit first)."""
 
    def construct(self):
        self.play_through(self.hidden_scene)
 