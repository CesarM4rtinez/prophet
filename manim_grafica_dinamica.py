from manim import *

class DynamicGoldSeries(Scene):
    def construct(self):
        title = Text("Precio simulado XAUUSD", font_size=44)
        title.to_edge(UP)

        axes = Axes(
            x_range=[0, 10, 1],
            y_range=[0.8, 1.5, 0.1],
            x_length=10,
            y_length=5,
            axis_config={"include_tip": True},
        )
        axes_labels = axes.get_axis_labels(x_label="Tiempo", y_label="Precio (normalizado)")

        curve = axes.plot(lambda x: 1.1 + 0.2 * np.sin(1.5 * x) + 0.05 * np.cos(5 * x), x_range=[0, 10], color=GOLD)
        curve.set_stroke(width=4)

        dot = Dot(point=axes.c2p(0, 1.1 + 0.2 * np.sin(0) + 0.05 * np.cos(0)), color=YELLOW)
        trace = TracedPath(dot.get_center, stroke_color=YELLOW, stroke_width=3)

        self.play(FadeIn(title), Write(axes), Write(axes_labels), run_time=2)
        self.play(Create(curve), run_time=4)
        self.add(trace, dot)
        self.play(MoveAlongPath(dot, curve), run_time=6, rate_func=linear)
        self.wait(2)