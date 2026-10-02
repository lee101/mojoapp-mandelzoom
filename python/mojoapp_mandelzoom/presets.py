"""Named views: the coordinates people actually want to look at.

These are the classic landmarks of the set plus a couple of deep-zoom anchors.
Each is a `View`, so a preset is a starting point for `zoom_sequence` rather
than a fixed picture -- `preset("seahorse")` is the top of a zoom you can keep
descending, which is the actual pitch of this app.

The coordinates are the published ones for these features; `scale` is set so
each preset frames its subject on an 16:9 canvas.
"""

from __future__ import annotations

from .render import View

#: name -> View. Kept as a dict literal so `presets_names()` and the docs cannot
#: drift from what is actually there.
PRESETS: dict[str, View] = {
    # The whole set, the default view.
    "full": View(-0.5, 0.0, 1.25, "full set"),
    # The cusp at -2, where the two big lobes meet.
    "cusp": View(-1.75, 0.0, 0.28, "cusp"),
    # The cardioid's far tip at 0.25 and the period-2 bulb beside it.
    "cardioid": View(0.285, 0.01, 0.075, "cardioid tip"),
    # Seahorse Valley: the first place filaments appear going deep.
    "seahorse": View(-0.743643887037151, 0.131825904205330, 0.0042, "seahorse valley"),
    # The classic double spiral.
    "spiral": View(-0.1592, 1.0317, 0.0085, "double spiral"),
    # An elephant valley, east of the cardioid.
    "elephant": View(0.2925755, 0.0149977, 0.0035, "elephant valley"),
    # Three-quarter spiral, past the elephant.
    "spiral3": View(-0.16070135, 1.0375665, 0.00035, "three-quarter spiral"),
    # Misiurewicz point: a preperiodic landing point, visually striking.
    "misiurewicz": View(-0.77568377, 0.13646737, 0.00025, "misiurewicz point"),
    # Deep seahorse. At half-height 1e-9 the escape time is O(1e9) in places,
    # so this only resolves with --max-iter in the thousands; at the default 500
    # it is a solid black frame, which is correct but useless as a poster. The
    # depth is here because `zoom_sequence` is the intended entry point --
    # start at `seahorse` and let the sequence walk in, watching the iteration
    # budget climb. `python -m mojoapp_mandelzoom card --preset deep
    # --max-iter 200000` resolves it, in about a second.
    "deep": View(-0.743643887037158, 0.131825904205311, 1.0e-9, "deep seahorse"),
}
