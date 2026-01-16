from __future__ import annotations

from patchcorex.scoring.rsw_e import RSWEScorer
from patchcorex.utils.registry import SCORERS


@SCORERS.register("rrsw_e")
class RRSWEScorer(RSWEScorer):
    pass
