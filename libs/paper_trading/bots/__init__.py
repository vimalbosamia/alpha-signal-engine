"""
Bot registry — all six paper-trading bot subclasses.

Import this module to get access to all bots and the ALL_BOTS registry list.
"""
from libs.paper_trading.bots.adaptive import AdaptiveBot
from libs.paper_trading.bots.mean_reversion import MeanReversionBot
from libs.paper_trading.bots.momentum import MomentumBot
from libs.paper_trading.bots.reversal import ReversalBot
from libs.paper_trading.bots.scalper import ScalperBot
from libs.paper_trading.bots.swing import SwingBot

# Order matters: first match wins. Strategy specialists first, catch-alls last.
ALL_BOTS = [MomentumBot, ReversalBot, MeanReversionBot, SwingBot, ScalperBot, AdaptiveBot]

__all__ = [
    "MomentumBot",
    "ReversalBot",
    "MeanReversionBot",
    "ScalperBot",
    "SwingBot",
    "AdaptiveBot",
    "ALL_BOTS",
]
