"""Plotting built on top of the metrics logger.

Replaces pisces2.live_plotting.LiveCVPlotter. Each plot function reads the
metrics CSV (long or wide) and derives its inputs from there, so there is a
single source of truth for what got logged during an experiment.
"""
