

from clients.nflreadpy.player_data import load_player_stats
import polars as pl

df = load_player_stats(2025)

df = df.filter(
    pl.col('display_name') == 'Drake Maye'
)