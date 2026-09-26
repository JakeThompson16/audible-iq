

import nflreadpy as nfl

df = nfl.load_ff_playerids()

print(df.head())
print(df.columns)