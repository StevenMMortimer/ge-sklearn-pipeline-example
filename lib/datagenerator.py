import numpy as np
import pandas as pd

from lib.constants import *


def make_fake_data():
    df = pd.DataFrame(data={'species': np.repeat(SPECIES, 8),
                            'color': COLOR * 8,
                            'beak_ratio': ['low', 'high'] * 8,
                            'claw_length': np.random.rand(16),
                            'wing_density': np.random.rand(16),
                            'weight': [110, 130] * 4 + [200, 220] * 4})

    np.random.seed(1)
    df['weight'] = df['weight'] + np.round(np.random.rand(16), 3)

    return df
