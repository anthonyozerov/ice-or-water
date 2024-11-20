import numpy as np
import pandas as pd


# dawson et al
def get_dawson_labels(df, q=None, thresh=None):
    assert q is not None or thresh is not None, "Either q or thresh must be provided"

    if q is not None:
        q1, q2 = q
        slow_thresh = np.nanquantile(df["vel"][df["grounded"]], q1)
        fast_thresh = np.nanquantile(df["vel"][df["grounded"]], q2)
    else:
        slow_thresh, fast_thresh = thresh

    def label(row):
        if row["vel"] < slow_thresh:
            val = "frozen"
        elif row["vel"] > fast_thresh:
            val = "thawed"
        else:
            val = "unlabeled"
        return val

    labels = pd.cut(
        df["vel"],
        [-np.inf, slow_thresh, fast_thresh, np.inf],
        labels=["frozen", "unlabeled", "thawed"],
    )

    return labels, [slow_thresh, fast_thresh]
