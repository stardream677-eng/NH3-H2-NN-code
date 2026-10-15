import glob
import joblib
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, ".")
import train_sc_v68 as tr
from enhanced_loss_and_weights import MaskedOnlineMoments

fi_path = glob.glob("output1/phase68_runs/v72_cfd_optimized/scalers_v68/feature_info_*.pkl")[0]
fi = joblib.load(fi_path)
y_dim = len(fi["output_cols"])

mom_std = tr.OnlineMoments(y_dim)
mom_masked = MaskedOnlineMoments(y_dim, sentinel_threshold=-35.0)

files = glob.glob("C:/Users/123/Desktop/2STest/second chapter/009/xl1/sc14/07/01/database_Transient_V21_QSSA/trajectories_nuig_2024/*.parquet")
for f in files:
    df = pd.read_parquet(f)
    macro_scale = fi.get("macro_rate_scale", np.ones(len(fi["macro_species"])))
    Y = tr.transform_target_array(
        df,
        fi["macro_species"],
        fi["micro_species"],
        macro_scale,
        output_parameterization=fi.get("output_parameterization", "direct_rate"),
    )
    mom_std.update(Y)
    mom_masked.update(Y)

s_std = mom_std.finalize()
s_masked = mom_masked.finalize()

print("Standard scaler_Y std range:", f"{s_std.scale_.min():.3f}", "to", f"{s_std.scale_.max():.3f}")
print("Masked scaler_Y std range:  ", f"{s_masked.scale_.min():.3f}", "to", f"{s_masked.scale_.max():.3f}")
for sp in ["OH", "H", "O", "HO2", "NO", "NH2", "HNO"]:
    col = f"log10_Y_qssa_{sp}"
    if col in fi["output_cols"]:
        idx = fi["output_cols"].index(col)
        ratio = s_std.scale_[idx] / s_masked.scale_[idx]
        print(f"  {sp:>4s}: std={s_std.scale_[idx]:.2f} -> masked={s_masked.scale_[idx]:.2f} (sensitivity boost: {ratio:.1f}x)")
