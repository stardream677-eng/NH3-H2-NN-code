"""
LHS Sampler V11 (Physical Species Edition) - 全域瞬态物种初始状态采样器 (重构增强版)
=============================================================================
核心改进：
1. 真实“燃料+空气”基准配比 (Air-Oxidizer Baseline)：
   基于干空气 N2/O2 = 3.7619 计算化学计量学，消除旧版独立采样 N2 导致的“空气配比样本为零”缺陷。
2. 物理稀释解耦 (Decoupled Dilution / EGR)：
   通过稀释率 Z_dil ∈ [0, 0.95] 控制外加 EGR (N2, H2O) 或惰性气体 (Ar, He) 稀释，
   使得纯燃料+空气与稀释工况平滑连续过渡。
3. 严格当量比保真：
   实际计算出的 phi_actual 严格等于目标 phi (机器精度级别误差 < 1e-14)。
4. 四层多工况分层覆盖：
   - Spec_Engine_Core (35%): 核心发动机工况 (宽温高压、低稀释、接近真实空气燃烧)
   - Spec_Cold_Oxidation (25%): 低温慢速氧化与长诱导期 (覆盖慢反应)
   - Spec_Radical_Burst (20%): 预燃室活性自由基射流喷射 (模拟着火过渡态)
   - Spec_Extreme_Boundaries (20%): 极端贫/富燃边界与超高稀释 (捕捉极限工况)
=============================================================================
"""

import math
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import qmc

warnings.filterwarnings("ignore")

SUPPORTED_MECHANISMS = (
    "nuig_2024",
    "tsinghua_2024c",
    "ucf_2024",
)

# Standard dry air N2 / O2 molar ratio (0.78084 / 0.20946 ~ 3.7619)
AIR_N2_O2_RATIO = 3.76190476


def normalize_mechanism_selection(
    mechanism: str | None = None,
    mechanisms: list[str] | tuple[str, ...] | None = None,
) -> list[str]:
    """Validate and canonicalize the mechanism selection for data generation."""
    if mechanism is not None and mechanisms is not None:
        raise ValueError("--mechanism and --mechanisms are mutually exclusive")

    raw = (
        [mechanism]
        if mechanism is not None
        else list(mechanisms)
        if mechanisms is not None
        else list(SUPPORTED_MECHANISMS)
    )
    selected = [str(name).strip().lower() for name in raw]
    if not selected or any(not name for name in selected):
        raise ValueError("At least one mechanism must be selected")

    unknown = sorted(set(selected) - set(SUPPORTED_MECHANISMS))
    if unknown:
        supported = ", ".join(SUPPORTED_MECHANISMS)
        raise ValueError(
            f"Unknown mechanism(s): {', '.join(unknown)}. "
            f"Supported mechanisms: {supported}"
        )
    if len(set(selected)) != len(selected):
        raise ValueError("Mechanism selection contains duplicates")
    return selected


class UniversalLHSSampler_Species:
    def __init__(self, mechanisms_dict: dict):
        self.mechanisms = mechanisms_dict
        self.mech_names = list(self.mechanisms.keys())

        # 23D 输出特征：覆盖分层标识、背景稀释度及所有关键组分
        self.out_cols = [
            "stratum_name",
            "T", "P", "phi", "X_H2_fuel",
            "phi_actual",
            "radical_total",
            "Z_dil", "f_N2", "f_AR", "f_HE", "f_H2O",
            "X_NH3", "X_H2", "X_O2",
            "X_N2", "X_AR", "X_HE", "X_H2O",
            "X_OH", "X_H", "X_O", "X_NO",
        ]

    def _map_val(self, lhs_col: np.ndarray, bnd: list, scale: str = "linear") -> np.ndarray:
        if float(bnd[0]) == float(bnd[1]):
            return np.full_like(lhs_col, float(bnd[0]), dtype=np.float64)
        if scale == "log":
            if float(bnd[0]) <= 0.0 or float(bnd[1]) <= 0.0:
                raise ValueError("log-scaled bounds must be strictly positive")
            return np.exp(lhs_col * np.log(bnd[1] / bnd[0]) + np.log(bnd[0]))
        return lhs_col * (bnd[1] - bnd[0]) + bnd[0]

    def _generate_composite(
        self,
        n_pts: int,
        bounds: dict,
        seed: int,
        include_radicals: bool = True,
        stratum_name: str = "custom",
    ) -> pd.DataFrame:
        """Generate physically consistent initial states via Latin Hypercube Sampling.
        
        Guarantees:
        1. When Z_dil == 0, oxidizer is exact standard air (N2 / O2 = 3.7619).
        2. Chemical stoichiometry strictly yields phi_actual == phi.
        3. All mole fractions sum to 1.0 (to IEEE-754 precision).
        """
        dims = [
            "T", "P", "phi", "X_H2_fuel", "Z_dil",
            "f_N2", "f_AR", "f_HE", "f_H2O",
            "X_OH", "X_H", "X_O", "X_NO",
        ]
        sampler = qmc.LatinHypercube(d=len(dims), seed=seed)
        lhs = sampler.random(n=n_pts)

        T = self._map_val(lhs[:, 0], bounds["T"], bounds.get("scale_T", "linear"))
        P = self._map_val(lhs[:, 1], bounds["P"], bounds.get("scale_P", "log"))
        phi = self._map_val(lhs[:, 2], bounds["phi"], bounds.get("scale_phi", "linear"))
        X_H2_fuel = self._map_val(lhs[:, 3], bounds["X_H2_fuel"], "linear")

        # Total added diluent / EGR fraction Z_dil ∈ [0, 0.95]
        Z_dil = self._map_val(lhs[:, 4], bounds.get("Z_dil", [0.0, 0.25]), "linear")

        # Relative split of the added diluent amongst N2_dil, AR, HE, H2O
        f_N2 = self._map_val(lhs[:, 5], bounds.get("f_N2", [0.5, 1.0]), "linear")
        f_AR = self._map_val(lhs[:, 6], bounds.get("f_AR", [0.0, 0.0]), "linear")
        f_HE = self._map_val(lhs[:, 7], bounds.get("f_HE", [0.0, 0.0]), "linear")
        f_H2O = self._map_val(lhs[:, 8], bounds.get("f_H2O", [0.0, 0.2]), "linear")

        f_sum = f_N2 + f_AR + f_HE + f_H2O
        f_sum = np.where(f_sum > 0, f_sum, 1.0)
        f_N2 /= f_sum
        f_AR /= f_sum
        f_HE /= f_sum
        f_H2O /= f_sum

        # Active radicals (pre-chamber jet injection)
        if include_radicals:
            X_OH = self._map_val(lhs[:, 9], bounds.get("X_OH", [0.0, 0.0]), bounds.get("scale_rad", "log"))
            X_H = self._map_val(lhs[:, 10], bounds.get("X_H", [0.0, 0.0]), bounds.get("scale_rad", "log"))
            X_O = self._map_val(lhs[:, 11], bounds.get("X_O", [0.0, 0.0]), bounds.get("scale_rad", "log"))
            X_NO = self._map_val(lhs[:, 12], bounds.get("X_NO", [0.0, 0.0]), bounds.get("scale_rad", "log"))
        else:
            X_OH = np.zeros(n_pts, dtype=np.float64)
            X_H = np.zeros(n_pts, dtype=np.float64)
            X_O = np.zeros(n_pts, dtype=np.float64)
            X_NO = np.zeros(n_pts, dtype=np.float64)

        X_rad = X_OH + X_H + X_O + X_NO
        # Ensure diluent + radicals does not exceed 0.999 (reserving >= 0.1% for fuel/O2)
        total_non_reactants = np.clip(Z_dil + X_rad, 0.0, 0.999)
        Z_inerts = np.maximum(0.0, total_non_reactants - X_rad)

        X_N2_dil = Z_inerts * f_N2
        X_AR = Z_inerts * f_AR
        X_HE = Z_inerts * f_HE
        X_H2O = Z_inerts * f_H2O

        # Remaining fraction allocated strictly to Fuel + Air
        X_reactants = 1.0 - total_non_reactants

        # Stoichiometric O2 demand per mole of NH3/H2 fuel blend:
        # H2 + 0.5 O2 -> H2O  (0.5 mol O2 per mol H2)
        # NH3 + 0.75 O2 -> 0.5 N2 + 1.5 H2O (0.75 mol O2 per mol NH3)
        a_stoich = 0.5 * X_H2_fuel + 0.75 * (1.0 - X_H2_fuel)
        n_O2 = a_stoich / phi
        n_N2_air = AIR_N2_O2_RATIO * n_O2
        n_total_reactants = 1.0 + n_O2 + n_N2_air

        # Actual mole fractions
        X_NH3 = X_reactants * (1.0 - X_H2_fuel) / n_total_reactants
        X_H2 = X_reactants * X_H2_fuel / n_total_reactants
        X_O2 = X_reactants * n_O2 / n_total_reactants
        X_N2 = X_reactants * n_N2_air / n_total_reactants + X_N2_dil

        # Validate phi_actual: phi = (0.75 X_NH3 + 0.5 X_H2) / X_O2
        phi_actual = np.divide(
            0.75 * X_NH3 + 0.5 * X_H2,
            X_O2,
            out=np.full_like(X_O2, np.nan),
            where=X_O2 > 0.0,
        )

        phi_err = np.abs(phi_actual - phi)
        if not np.all(np.isfinite(phi_actual)) or not np.all(phi_err < 1e-8):
            raise ValueError(f"Sampled composition does not match target phi (max err: {np.max(phi_err):.3e})")

        total_sum = X_NH3 + X_H2 + X_O2 + X_N2 + X_AR + X_HE + X_H2O + X_rad
        if not np.allclose(total_sum, 1.0, atol=1e-10):
            raise ValueError(f"Mole fractions do not sum to 1.0 (max deviation: {np.max(np.abs(total_sum - 1.0)):.3e})")

        df = pd.DataFrame({
            "stratum_name": stratum_name,
            "T": T, "P": P, "phi": phi, "phi_actual": phi_actual,
            "radical_total": X_rad,
            "X_H2_fuel": X_H2_fuel,
            "Z_dil": Z_dil,
            "f_N2": f_N2, "f_AR": f_AR, "f_HE": f_HE, "f_H2O": f_H2O,
            "X_NH3": X_NH3, "X_H2": X_H2, "X_O2": X_O2,
            "X_N2": X_N2, "X_AR": X_AR, "X_HE": X_HE,
            "X_H2O": X_H2O, "X_OH": X_OH, "X_H": X_H, "X_O": X_O, "X_NO": X_NO,
        })
        return df[self.out_cols]

    def generate_species_database(
        self,
        total_pts: int = 15000,
        include_radicals: bool = True,
    ) -> pd.DataFrame:
        print("\n" + "=" * 70)
        print("[V11 Species] Generating Physical Initial States (Air-Oxidizer Baseline)")
        print("=" * 70)

        strata = [
            # 🚀 Layer 1: 发动机核心点火区 (35%) - 真实燃料+空气，微量/中等EGR，高压广温，纯基线无预先自由基
            {
                "name": "Spec_Engine_Core",
                "weight": 0.35,
                "T": [650.0, 1500.0], "scale_T": "linear",
                "P": [5.0, 150.0], "scale_P": "log",
                "phi": [0.5, 2.0], "scale_phi": "linear",
                "X_H2_fuel": [0.0, 0.6],
                "Z_dil": [0.0, 0.25],  # 0% (纯空气) 到 25% EGR
                "f_N2": [0.6, 1.0], "f_H2O": [0.0, 0.25], "f_AR": [0.0, 0.0], "f_HE": [0.0, 0.0],
                "X_OH": [0.0, 0.0],
            },

            # ❄️ Layer 2: 低温慢速氧化与长诱导期 (25%) - 捕捉自燃诱导动力学与中高稀释，严格无初生自由基
            {
                "name": "Spec_Cold_Oxidation",
                "weight": 0.25,
                "T": [300.0, 900.0], "scale_T": "linear",
                "P": [1.0, 120.0], "scale_P": "log",
                "phi": [0.2, 3.0], "scale_phi": "log",
                "X_H2_fuel": [0.0, 1.0],
                "Z_dil": [0.10, 0.80],  # 稀释淬熄区
                "f_N2": [0.5, 0.9], "f_AR": [0.0, 0.3], "f_H2O": [0.0, 0.2],
                "X_OH": [0.0, 0.0],
            },

            # 🔥 Layer 3: 预燃室活性自由基射流区 (20%) - 模拟已燃气卷吸与火核喷射（专门自由基注入层）
            {
                "name": "Spec_Radical_Burst",
                "weight": 0.20,
                "T": [1100.0, 2400.0], "scale_T": "linear",
                "P": [1.0, 150.0], "scale_P": "log",
                "phi": [0.5, 2.0], "scale_phi": "linear",
                "X_H2_fuel": [0.0, 0.8],
                "Z_dil": [0.05, 0.40],
                "f_N2": [0.5, 0.8], "f_H2O": [0.1, 0.4],
                "X_OH": [1e-5, 5e-3], "X_H": [1e-5, 5e-3], "X_O": [1e-5, 5e-3], "X_NO": [1e-5, 5e-3],
                "scale_rad": "log",
            },

            # 🌪️ Layer 4: 极端贫/富燃边界与稀释极限 (20%) - 射流边缘与喷雾死水区
            {
                "name": "Spec_Extreme_Boundaries",
                "weight": 0.20,
                "T": [400.0, 2200.0], "scale_T": "linear",
                "P": [0.5, 120.0], "scale_P": "log",
                "phi": [0.05, 4.5], "scale_phi": "log",
                "X_H2_fuel": [0.0, 1.0],
                "Z_dil": [0.0, 0.95],  # 允许全范围稀释
                "f_N2": [0.2, 0.8], "f_AR": [0.0, 0.5], "f_H2O": [0.0, 0.3],
                "X_OH": [0.0, 0.0],
            },
        ]

        return self._build_dataset(strata, total_pts, base_seed=500, include_radicals=include_radicals)

    def _build_dataset(self, strata: list, total_pts: int, base_seed: int, include_radicals: bool) -> pd.DataFrame:
        dfs = []
        for mech_id, mech_name in enumerate(self.mech_names):
            mech_dfs = []
            pts_accumulated = 0
            local_sample_id = 0

            for i, layer in enumerate(strata):
                n_pts = total_pts - pts_accumulated if i == len(strata) - 1 else int(total_pts * layer["weight"])
                pts_accumulated += n_pts

                seed = base_seed + i * 10
                df_layer = self._generate_composite(n_pts, layer, seed, include_radicals=include_radicals, stratum_name=layer["name"])

                df_layer["stratum_name"] = layer["name"]
                df_layer["sample_id"] = range(local_sample_id, local_sample_id + n_pts)
                local_sample_id += n_pts
                df_layer["mechanism_id"] = mech_id
                df_layer["mechanism_name"] = mech_name
                mech_dfs.append(df_layer)

            dfs.append(pd.concat(mech_dfs, ignore_index=True))

        df_final = pd.concat(dfs, ignore_index=True)
        # 全局混洗逻辑
        df_final = df_final.sort_values(["sample_id", "mechanism_name"]).reset_index(drop=True)
        unique_sample_ids = df_final["sample_id"].unique()
        shuffled_ids = np.random.RandomState(42).permutation(unique_sample_ids)
        df_final["_sort_key"] = df_final["sample_id"].map({old: new for new, old in enumerate(shuffled_ids)})
        df_final = df_final.sort_values(["_sort_key", "mechanism_name"]).drop("_sort_key", axis=1).reset_index(drop=True)

        return df_final


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="LHS Sampler V11 (Physical Species Edition)")
    parser.add_argument("--total_pts", "--n_samples", dest="total_pts", type=int, default=15000, help="Total points per mechanism (default: 15000)")
    parser.add_argument("--output", type=str, default="lhs_samples_Species_Initial_v1.csv", help="Output CSV path")
    parser.add_argument(
        "--mechanism",
        type=str,
        default=None,
        help="生成单个机理的数据，例如 nuig_2024。",
    )
    parser.add_argument(
        "--mechanisms",
        type=str,
        nargs="+",
        default=None,
        help="生成多个机理的数据；与 --mechanism 互斥。",
    )
    parser.add_argument(
        "--list-mechanisms",
        action="store_true",
        help="列出可用机理名称并退出。",
    )
    args = parser.parse_args()

    if args.list_mechanisms:
        print("\n".join(SUPPORTED_MECHANISMS))
        raise SystemExit(0)

    try:
        selected_mechanisms = normalize_mechanism_selection(
            mechanism=args.mechanism,
            mechanisms=args.mechanisms,
        )
    except ValueError as exc:
        parser.error(str(exc))

    local_mechanisms = {m: {} for m in selected_mechanisms}
    sampler = UniversalLHSSampler_Species(local_mechanisms)

    df_species_init = sampler.generate_species_database(total_pts=args.total_pts)

    output_path = Path(args.output).resolve()
    df_species_init.to_csv(output_path, index=False)

    print(f"[SUCCESS] Saved Species Initial States: {len(df_species_init)} rows to {output_path}")
    print("\n层级分布验证：")
    print(df_species_init["stratum_name"].value_counts().sort_index())
