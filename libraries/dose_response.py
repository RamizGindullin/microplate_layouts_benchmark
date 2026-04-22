import os

import numpy as np
import pandas as pd
import scipy.optimize as opt
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import r2_score

import libraries.utilities as util
import libraries.disturbances as dt


def ll4(x, b, c, d, e):
    """Four-parameter log-logistic dose–response model (LL.4 from R drc).

    Parameters
    ----------
    b : Hill slope
    c : minimum response
    d : maximum response
    e : EC50
    """
    return c + (d - c) / (1 + np.exp(b * (np.log(x) - np.log(e))))


def pDose(x):
    """Convert concentration to -log10(µM) scale used in drug discovery."""
    return -np.log10(1e-5 * x)


def IC50(b, c, d, e):
    """Compute absolute IC50 from LL.4 parameters."""
    return np.exp(np.log((((d - c) / (50 - c)) - 1) * (e ** b)) / b)


def fill_plate(
    layout,
    plate_content,
    neg_control_value=100,
    num_compounds=36,
    concentrations=4,
    replicates=2,
    expected_noise=0.25,
):
    num_rows, num_columns = layout.shape
    plate = np.full((num_rows, num_columns), 0.0)
    neg_control = np.max(layout)

    for row_index in range(num_rows):
        for col_index in range(num_columns):
            cell = layout[row_index][col_index]
            if cell == neg_control:
                plate[row_index][col_index] = (
                    neg_control_value + expected_noise * (np.random.random() - 0.5)
                )
            elif cell > 0:
                plate[row_index][col_index] = (
                    plate_content.iloc[neg_control - cell - 1].response
                    + abs(expected_noise * (np.random.random() - 0.5))
                )
    return plate


def generate_plate_content(dose_response_params, replicates):
    dr_data = []
    for _ in range(replicates):
        for curve in dose_response_params:
            cur_data = pd.DataFrame(
                data={
                    "compound": curve["compound"],
                    "dose": curve["startDose"]
                    / np.power(curve["dilution"], range(curve["nDose"])),
                }
            )
            cur_data["logDose"] = pDose(cur_data.dose)
            cur_data["response"] = cur_data.dose.apply(
                lambda x: ll4(x, *[curve[i] for i in ["b", "c", "d", "e"]])
            )
            rep_data = cur_data.copy()
            dr_data.append(rep_data)
    return pd.concat(dr_data)


def collect_plate_results(layout, plate):
    num_rows, num_columns = layout.shape
    neg_control = np.max(layout)
    results = np.full(neg_control - 1, float("nan"))

    for row_index in range(num_rows):
        for col_index in range(num_columns):
            cell = layout[row_index][col_index]
            # Fix: was `0 < cell & cell < neg_control` (bitwise precedence bug)
            if 0 < cell < neg_control:
                results[neg_control - cell - 1] = plate[row_index][col_index]
    return results


def fit_data(
    result_data,
    response_column,
    result_column,
    df_params=None,
    layout_type="",
    neg_control_values=None,
    output_dir=".",
):
    """Fit LL.4 dose–response curves to plate data.

    Parameters
    ----------
    output_dir : str
        Directory for debug curve PNGs (only written when df_params is not None).
        Replaces the former os.chdir() approach.
    """
    compound_data = result_data.groupby("compound")
    fit_data_list = []

    for name, group_t in compound_data:
        warnings.filterwarnings(
            "ignore", message="divide by zero encountered in true_divide"
        )
        group = group_t[np.logical_not(np.isnan(group_t[result_column]))]

        if len(group) > 0:
            p0 = [
                0.5,
                min(group[result_column]),
                max(group[result_column]),
                np.median(group.dose),
            ]
            low_b = [0, -np.inf, min(group[result_column]), 0]
            up_b = [np.inf, max(group[result_column]), np.inf, np.inf]

            try:
                if neg_control_values is None:
                    fit_coefs, _ = opt.curve_fit(
                        ll4,
                        group.dose,
                        group[result_column],
                        p0,
                        max_nfev=10_000_000,
                        bounds=(low_b, up_b),
                    )
                else:
                    neg_dose = np.min(group.dose) / 100
                    mean_neg_ctrl = [
                        np.mean(
                            [
                                neg_control_values[4 * i],
                                neg_control_values[4 * i + 1],
                                neg_control_values[4 * i + 2],
                                neg_control_values[4 * i + 3],
                            ]
                        )
                        for i in range(4)
                    ]
                    neg_dose_array = np.full_like(
                        mean_neg_ctrl, neg_dose, dtype=np.float64
                    )
                    fit_coefs, _ = opt.curve_fit(
                        ll4,
                        np.concatenate([group.dose, neg_dose_array]),
                        np.concatenate([group[result_column], mean_neg_ctrl]),
                        p0,
                        max_nfev=10_000_000,
                        bounds=(low_b, up_b),
                    )

                resids = group[result_column] - group.dose.apply(
                    lambda x: ll4(x, *fit_coefs)
                )
                true_resids = group[response_column] - group[result_column]
                r2s = r2_score(
                    group[result_column],
                    group.dose.apply(lambda x: ll4(x, *fit_coefs)),
                )
                cur_fit = dict(
                    list(zip(["b", "c", "d", "e"], fit_coefs))
                    + [("r2_score", r2s)]
                    + [("residuals", resids ** 2), ("true_residuals", true_resids ** 2)]
                )

            except Exception:
                print("Curve fit failed (too many iterations or convergence error)")
                cur_fit = dict(
                    [("b", float("nan")), ("c", float("nan")),
                     ("d", float("nan")), ("e", float("nan")),
                     ("r2_score", -np.inf)]
                    + [("residuals", [np.inf]), ("true_residuals", [np.inf])]
                )
        else:
            cur_fit = dict(
                [("b", float("nan")), ("c", float("nan")),
                 ("d", float("nan")), ("e", float("nan")),
                 ("r2_score", -np.inf)]
                + [("residuals", [np.inf]), ("true_residuals", [np.inf])]
            )

        cur_fit["compound"] = name

        # Debug plotting — only active when df_params is passed.
        # Writes PNGs to output_dir (explicit), not cwd.
        if df_params is not None and not np.isnan(cur_fit["b"]):
            true_curve = df_params.iloc[[name]]
            print(f"\nPlotting curve for compound {name}")
            print(f"R²: {r2s}")
            print(f"True IC50: {true_curve['e'].values}")
            print(f"Estimated IC50: {cur_fit['e']}")
            print(
                f"Abs log error: "
                f"{abs(np.log10(true_curve['e'].values) - np.log10(cur_fit['e']))}"
            )

            ref_dose = np.linspace(
                min(result_data.logDose) * 0.9,
                max(result_data.logDose) * 1.1,
                256,
            )
            ref_dose_conc = 10 ** -ref_dose

            sns.lmplot(
                x="logDose",
                y="results",
                data=result_data[result_data["compound"] == name],
                fit_reg=False,
                height=2.75,
                scatter_kws={"s": 6},
                palette=["#4DBBD599"],
            )
            plt.plot(
                ref_dose,
                [ll4(i, *[true_curve[i].values[0] for i in ["b", "c", "d", "e"]])
                 for i in ref_dose_conc],
                color="#DC000099",
                label="Original",
                linewidth=2,
            )
            plt.plot(
                ref_dose,
                [ll4(i, *[cur_fit[i] for i in ["b", "c", "d", "e"]])
                 for i in ref_dose_conc],
                color="#3C5488FF",
                label="Estimated",
                linewidth=2,
            )
            plt.ylim(-5, 135)

            if neg_control_values is not None:
                sns.regplot(
                    x=pDose(neg_dose_array),
                    y=mean_neg_ctrl,
                    scatter=True,
                    fit_reg=False,
                    color="orange",
                    marker="*",
                    scatter_kws={"s": 6},
                    label="Controls",
                )

            plt.ylabel("Response (%)", fontsize=10)
            plt.xlabel("Log(Concentration)", fontsize=10)
            lgnd = plt.legend(loc="lower right", fontsize=10)
            lgnd.legend_handles[2].set_sizes([30])

            out_path = os.path.join(
                output_dir, f"{layout_type}_compound_{name}-right-half.png"
            )
            plt.savefig(out_path, bbox_inches="tight", dpi=1200)
            plt.close()

        fit_data_list.append(cur_fit)

    return pd.DataFrame(fit_data_list).set_index("compound")


def fit_data_min_req(result_data, response_column, result_column):
    compound_data = result_data.groupby(["compound"])
    fit_data_list = []

    for name, group_t in compound_data:
        group = group_t[np.logical_not(np.isnan(group_t[result_column]))]

        if len(group) > 0:
            p0 = [
                0.5,
                min(group[result_column]),
                max(group[result_column]),
                np.median(group.dose),
            ]
            low_b = [-np.inf, 0, -np.inf, 0]
            up_b = [np.inf, np.inf, np.inf, np.inf]

            # Fix: was `maxfev` (SciPy < 1.0 spelling); correct kwarg is `max_nfev`
            fit_coefs, _ = opt.curve_fit(
                ll4,
                group.dose,
                group[result_column],
                p0,
                max_nfev=10_000_000,
                bounds=(low_b, up_b),
            )
            resids = group[result_column] - group.dose.apply(
                lambda x: ll4(x, *fit_coefs)
            )
            true_resids = group[response_column] - group[result_column]
            cur_fit = dict(
                list(zip(["b", "c", "d", "e"], fit_coefs))
                + [("residuals", resids ** 2), ("true_residuals", true_resids ** 2)]
            )
        else:
            cur_fit = dict(
                [("b", float("nan")), ("c", float("nan")),
                 ("d", float("nan")), ("e", float("nan"))]
                + [("residuals", [np.inf]), ("true_residuals", [np.inf])]
            )

        cur_fit["compound"] = name
        fit_data_list.append(cur_fit)

    return pd.DataFrame(fit_data_list).set_index("compound")


def plate_curves_after_error(
    layout_dir,
    layout_file,
    plate_content,
    expected_noise,
    error_function,
    error,
    normalization_function,
    min_dist=0,
    lose_from_row=0,
    lose_to_row=0,
    df_params=None,
    plate_type=None,
    compounds=None,
    concentrations=None,
    replicates=None,
    output_dir=".",
):
    """Run one plate experiment and fit dose–response curves.

    Parameters
    ----------
    output_dir : str
        Directory where debug PNGs are saved when df_params is not None.
        Previously the caller was expected to os.chdir() to the desired
        directory before calling this function; output_dir replaces that.
    """
    plate_content, neg_control_values = _run_experiment(
        layout_dir,
        layout_file,
        plate_content,
        expected_noise,
        error_function,
        error,
        normalization_function,
        min_dist,
        lose_from_row,
        lose_to_row,
        plate_type,
        compounds,
        concentrations,
        replicates,
    )
    return fit_data(
        plate_content,
        "response",
        "results",
        neg_control_values=neg_control_values,
        df_params=df_params,
        layout_type=layout_file,
        output_dir=output_dir,
    )


def _run_experiment(
    layout_dir,
    layout_file,
    plate_content,
    expected_noise,
    error_function,
    error,
    normalization_function,
    min_dist,
    lose_from_row,
    lose_to_row,
    plate_type=None,
    compounds=None,
    concentrations=None,
    replicates=None,
):
    layout = np.load(layout_dir + layout_file)
    if plate_type is not None and plate_type["type"] == "COMPD":
        layout = update_compd_layout(layout, compounds, concentrations, replicates)

    neg_control_id = np.max(layout)
    plate = fill_plate(layout, plate_content, neg_control_value=100, expected_noise=expected_noise)
    plate = error_function(plate, error)
    plate = dt.lose_rows(plate, lose_from_row, lose_to_row)
    layout = dt.lose_rows(layout, lose_from_row, lose_to_row)
    plate = normalization_function(plate, layout, neg_control_id, min_dist=min_dist)

    mean_neg_ctrl = mean_controls(plate, layout, neg_control_id)
    results = collect_plate_results(layout, plate)
    plate_content["results"] = results
    return plate_content, mean_neg_ctrl


# Private alias kept for any callers that may reference the old dunder name.
# Will be removed in the layout-registry refactor pass.
__run_experiment = _run_experiment


def update_compd_layout(plate_old, compounds, concentrations, replicates):
    """Re-encode a COMPD layout from per-concentration indexing to per-well indexing."""
    num_rows, num_cols = plate_old.shape
    plate_new = np.zeros((num_rows, num_cols), dtype=int)

    max_control_old = 1 + compounds * concentrations
    max_control_new = 1 + compounds * concentrations * replicates

    list_coordinates = [[] for _ in range(compounds * concentrations)]

    for r in range(num_rows):
        for c in range(num_cols):
            if plate_old[r][c] == max_control_old:
                plate_new[r][c] = max_control_new
            elif plate_old[r][c] > 0:
                list_coordinates[plate_old[r][c] - 1].append((r, c))

    for i in range(compounds * concentrations):
        for j, (r, c) in enumerate(list_coordinates[i]):
            plate_new[r][c] = int(1 + i + j * compounds * concentrations)

    return plate_new


def plate_min_curves_after_error(
    layout_dir,
    layout_file,
    plate_content,
    expected_noise,
    error_function,
    error,
    normalization_function,
    min_dist,
    lose_from_row=0,
    lose_to_row=0,
):
    plate_content, _ = _run_experiment(
        layout_dir,
        layout_file,
        plate_content,
        expected_noise,
        error_function,
        error,
        normalization_function,
        min_dist,
        lose_from_row,
        lose_to_row,
    )
    return fit_data_min_req(plate_content, "response", "results")


def mean_controls(plate_array, layout, control_id):
    control_locations = util.get_controls_layout(layout, neg_control=control_id)
    if control_locations.sum() < 1:
        return float("nan")

    plate_df = pd.DataFrame(plate_array).stack().reset_index()
    plate_df.columns = ["Rows", "Columns", "Intensity"]
    controls_df = pd.DataFrame(layout).stack().reset_index()
    controls_df.columns = ["Rows", "Columns", "Type"]
    data = pd.merge(plate_df, controls_df, how="left", on=["Rows", "Columns"])
    return data.loc[data["Type"] == control_id, ["Intensity"]].to_numpy().reshape((-1,))


def generate_compound_curves(
    compounds,
    concentrations,
    dilution,
    low_e,
    slopes=None,
    start_dose=10000,
    step=5,
):
    if slopes is None:
        slopes = [0.5, 1, 1.5]
    return [
        {
            "compound": i,
            "b": slopes[i % len(slopes)],
            "c": 0,
            "d": 100,
            "e": low_e + step * np.random.random(),
            "startDose": start_dose,
            "nDose": concentrations,
            "dilution": dilution,
        }
        for i in range(compounds)
    ]