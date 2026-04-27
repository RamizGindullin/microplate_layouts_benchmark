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
                    lgnd.legend_handles[2].set_sizes([30])
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
    if plate_type is not None and plate_type.get("requires_layout_update", False):
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

    plate_df = pd.DataFrame(plate_array).stack(future_stack=True)future_stack=True.reset_index()
    plate_df.columns = ["Rows", "Columns", "Intensity"]
    controls_df = pd.DataFrame(layout).stack(future_stack=True).reset_index()
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


def _run_one_plate(
    plate_type,
    layout_file,
    plate_content,
    et,
    current_e,
    compounds,
    concentrations,
    replicates,
    limit,
    df_params,
    expected_noise=0.01,
):
    """Run one plate experiment and return (abs_results, rel_results, plate_residuals).

    Inner loop body extracted from the original notebook's ``function()`` helper
    (dose-response-experiments.ipynb, astra-uu-se/COMPD, cell 3).

    Column 0 of every output array is ``plate_type["display_type"]`` (e.g. "COMPD",
    "PLAID", "Random") rather than the raw layout filename.  This mirrors the
    screening benchmark, where ``display_type`` is written directly into the CSV so
    that ``_classify_dose_response_layout_series`` in utilities.py can validate it
    without needing to parse filenames.
    """
    layout_dir = plate_type["dir"]
    compounds_array = df_params.index.to_numpy()
    lost_rows = limit["to"] - limit["from"]
    display_type = plate_type["display_type"]

    fit_table = plate_curves_after_error(
        layout_dir,
        layout_file,
        plate_content.copy(),
        expected_noise,
        et["error_function"],
        et["error"],
        plate_type["error_correction"],
        lose_from_row=limit["from"],
        lose_to_row=limit["to"],
        plate_type=plate_type,
        compounds=compounds,
        concentrations=concentrations,
        replicates=replicates,
    )

    obtained_absolute_ic50 = IC50(
        fit_table["b"], fit_table["c"], fit_table["d"], fit_table["e"]
    )

    res_array = np.concatenate(fit_table["residuals"].to_numpy())
    true_res_array = np.concatenate(fit_table["true_residuals"].to_numpy())
    res_size = len(res_array)

    # Residuals CSV columns (7):
    # ["layout", "error_type", "Error", "E", "rows lost", "residuals", "true_residuals"]
    # "E" = current_e (the EC50 bucket for this batch), kept for schema compatibility.
    plate_residuals = np.vstack([
        np.full(res_size, display_type),
        np.full(res_size, et["type"]),
        np.full(res_size, et["error"]),
        np.full(res_size, current_e),
        np.full(res_size, lost_rows),
        res_array,
        true_res_array,
    ])

    plate_rel_ic50 = np.absolute(
        np.subtract(np.log10(df_params["e"]), np.log10(fit_table["e"]))
    )
    plate_abs_ic50 = np.absolute(
        np.subtract(
            np.log10(df_params["abs IC50"]),
            np.log10(obtained_absolute_ic50),
        )
    )

    display_type_array = np.full(len(compounds_array), display_type)
    error_type_array  = np.full(len(compounds_array), et["type"])
    error_array       = np.full(len(compounds_array), et["error"])
    e_array           = np.full(len(compounds_array), current_e)
    r_lost_array      = np.full(len(compounds_array), lost_rows)

    # IC50/d_max CSV columns (16):
    # ["layout", "compound", "MSE", "error type", "Error", "E", "rows lost",
    #  "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"]
    rel_results = np.vstack([
        display_type_array,
        compounds_array,
        plate_rel_ic50.to_numpy(),
        error_type_array,
        error_array,
        e_array,
        r_lost_array,
        fit_table["r2_score"].to_numpy(),
        df_params["b"].to_numpy(), df_params["c"].to_numpy(),
        df_params["d"].to_numpy(), df_params["e"].to_numpy(),
        fit_table["b"].to_numpy(), fit_table["c"].to_numpy(),
        fit_table["d"].to_numpy(), fit_table["e"].to_numpy(),
    ])
    abs_results = np.vstack([
        display_type_array,
        compounds_array,
        plate_abs_ic50.to_numpy(),
        error_type_array,
        error_array,
        e_array,
        r_lost_array,
        fit_table["r2_score"].to_numpy(),
        df_params["b"].to_numpy(), df_params["c"].to_numpy(),
        df_params["d"].to_numpy(), df_params["e"].to_numpy(),
        fit_table["b"].to_numpy(), fit_table["c"].to_numpy(),
        fit_table["d"].to_numpy(), fit_table["e"].to_numpy(),
    ])

    return abs_results, rel_results, plate_residuals


def full_dose_response_evaluation(
    plate_types_location,
    error_types,
    e_from=1,
    e_to=100,
    e_step=5,
    compounds=48,
    concentrations=6,
    replicates=1,
    dilution=18,
    error_nl=0.055,
    lose_rows_from=1,
    lose_rows_to=2,
    today="test-",
    id_text="",
    data_directory="generated-data/dose-response/",
    expected_noise=0.01,
):
    """Run the full dose-response benchmark for one scenario and write result CSVs.

    Ported from ``dose-response-experiments.ipynb`` (astra-uu-se/COMPD, cell 3).
    The original notebook's inner ``function()`` helper is now ``_run_one_plate``.

    Column 0 of every written CSV row is ``display_type`` (e.g. "COMPD", "PLAID",
    "Random"), mirroring ``run_screening_benchmark.py`` which writes
    ``plate_type.display_type`` directly rather than raw layout filenames.
    ``_prepare_dose_response_results_frame`` in ``utilities.py`` expects this and
    validates it via ``_classify_dose_response_layout_series``.

    Parameters
    ----------
    plate_types_location : list[dict]
        Each dict must have keys: ``type``, ``display_type``, ``dir``, ``regex``,
        ``error_correction``.  Build with ``benchmark_common.dose_response_plate_types()``.
    error_types : list[dict]
        Each dict must have keys: ``type``, ``error_function``, ``error``.
    e_from, e_to, e_step : int
        Range of EC50 values to sweep (``range(e_from, e_to, e_step)``).
    today : str
        Date/tag prefix (e.g. ``"20250706-"``).
    id_text : str
        Scenario suffix appended after ``today`` in filenames.
    data_directory : str
        Directory where CSV files are written; created if absent.
    """
    import re as _re

    os.makedirs(data_directory, exist_ok=True)

    tag = today + id_text if id_text else today

    abs_path = os.path.join(
        data_directory,
        f"absolute_ic50_data-{compounds}-{concentrations}-dil{dilution}"
        f"-{replicates}-{error_nl}-{tag}.csv",
    )
    rel_path = os.path.join(
        data_directory,
        f"relative_ic50_data-{compounds}-{concentrations}-dil{dilution}"
        f"-{replicates}-{error_nl}-{tag}.csv",
    )
    res_path = os.path.join(
        data_directory,
        f"residuals-{compounds}-{concentrations}-dil{dilution}"
        f"-{replicates}-{error_nl}-{tag}.csv",
    )

    with (
        open(abs_path, "a") as abs_f,
        open(rel_path, "a") as rel_f,
        open(res_path, "a") as res_f,
    ):
        for current_e in range(e_from, e_to, e_step):
            print(f"\nTesting compounds with e in range {current_e}-{current_e + e_step}:")

            params = generate_compound_curves(
                compounds, concentrations, dilution, current_e
            )
            df_params = pd.DataFrame.from_dict(params).set_index("compound")
            df_params["abs IC50"] = IC50(
                df_params["b"], df_params["c"], df_params["d"], df_params["e"]
            )

            plate_content = generate_plate_content(
                dose_response_params=params, replicates=replicates
            )

            for plate_type in plate_types_location:
                print(f"  Using {plate_type['display_type']} layouts...")
                layout_dir = plate_type["dir"]
                try:
                    layouts = os.listdir(layout_dir)
                except FileNotFoundError:
                    print(f"    WARNING: layout directory not found: {layout_dir!r}")
                    continue

                for layout_file in layouts:
                    if _re.search(plate_type["regex"], layout_file) is None:
                        continue

                    for et in error_types:
                        # Original notebook iterates range(1, 2) — i.e. only lost_rows=1.
                        for lost_rows in range(1, 2):
                            limits = [{"from": 16 - lost_rows, "to": 16}]
                            for limit in limits:
                                abs_results, rel_results, plate_residuals = _run_one_plate(
                                    plate_type=plate_type,
                                    layout_file=layout_file,
                                    plate_content=plate_content.copy(),
                                    et=et,
                                    current_e=current_e,
                                    compounds=compounds,
                                    concentrations=concentrations,
                                    replicates=replicates,
                                    limit=limit,
                                    df_params=df_params,
                                    expected_noise=expected_noise,
                                )
                                np.savetxt(abs_f, abs_results.T, delimiter=",", fmt="%s")
                                np.savetxt(rel_f, rel_results.T, delimiter=",", fmt="%s")
                                np.savetxt(res_f, plate_residuals.T, delimiter=",", fmt="%s")