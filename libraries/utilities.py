import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import re
from statannotations.Annotator import Annotator # to add p-values to plots
from scipy import stats
from random import randrange
from sklearn import metrics
from moepy import lowess
from scipy.interpolate import interp1d
from benchmark_common import (
    DOSE_RESPONSE_LAYOUT_BOX_PAIRS_BY_REPLICATE,
    DOSE_RESPONSE_LAYOUT_BOX_PAIRS,
    DOSE_RESPONSE_LAYOUT_ORDER,
    DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER,
    DOSE_RESPONSE_LAYOUT_SPECS,
    SCREENING_LAYOUT_BOX_PAIRS,
    SCREENING_LAYOUT_ORDER,
    SCREENING_LAYOUT_SPECS,
)


# ---------------------------------------------------------------------------
# Layout classification (registry-driven)
# ---------------------------------------------------------------------------

def _classify_dose_response_layout_series(layout_series):
    """Validate a pandas Series of dose-response display_type values from CSVs.

    Raises ValueError on unrecognised values so CSV schema mismatches surface
    immediately rather than silently corrupting plots.
    """
    known = {s.display_type for s in DOSE_RESPONSE_LAYOUT_SPECS}
    def _map(v):
        if v in known:
            return v
        raise ValueError(
            f"Unrecognised dose-response layout {v!r}. "
            f"Expected one of {sorted(known)}. "
            "Check that benchmark scripts write display_type from "
            "benchmark_common.DOSE_RESPONSE_LAYOUT_SPECS into result CSVs."
        )
    return layout_series.map(_map)


def _classify_screening_layout_series(layout_series):
    """Validate a pandas Series of screening display_type values from CSVs.

    No longer called for active classification (CSVs carry a display_type column).
    Retained as an early-warning validator: call on df["display_type"] after
    pd.read_csv to surface any CSV schema mismatches immediately.
    """
    known = {s.display_type for s in SCREENING_LAYOUT_SPECS}
    def _map(v):
        if v in known:
            return v
        raise ValueError(
            f"Unrecognised screening layout {v!r}. "
            f"Expected one of {sorted(known)}. "
            "Check that benchmark scripts write display_type from "
            "benchmark_common.SCREENING_LAYOUT_SPECS into result CSVs."
        )
    return layout_series.map(_map)


def _apply_boxplot_annotations(ax, data, x, y, pairs, order=None, hue=None, hue_order=None, plot='boxplot'):
    annotator = Annotator(ax, pairs, data=data, x=x, y=y, order=order, hue=hue, hue_order=hue_order, plot=plot)
    annotator.configure(test='Mann-Whitney', text_format='star', loc='outside', line_width=1)
    annotator.apply_and_annotate()
    return annotator


def _concat_dose_response_frames(data_1rep, data_2rep, data_3rep, value_name):
    frame1 = pd.DataFrame(data_1rep[1:, 3:].T, columns=data_1rep[0, 3:])
    frame1["Replicate"] = 1
    frame2 = pd.DataFrame(data_2rep[1:, 3:].T, columns=data_2rep[0, 3:])
    frame2["Replicate"] = 2
    frame3 = pd.DataFrame(data_3rep[1:, 3:].T, columns=data_3rep[0, 3:])
    frame3["Replicate"] = 3
    df = pd.concat([frame1, frame2, frame3], ignore_index=True)
    return df.melt(id_vars=["Replicate"], var_name="Layout type", value_name=value_name)


def _coerce_numeric_column(df, column_name):
    df = df.copy()
    df[column_name] = pd.to_numeric(df[column_name], errors="coerce")
    return df



def _stack_replicate_results_frames(replicate_arrays):
    """Return a single long DataFrame from an iterable of replicate result arrays.

    replicate_arrays should be an iterable of np.ndarray objects with the
    standard dose-response result columns. The first replicate is labelled 1,
    the second 2, and so on, regardless of how many are provided.
    """
    columns = [
        "layout", "compound", "MSE", "error type", "Error", "E", "rows lost",
        "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e",
    ]
    frames = []
    for idx, arr in enumerate(replicate_arrays, start=1):
        frame = pd.DataFrame(arr, columns=columns)
        frame.insert(0, "replicates", idx)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["replicates"] + columns)
    df = pd.concat(frames, ignore_index=True)
    df["layout"] = _classify_dose_response_layout_series(df["layout"])
    return df


def _stack_replicate_residuals_frames(replicate_arrays):
    """Return a long residuals DataFrame from an iterable of replicate arrays."""
    columns = [
        "layout", "error_type", "Error", "E", "rows lost", "residuals",
        "true_residuals",
    ]
    frames = []
    for idx, arr in enumerate(replicate_arrays, start=1):
        frame = pd.DataFrame(arr, columns=columns)
        frame.insert(0, "replicates", idx)
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["replicates"] + columns)
    df = pd.concat(frames, ignore_index=True)
    df["layout"] = _classify_dose_response_layout_series(df["layout"])
    df["residuals"] = pd.to_numeric(df["residuals"], errors="coerce")
    df["true_residuals"] = pd.to_numeric(df["true_residuals"], errors="coerce")
    df["rows lost"] = pd.to_numeric(df["rows lost"], errors="coerce")
    # Replace inf from failed/diverged fits with NaN so downstream
    # statistics (t-test, mean) are not contaminated
    df["true_residuals"] = df["true_residuals"].replace([np.inf, -np.inf], np.nan)
    df["residuals"] = df["residuals"].replace([np.inf, -np.inf], np.nan)
    return df


def _prepare_dose_response_results_frame(data_1rep, data_2rep, data_3rep):
    """Backward-compatible wrapper using the generic replicate stacker.

    Existing callers provide exactly three replicate arrays; this wrapper keeps
    that API stable while allowing the internal implementation to support any
    number of replicates via _stack_replicate_results_frames.
    """
    return _stack_replicate_results_frames([data_1rep, data_2rep, data_3rep])



def _prepare_dose_response_residuals_frame(residuals_1rep, residuals_2rep, residuals_3rep):
    """Backward-compatible wrapper using the generic residuals stacker."""
    return _stack_replicate_residuals_frames(
        [residuals_1rep, residuals_2rep, residuals_3rep]
    )


def plot_plate(plate_array, title="", mask=None, filename=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.xaxis.tick_top()
    plt.title(title, fontsize = 15) 
    sns.heatmap(plate_array,linewidth=0.3,square=True,mask=mask,vmin=vmin,vmax=vmax)
    if filename:
        fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)
    
    
def get_controls_layout(layout, neg_control = None):
    num_rows, num_columns = layout.shape
    
    if neg_control is None:
        control = np.max(layout)
    else:
        control = neg_control

    control_layout = np.full((num_rows, num_columns), 0)

    for row_i in range(num_rows):
        for col_i in range(num_columns):
            if layout[row_i][col_i] == control:
                control_layout[row_i][col_i] = 1

    return(control_layout)


# TODO(dead-code): _shape_layout is not called by any benchmark script.
def _shape_layout(layout, num_rows, num_columns, size_empty_edge):
    layout = np.reshape(layout,(-1, num_columns-2*size_empty_edge))
    
    if size_empty_edge > 0:
        vertical_edge = np.reshape(np.full(size_empty_edge*(num_rows-2*size_empty_edge),0), (-1,size_empty_edge))
                         
        layout = np.hstack((vertical_edge,layout))
    
        layout = np.hstack((layout,vertical_edge))
    
        horizontal_edge = np.reshape(np.full(size_empty_edge*num_columns,0), (-1,num_columns))
    
        layout = np.vstack((horizontal_edge,layout))
    
        layout = np.vstack((layout,horizontal_edge))
    
    return layout


# TODO(dead-code): check_duplicated_layouts is not called by any benchmark script.
def check_duplicated_layouts(layout_dir = 'screening_manual_layouts/'):

    layouts = sorted(os.listdir(layout_dir))
    duplicates = False
    
    for layout_file in layouts:
        match = re.search('plate_layout_*',layout_file)

        if match == None:
            continue

        layout_1 = np.load(layout_dir+layout_file)  
            
        for layout_file_2 in layouts:
            match = re.search('plate_layout_*',layout_file_2)

            if (match == None) or (layout_file == layout_file_2):
                continue
                
            layout_2 = np.load(layout_dir+layout_file_2)  

            if (layout_1 == layout_2).all():
                print("Layouts "+layout_file+" "+layout_file_2+" are the same")
                duplicates = True
    
    if not duplicates:
        print('There are no duplicates')
        
    return duplicates


# TODO(dead-code): plot_well_series_precomputed_normalization is not called by any
# benchmark script. Retained for potential external use.
def plot_well_series_precomputed_normalization(plate_array, norm_plate, layout, neg_control_id, pos_control_id, order=0, vmin=None, vmax=None, filename=None):    
    ''' Creates the well series plots used for the PLAID bioseminar presentation
    
    Args:
        plate_array: np array containing the raw data from the experiments
        norm_plate: np array containing the corrected/normalized data from the experiments
        layout: an np array containing the layout used in plate_array
        neg_control_id: id (number) of the negative controls such that if layout[i][j] == neg_control_id then the i,j well 
            contains a negative control
        norm_function: function used to normalize the plate, for example nrm.normalize_plate_lowess_2d
    '''
    
    num_rows, num_columns = layout.shape
    
    ### Reformat original input data
    plate_df = pd.DataFrame(plate_array)
    
    intensity_df = plate_df.stack(future_stack=True).reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]
    
    types_df = pd.DataFrame(layout).stack(future_stack=True).reset_index()
    types_df.columns = ["Rows","Columns","Type"]
    
    combined_df = pd.merge(intensity_df, types_df,  how='left', on=['Rows','Columns'])
    combined_df['Rows'] += 1
    combined_df['Columns'] += 1
    
    
    ### Reformat normalized/corrected plate
    n_plate_df = pd.DataFrame(norm_plate)
    
    n_intensity_df = n_plate_df.stack(future_stack=True).reset_index()
    n_intensity_df.columns = ["Rows","Columns","Intensity"]
    
    n_combined_df = pd.merge(n_intensity_df, types_df,  how='left', on=['Rows','Columns'])
    n_combined_df['Rows'] += 1
    n_combined_df['Columns'] += 1
    
    
    ### Plot heatmap before normalization
    unstack_df = combined_df[["Rows","Columns","Intensity"]].copy()
    unstacked_df = pd.pivot_table(unstack_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc='sum')

    
    ### Plot heatmap after normalization
    unstack_adjusted_df = n_combined_df[["Rows","Columns","Intensity"]].copy()
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc='sum')
    
    
    ### Plotting well series with original and normalized data
    fig, ax = plt.subplots(figsize=(11,7))
    
    ax.set(xlim=(0,num_columns+1))
    
    ## Add all the samples in the original data (except controls)
    ax = sns.regplot(data=combined_df[(combined_df.Type!=pos_control_id) & (combined_df.Type!=neg_control_id)], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, scatter_kws={"color":"orange","alpha":0.3})
    
    ## Positive controls from the raw/original data
    ax = sns.regplot(data=combined_df[combined_df.Type==pos_control_id], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, scatter_kws={"color":"orange","alpha":0.3})

    # Add negative controls from the raw/original data
    ax = sns.regplot(data=combined_df[combined_df.Type==neg_control_id], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, marker='*',scatter_kws={"color":"purple"}, truncate=False, order=order)
    
    # Add normalized data
    ax = sns.regplot(data=n_combined_df, x="Columns", y="Intensity", x_jitter=0.3, fit_reg=True, marker='x',scatter_kws={"color":"blue"}, truncate=False, order=order)
    
    ax.set_xticks(range(1,num_columns+1))
    
    if filename:
        fig.savefig(filename)
    
    
def plot_barplot_residuals_data(residuals_1rep, residuals_2rep, residuals_3rep, fig_name, y_max=None, leg_loc="lower right", leg_ncol=3, leg_fontsize=8, pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]], hue_order=DOSE_RESPONSE_RESIDUALS_LAYOUT_ORDER, box_pairs=None, fig_dir=''):
    """ Plots residual plots for dose response experiments as in the manuscript. """
    residuals_df = _prepare_dose_response_residuals_frame(
        residuals_1rep, residuals_2rep, residuals_3rep
    )
    residuals_df = residuals_df.rename(columns={"replicates": "Replicates", "layout": "Layout type", "true_residuals": "Residuals"})
    residuals_df["Replicates"] = residuals_df["Replicates"].astype(int)
    residuals_df = residuals_df[residuals_df['rows lost'] <= 1]
    if box_pairs is None:
        box_pairs = DOSE_RESPONSE_LAYOUT_BOX_PAIRS_BY_REPLICATE
    
    tmp = residuals_df[residuals_df["Replicates"]==1]
    
    fig,ax = plt.subplots(figsize=(4,3))
    palette = [spec.residuals_color for spec in DOSE_RESPONSE_LAYOUT_SPECS]
    
    sns.barplot(
        data=residuals_df,
        x='Replicates', y='Residuals',
        hue='Layout type',
        ax=ax,
        palette=sns.color_palette(palette, len(hue_order)),
        hue_order=hue_order,
    )
    plt.ylabel("Mean residuals", fontsize=10)
    
    if not y_max is None:
        ax.set_ylim(0, y_max)
    
    annotator = Annotator(
        ax,
        data=residuals_df,
        x='Replicates', y='Residuals',
        pairs=box_pairs,
        order=[1, 2, 3],
        hue='Layout type',
        hue_order=hue_order,
        plot='barplot',
    )
    annotator.configure(
        test='t-test_ind',
        text_format='star',
        loc='inside',
        pvalue_thresholds=pvalue_thresholds,
        text_offset=0,
        line_offset=0.05,
        fontsize = 6,
    )
    annotator.apply_and_annotate()
    
    #ax.legend(loc=leg_loc, ncol=leg_ncol, fontsize=leg_fontsize)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.58, 1.01),
        ncol=len(hue_order), fontsize=8,
        borderaxespad=0
    )
    fig.savefig(fig_dir + fig_name + '.png', bbox_inches='tight', dpi=300)
    plt.close(fig)

def plot_barplot_replicate_data(
    data_1rep, data_2rep, data_3rep,
    fig_name="", fig_dir="", fig_type="d_diff",
    y_max=None, leg_ncol=1, leg_loc="best", leg_fontsize=8,
    pvalue_thresholds=None,
):
    """ Plots barplots for absolute and relative EC50/IC50 for dose response experiments as in the manuscript. 
        It also plots d_diff, that is, the average difference between the expected and obtained maximum (d) of the
        dose-response 4PL sigmoid curve.
    
    Args:
        data_1rep:
        data_2rep:
        data_3rep:
        fig_name: string added to the image file name.
        fig_type: string. "relic50" for relative IC50, "absic50" for absolute IC50, 
                  "diff_d" for difference in the maximum (d) value
        y_max: maximum value for the y (vertical) axis
        leg_loc: location of the legend, for example, "upper left", "lower right"
        leg_ncol: number of columns in the legend
        leg_fontsize: font size for the legend
        pvalue_thresholds:
    """    
    
    results_df = _stack_replicate_results_frames([data_1rep, data_2rep, data_3rep])

    # Coerce numeric columns
    for col in ("MSE", "E", "r2_score", "d", "fit_d"):
        results_df[col] = pd.to_numeric(results_df[col], errors="coerce")

    # Drop rows where MSE is NaN (failed fits)
    results_df = results_df[results_df["MSE"].notna()]

    # Derived columns
    results_df["diff_d"] = (results_df["d"] - results_df["fit_d"]).abs()

    results_df = results_df.sort_values(
        "layout", key=lambda s: s.apply(DOSE_RESPONSE_LAYOUT_ORDER.index)
    )

    # Cast replicates to str so seaborn tick labels and Annotator order agree
    results_df["replicates"] = results_df["replicates"].astype(str)
    str_order = ["1", "2", "3"]

    if pvalue_thresholds is None:
        pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]]

    palette = [s.color for s in DOSE_RESPONSE_LAYOUT_SPECS]
    hue_order = DOSE_RESPONSE_LAYOUT_ORDER

    # Build string-keyed box_pairs to match string replicates
    box_pairs = [
        ((str(rep), DOSE_RESPONSE_LAYOUT_ORDER[i]), (str(rep), DOSE_RESPONSE_LAYOUT_ORDER[j]))
        for rep in (1, 2, 3)
        for i in range(len(DOSE_RESPONSE_LAYOUT_ORDER))
        for j in range(i + 1, len(DOSE_RESPONSE_LAYOUT_ORDER))
    ]

    # Determine plot column, filter, and y-label (do NOT mutate fig_type)
    if fig_type == "relic50":
        plot_col = "MSE"
        plot_data = results_df[results_df["MSE"] != np.inf]
        ylabel = "Mean absolute log10 difference"
    elif fig_type == "absic50":
        for col in ("e", "fit_e"):
            results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
        results_df["abs_e_diff"] = (results_df["e"] - results_df["fit_e"]).abs()
        plot_col = "abs_e_diff"
        plot_data = results_df.dropna(subset=["abs_e_diff"])
        ylabel = "Mean absolute IC50 difference"
    else:
        plot_col = "diff_d"
        plot_data = results_df
        ylabel = "Mean absolute d difference"

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(4, 3))
    if y_max:
        ax.set_ylim(top=y_max)

    sns.barplot(
        x="replicates", y=plot_col,
        data=plot_data,
        hue="layout", hue_order=hue_order,
        order=str_order,
        palette=palette, ax=ax,
    )
    ax.set_ylabel(ylabel, fontsize=10)

    annotator = Annotator(
        ax, pairs=box_pairs, data=plot_data,
        x="replicates", y=plot_col,
        hue="layout", order=str_order, hue_order=hue_order,
    )
    annotator.configure(
        test="t-test_ind", text_format="star",
        loc="inside", pvalue_thresholds=pvalue_thresholds, text_offset=-1,
    )
    annotator.apply_and_annotate()

    # Draw legend AFTER annotations so handles are bar patches, not annotation lines
    #ax.legend(fontsize=leg_fontsize, loc=leg_loc, ncol=leg_ncol)
    ax.legend(
        loc="lower center", bbox_to_anchor=(0.58, 1.01),
        ncol=len(hue_order), fontsize=8,
        borderaxespad=0
    )

    fig.savefig(
        f"{fig_dir}dose-response-{fig_type}{fig_name}.png",
        bbox_inches="tight", dpi=300,
    )
    plt.close(fig)


def plot_r2_percentage(
    data_1rep, data_2rep, data_3rep,
    fig_name='', fig_dir='',
    y_max=None,
    leg_loc="upper right", leg_ncol=1, leg_fontsize=8,
    hue_order=None,
    pvalue_thresholds=None,
    r2_threshold=0.8,
):
    """
    Plotting the percentage of low-quality curves for dose-response simulations
    as in the manuscript.
    """
    if hue_order is None:
        hue_order = DOSE_RESPONSE_LAYOUT_ORDER
    if pvalue_thresholds is None:
        pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]]

    results_df = _stack_replicate_results_frames([data_1rep, data_2rep, data_3rep])
    results_df["r2_score"] = pd.to_numeric(results_df["r2_score"], errors="coerce")
    results_df = results_df.sort_values(
        "layout", key=lambda s: s.apply(DOSE_RESPONSE_LAYOUT_ORDER.index)
    )

    # Boolean indicator used for both the barplot (mean = percentage/100) and t-test
    results_df["is_low_r2"] = (results_df["r2_score"] < r2_threshold).astype(float)

    # Cast replicates to str so tick labels and Annotator order agree
    results_df["replicates"] = results_df["replicates"].astype(str)
    str_order = ["1", "2", "3"]

    box_pairs = [
        ((str(rep), hue_order[i]), (str(rep), hue_order[j]))
        for rep in (1, 2, 3)
        for i in range(len(hue_order))
        for j in range(i + 1, len(hue_order))
    ]

    palette = [s.color for s in DOSE_RESPONSE_LAYOUT_SPECS]
    comparison_labels = list(hue_order)

    fig, ax = plt.subplots(figsize=(3.2, 3.2))

    # Plot mean of is_low_r2 (0–1) directly; format y-axis as percentages
    sns.barplot(
        x="replicates",
        y="is_low_r2",
        data=results_df,
        hue="layout",
        hue_order=hue_order,
        order=str_order,
        palette=sns.color_palette(palette, len(hue_order)),
        ax=ax,
    )

    # Format y-axis as 0-100% ticks
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda val, _: f"{val*100:.0f}")
    )
    ax.set_ylabel("Percentage of low quality curves", fontsize=10)

    if y_max is not None:
        ax.set_ylim(top=y_max / 100.0)  # caller passes 0-100, internal is 0-1

    annotator = Annotator(
        ax,
        pairs=box_pairs,
        data=results_df,
        x="replicates",
        y="is_low_r2",
        hue="layout",
        order=str_order,
        hue_order=hue_order,
    )
    annotator.configure(
        test="t-test_ind",
        text_format="star",
        loc="inside",
        pvalue_thresholds=pvalue_thresholds,
        text_offset=-1,
    )
    annotator.apply_and_annotate()

    ax.legend(
        loc="lower center", bbox_to_anchor=(0.58, 1.01),
        ncol=len(hue_order), fontsize=8,
        borderaxespad=0
    )

    fig.savefig(
        fig_dir + "percentage-low-r2" + fig_name + ".png",
        bbox_inches="tight", dpi=300,
    )
    plt.close(fig)


# TODO(dead-code): create_latex_table is superseded by create_latex_table_wide.
# Not called by any benchmark script.
def create_latex_table(data, tex_filename, column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df['MSE'] = pd.to_numeric(results_df['MSE'], errors='coerce')
    results_df = results_df.sort_values("MSE")
    results_df = results_df[np.logical_not(np.isnan(results_df['MSE']))]

    results_df['layout'] = _classify_dose_response_layout_series(results_df['layout'])

    results_df.d = pd.to_numeric(results_df.d, errors='coerce')
    results_df.fit_d = pd.to_numeric(results_df.fit_d, errors='coerce')

    results_df.insert(0, 'diff_d', 0)
    results_df.diff_d = abs(results_df.d - results_df.fit_d)    
    
    random_description = results_df[results_df['layout']=='Random'].describe()
    plaid_description = results_df[results_df['layout']=='PLAID'].describe()
    border_description = results_df[results_df['layout']=='COMPD'].describe()

    latex_f.write(" & Random & PLAID & COMPD \\\\ ")
    latex_f.write("\n\\hline\n")
    
    rows = [{'row_id':'count', 'row_name':'\\tabCount{}'},
            {'row_id':'mean',  'row_name':'\\tabMean{}'},
            {'row_id':'std',   'row_name':'\\tabSTD{}'},
            {'row_id':'min',   'row_name':'\\tabMin{}'},
            {'row_id':'25%',   'row_name':'\\tabQone{}'},
            {'row_id':'50%',   'row_name':'\\tabMedian{}'},
            {'row_id':'75%',   'row_name':'\\tabQthree{}'},
            {'row_id':'max',   'row_name':'\\tabMax{}'}]
    
    for row in rows:
        latex_f.write(row['row_name']+" & "+str(round(random_description.loc[row['row_id'],column_name],2))+" & "+str(round(plaid_description.loc[row['row_id'],column_name],2))+" & "+str(round(border_description.loc[row['row_id'],column_name],2))+"\\\\ \n")
    
    latex_f.write("\\hline")
        
    # Close file
    latex_f.close()
    
    
def create_latex_table_wide(
    data_1rep,
    data_2rep,
    data_3rep,
    tex_filename,
    table_text="Relative \\ECIC",
    column_name="MSE",
):
    """LaTeX table of per-layout, per-replicate summary stats (mean ± std).

    Uses the layout registry and generic replicate stacker so adding a new
    layout only requires updating DOSE_RESPONSE_LAYOUT_SPECS.
    """
    latex_f = open(tex_filename, "w")

    # Stack 1/2/3-replicate arrays into a single long frame
    results_df = _stack_replicate_results_frames(
        [data_1rep, data_2rep, data_3rep]
    )

    # Clean numeric column
    results_df[column_name] = pd.to_numeric(
        results_df[column_name], errors="coerce"
    )
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    results_df = results_df.dropna(subset=[column_name])

    # Layout order driven by registry
    layouts = [
        spec.display_type
        for spec in sorted(DOSE_RESPONSE_LAYOUT_SPECS, key=lambda s: s.plot_order)
    ]

    # Header row: multirow label + layout names
    latex_f.write(r"\multirow{4}{*}{" + table_text + "}")
    for layout in layouts:
        latex_f.write(" & " + layout)
    latex_f.write(r"\\ " + "\n")

    # One row per replicate, with mean ± std for each layout
    for rep in (1, 2, 3):
        latex_f.write(f"Rep {rep}")
        for layout in layouts:
            sub = results_df[
                (results_df["layout"] == layout)
                & (results_df["replicates"] == rep)
            ][column_name]

            if sub.empty:
                cell = "--"
            else:
                mean = sub.mean()
                std = sub.std()
                cell = f"{mean:.2f} $\\pm$ ({std:.2f})"

            latex_f.write(" & " + cell)
        latex_f.write(r"\\ " + "\n")

    latex_f.write(r"\hline" + "\n")
    latex_f.close()

    
    
    
    
    
def create_latex_table_pvalues_wide(
    data_1rep,
    data_2rep,
    data_3rep,
    tex_filename,
    table_text="Relative \\ECIC",
    column_name="MSE",
):
    """LaTeX table of per-replicate pairwise p-values between layouts.

    For each replicate (1,2,3) and each layout pair, writes a t-test p-value.
    Layout ordering and membership come from DOSE_RESPONSE_LAYOUT_SPECS.
    """
    latex_f = open(tex_filename, "w")

    # Stack arrays into a single long frame
    results_df = _stack_replicate_results_frames(
        [data_1rep, data_2rep, data_3rep]
    )

    results_df[column_name] = pd.to_numeric(
        results_df[column_name], errors="coerce"
    )
    results_df = results_df.replace([np.inf, -np.inf], np.nan)
    results_df = results_df.dropna(subset=[column_name])

    layouts = [
        spec.display_type
        for spec in sorted(DOSE_RESPONSE_LAYOUT_SPECS, key=lambda s: s.plot_order)
    ]

    latex_f.write(r"\multirow{4}{*}{" + table_text + "}")

    # One block per layout pair, with three replicate p-values
    for i, layout_1 in enumerate(layouts):
        for layout_2 in layouts[i + 1 :]:
            latex_f.write(" & " + layout_1 + " -- " + layout_2)

            for rep in (1, 2, 3):
                arr1 = results_df.loc[
                    (results_df["layout"] == layout_1)
                    & (results_df["replicates"] == rep),
                    column_name,
                ]
                arr2 = results_df.loc[
                    (results_df["layout"] == layout_2)
                    & (results_df["replicates"] == rep),
                    column_name,
                ]

                if arr1.empty or arr2.empty:
                    cell = "--"
                else:
                    _, pvalue = stats.ttest_ind(
                        arr1, arr2, equal_var=False
                    )
                    cell = f"{pvalue:.2e}"

                latex_f.write(" & " + cell)

            latex_f.write(r"\\ " + "\n")

    latex_f.write(r"\hline" + "\n")
    latex_f.close()

    
    
def full_controls_layout(layout, activity_layout, neg_control_id, pos_control_id):
    extended_controls_layout = np.copy(layout)
    num_rows, num_columns = layout.shape
    
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (layout[row_index][col_index] > 0 and activity_layout[row_index][col_index] == 0):
                extended_controls_layout[row_index][col_index] = neg_control_id
                            
            elif (layout[row_index][col_index] > 0 and activity_layout[row_index][col_index] == 1):
                 extended_controls_layout[row_index][col_index] = pos_control_id
                    
    return extended_controls_layout


def plate_to_random_layout(layout, activity_layout, num_neg_controls, num_pos_controls, neg_control_id, pos_control_id):
    num_rows, num_columns = layout.shape
    random_layout = np.full((num_rows, num_columns), 0)
    
    while num_neg_controls>0:
        rand_row = randrange(num_rows)
        rand_col = randrange(num_columns)
        
        if (layout[rand_row][rand_col] > 0) and (random_layout[rand_row][rand_col] == 0) and (activity_layout[rand_row][rand_col] == 0):
            random_layout[rand_row][rand_col] = neg_control_id
            num_neg_controls = num_neg_controls - 1

    while num_pos_controls>0:
        rand_row = randrange(num_rows)
        rand_col = randrange(num_columns)
        
        if (layout[rand_row][rand_col] > 0) and (random_layout[rand_row][rand_col] == 0) and (activity_layout[rand_row][rand_col] == 1):
            random_layout[rand_row][rand_col] = pos_control_id
            num_pos_controls = num_pos_controls - 1
            
    
    return random_layout


def plate_to_border_layout(layout, activity_layout, num_neg_controls, num_pos_controls, neg_control_id, pos_control_id):
    num_rows, num_columns = layout.shape
    border_layout = np.full((num_rows, num_columns), 0)

    for col_i in range(num_columns):
        for row_i in range(num_rows):
            for j in [col_i,num_columns-col_i-1]:
                if (layout[row_i][j] > 0) and (border_layout[row_i][j] == 0):
                    if num_neg_controls > 0 and (activity_layout[row_i][j] == 0):
                        border_layout[row_i][j] = neg_control_id
                        num_neg_controls = num_neg_controls - 1

                    elif num_pos_controls > 0 and (activity_layout[row_i][j] == 1):
                        border_layout[row_i][j] = pos_control_id
                        num_pos_controls = num_pos_controls - 1
        
        if num_neg_controls == 0 and num_pos_controls == 0: break
    
    return border_layout



def plotting_residual_metrics(screening_scores_data_filename, metric='Zfactor', fig_name=None, y_min=None, y_max=None, palette=None, plots_directory='', box_pairs=SCREENING_LAYOUT_BOX_PAIRS, order=SCREENING_LAYOUT_ORDER):
    print(screening_scores_data_filename)
    
    screening_scores_df = pd.read_csv(screening_scores_data_filename)
    # Drop any rows where display_type is the literal header string (repeated header rows)
    screening_scores_df = screening_scores_df[screening_scores_df['display_type'] != 'display_type']
    # Normalise capitalisation to match SCREENING_LAYOUT_ORDER
    screening_scores_df['display_type'] = screening_scores_df['display_type'].str.strip()

    screening_scores_df[metric+'_expected'] = pd.to_numeric(screening_scores_df[metric+'_expected'], errors='coerce')
    screening_scores_df[metric] = pd.to_numeric(screening_scores_df[metric], errors='coerce')

    frames = []
    for layout_name in order:
        sub = screening_scores_df[screening_scores_df['display_type'].str.lower() == layout_name.lower()]
        mse_df = pd.DataFrame(
            np.square(sub[metric + '_expected'] - sub[metric]), columns=['MSE']
        )
        mse_df.insert(0, 'layout', layout_name)
        frames.append(mse_df)
    results_df = pd.concat(frames, ignore_index=True)
    
    sns.set_theme(style="whitegrid")

    if palette is None:
        palette = sns.color_palette("Greens",len(order))

    fig, ax = plt.subplots(figsize=(4,3))
    
    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)

    ax = sns.barplot(x='layout', y="MSE", data=results_df, hue="layout", palette=palette, order=order, legend=False)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel("MSE", fontsize = 10)

    annotator = Annotator(ax, pairs=box_pairs, data=results_df, x='layout', y="MSE", order=order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', text_offset=-1)
    annotator.apply_and_annotate()

    if fig_name:
        fig.savefig(plots_directory+"screening-"+metric+"-mse-"+fig_name+".png",bbox_inches='tight',dpi=800)
    plt.close(fig)



def _load_screening_residuals(residuals_filename):
    """Read screening residuals CSV, drop lost-row plates, add derived columns.

    Returns a cleaned DataFrame with columns including 'display_type',
    'obtained_result_inv' (negated for scoring), and 'layout' (alias of
    display_type kept for backward compatibility with callers that used
    the old column name).
    """
    df = pd.read_csv(residuals_filename)
    df = df[df.lost_rows < 1].copy()
    # Validate display_type values against the registry early so mismatches
    # surface with a clear error rather than silently empty plots.
    _classify_screening_layout_series(df["display_type"])
    df["layout"] = df["display_type"]
    df["obtained_result_inv"] = -df["obtained_result"]
    return df


def _plot_roc_pr_all_batches(ax, df, display_type, color, curve_type):
    """Compute and plot a mean ROC or PR curve averaged across all batches.

    Each batch curve is interpolated onto a shared grid, then the mean and
    +/-1-std band are drawn. The legend label reports the mean AUC +/- std.

    Args:
        ax: matplotlib Axes to plot onto.
        df: full screening residuals DataFrame (already filtered for lost_rows<1).
        display_type: layout label to select rows.
        color: line colour string.
        curve_type: "roc" or "pr".

    Returns:
        Legend label string including mean AUC value +/- std.
    """
    sub = df[df["display_type"] == display_type]
    grid = np.linspace(0, 1, 200)
    interp_curves = []
    auc_vals = []

    for _batch_id, grp in sub.groupby("batch"):
        y_true = grp["activity"].to_numpy()
        y_score = grp["obtained_result_inv"].to_numpy()
        if len(np.unique(y_true)) < 2:
            continue

        if curve_type == "roc":
            fpr, tpr, _ = metrics.roc_curve(y_true, y_score)
            auc_vals.append(metrics.roc_auc_score(y_true, y_score))
            interp_curves.append(np.interp(grid, fpr, tpr))
        else:
            precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
            auc_vals.append(metrics.auc(recall, precision))
            # precision_recall_curve returns decreasing recall; reverse for interp
            interp_curves.append(np.interp(grid, recall[::-1], precision[::-1]))


    curves = np.array(interp_curves)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)
    mean_auc = float(np.mean(auc_vals))
    std_auc = float(np.std(auc_vals))

    ax.plot(grid, mean_curve, color=color)
    ax.fill_between(grid, mean_curve - std_curve, mean_curve + std_curve, color=color, alpha=0.15)
    #print(display_type, color, )
    
    return f"{display_type} (AUC = {mean_auc:.2f} \u00b1 {std_auc:.2f})"


def plot_roc_curves(residuals_filename, fig_name=None, fig_dir=''):
    """Plot mean ROC curves (± 1-std band) across all batches for every layout.

    Layout order, display labels, and colours are driven by the registry so
    adding a new layout requires only a change to SCREENING_LAYOUT_SPECS.

    Args:
        residuals_filename: path to the screening residuals CSV.
        fig_name: output filename (saved to fig_dir); skipped if None.
        fig_dir: directory prefix for the saved figure.
    """
    layout_colors = [spec.color for spec in SCREENING_LAYOUT_SPECS]
    df = _load_screening_residuals(residuals_filename)

    fig, ax = plt.subplots(figsize=(3, 2))
    legend_labels = []
    for spec, color in zip(SCREENING_LAYOUT_SPECS, layout_colors):
        label = _plot_roc_pr_all_batches(ax, df, spec.display_type, color, "roc")
        legend_labels.append(label)

    ax.set_ylabel("True Positive Rate", fontsize=10)
    ax.set_xlabel("False Positive Rate", fontsize=10)
    ax.legend(handles=ax.get_lines(), labels=legend_labels, loc="lower right", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    if fig_name:
        fig.savefig(os.path.join(fig_dir, fig_name), bbox_inches="tight", dpi=300)
    plt.close(fig)


def plot_pr_curves(residuals_filename, fig_name=None, fig_dir=''):
    """Plot mean Precision-Recall curves (+/- 1-std band) across all batches.

    Layout order, display labels, and colours are driven by the registry so
    adding a new layout requires only a change to SCREENING_LAYOUT_SPECS.

    Args:
        residuals_filename: path to the screening residuals CSV.
        fig_name: output filename (saved to fig_dir); skipped if None.
        fig_dir: directory prefix for the saved figure.
    """
    layout_colors = [spec.color for spec in SCREENING_LAYOUT_SPECS]
    df = _load_screening_residuals(residuals_filename)

    fig, ax = plt.subplots(figsize=(3, 2))
    legend_labels = []
    for spec, color in zip(SCREENING_LAYOUT_SPECS, layout_colors):
        label = _plot_roc_pr_all_batches(ax, df, spec.display_type, color, "pr")
        legend_labels.append(label)

    ax.set_ylabel("Precision", fontsize=10)
    ax.set_xlabel("Recall", fontsize=10)
    ax.legend(handles=ax.get_lines(), labels=legend_labels, loc="lower right", fontsize=8)
    ax.tick_params(axis="both", labelsize=8)
    if fig_name:
        fig.savefig(os.path.join(fig_dir, fig_name), bbox_inches="tight", dpi=300)
    plt.close(fig)


# TODO(dead-code): pr_auc_score is not called by either benchmark script.
# Remove once confirmed no external callers exist.
def pr_auc_score(y_true, y_score):
    precision, recall, _ = metrics.precision_recall_curve(y_true, y_score)
    return metrics.auc(recall, precision)


def _scatter_sample_groups(ax, df, x_col, y_col, neg_control_id, pos_control_id, s=14):
    """Draw four overlaid scatterplots for the four sample/control groups."""
    neg_samples   = df[(df["activity"] < 1) & (df["comp_id"] != neg_control_id) & (df["comp_id"] != pos_control_id)]
    neg_controls  = df[df["comp_id"] == neg_control_id]
    pos_samples   = df[(df["activity"] > 0) & (df["comp_id"] < pos_control_id)]
    pos_controls  = df[df["comp_id"] == pos_control_id]

    sns.scatterplot(x=x_col, y=y_col, data=neg_samples,  color="#d998b1", s=s, ax=ax)
    sns.scatterplot(x=x_col, y=y_col, data=neg_controls, color="#bf1f5f", s=s, ax=ax)
    sns.scatterplot(x=x_col, y=y_col, data=pos_samples,  color="#a8bfe6", s=s, ax=ax)
    sns.scatterplot(x=x_col, y=y_col, data=pos_controls, color="#3c7ef0", s=s, ax=ax)
    return ax


def _save_screening_scatter(df, x_col, y_col, neg_control_id, pos_control_id,
                             min_value, max_value, filepath):
    """Create, save, and close a single screening scatter figure."""
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_ylim(min_value, max_value)
    _scatter_sample_groups(ax, df, x_col, y_col, neg_control_id, pos_control_id)
    ax.set_xlabel("Plate number", fontsize=10)
    ax.set_ylabel("Response", fontsize=10)
    ax.set_xticks(range(5, 41, 5))
    ax.legend(
        labels=["Negative samples", "Negative control", "Positive samples", "Positive control"],
        ncol=2, loc="upper center", fontsize=8,
    )
    if filepath:
        fig.savefig(filepath, bbox_inches="tight", dpi=1200)
    plt.close(fig)


def plot_screening_plates(residuals_filename, fig_name=None, fig_dir='', max_value=300):
    """Plot per-layout screening scatter figures driven by SCREENING_LAYOUT_SPECS."""
    df = _load_screening_residuals(residuals_filename)

    sns.set_theme(style="whitegrid")

    neg_control_id = int(np.max(df["comp_id"]))
    pos_control_id = neg_control_id - 1
    min_value = float(df["obtained_result"].min()) - 10

    expected_spec = next(s for s in SCREENING_LAYOUT_SPECS if s.display_type == "PLAID")
    expected_df = df[df["display_type"] == expected_spec.display_type]

    prefix = os.path.join(fig_dir, "screening-bowl-" + fig_name) if fig_name else None
    _save_screening_scatter(
        expected_df, "plate_id", "expected_result",
        neg_control_id, pos_control_id,
        min_value, max_value,
        filepath=prefix + "-expected.png" if prefix else None,
    )

    for spec in SCREENING_LAYOUT_SPECS:
        layout_df = df[df["display_type"] == spec.display_type]
        _save_screening_scatter(
            layout_df, "plate_id", "obtained_result",
            neg_control_id, pos_control_id,
            min_value, max_value,
            filepath=prefix + f"-{spec.key}.png" if prefix else None,
        )

def plot_well_series_lowess_internal(plate_array, layout, neg_control_id=-1, pos_control_id=-1, order=0, vmin=None, vmax=None, filename=None):
    
    plate_df = pd.DataFrame(plate_array)
    
    intensity_df = plate_df.stack(future_stack=True).reset_index()
    intensity_df.columns = ["Rows","Columns","Intensity"]
    
    types_df = pd.DataFrame(layout).stack(future_stack=True).reset_index()
    types_df.columns = ["Rows","Columns","Type"]
    
    combined_df = pd.merge(intensity_df, types_df,  how='left', on=['Rows','Columns'])
    
    unstack_df = combined_df[["Rows","Columns","Intensity"]].copy()
    unstacked_df = pd.pivot_table(unstack_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc='sum')
    plot_plate(unstacked_df, title="Input",filename=filename+'heatmap-before')
    
    y_adjusted = combined_df.copy()
    y_adjusted.reset_index()
    
    lowess_rows_model = lowess.Lowess()
    lowess_rows_model.fit(y_adjusted[y_adjusted.Type==neg_control_id].Rows.to_numpy(),y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy(),frac=1,num_fits=5000)

    xnew_rows = np.array([i for i in range(0,16)])
    y_pred_rows = lowess_rows_model.predict(xnew_rows)
    
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= y_pred_rows[y_adjusted.loc[y_adjusted['Type']>0,['Rows']]] 
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += np.nanmean(y_pred_rows)
    
    lowess_model = lowess.Lowess()
    lowess_model.fit(y_adjusted[y_adjusted.Type==neg_control_id].Columns.to_numpy(),y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy(),frac=1,num_fits=5000)

    xnew = np.array([i for i in range(0,24)])
    y_pred = lowess_model.predict(xnew)

    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= y_pred[y_adjusted.loc[y_adjusted['Type']>0,['Columns']]] 
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += np.nanmean(y_pred)
    
    fig, ax = plt.subplots(figsize=(10,6))
    ax.set(xlim=(0,25))
    ax = sns.regplot(data=combined_df[(combined_df.Type!=pos_control_id) & (combined_df.Type!=neg_control_id)], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, scatter_kws={"color":"orange","alpha":0.3})
    ax = sns.regplot(data=combined_df[combined_df.Type==pos_control_id], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, marker="+",scatter_kws={"color":"blue"})
    ax = sns.regplot(data=combined_df[combined_df.Type==neg_control_id], x="Columns", y="Intensity", x_jitter=0.3, fit_reg=False, marker='*',scatter_kws={"color":"purple"}, truncate=False, order=order) 
    ax = sns.regplot(data=y_adjusted, x="Columns", y="Intensity", x_jitter=0.3, fit_reg=True, marker='x',scatter_kws={"color":"blue"}, truncate=False, order=order)
    
    plt.plot(xnew, y_pred, '--', label='Estimate', color='k', zorder=3)
    ax.set_xticks(range(1,25))
    
    if (filename):
        fig.savefig(filename)
    plt.close(fig)
        
    unstack_adjusted_df = y_adjusted[["Rows","Columns","Intensity"]].copy()
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc='sum')
    
    return unstacked_adjusted_df.to_numpy()


def plot_well_series(*args, **kwargs):
    """Backward-compatible dispatcher for the two historical plot_well_series APIs."""
    if len(args) >= 5:
        return plot_well_series_precomputed_normalization(*args, **kwargs)
    return plot_well_series_lowess_internal(*args, **kwargs)
