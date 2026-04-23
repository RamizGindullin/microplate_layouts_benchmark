import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
import os
import re
from statannotations.Annotator import Annotator # to add p-values to plots
from scipy import stats
from random import randrange
from sklearn import metrics
import statistics
from moepy import lowess
from scipy.interpolate import interp1d
from benchmark_common import (
    DOSE_RESPONSE_LAYOUT_BOX_PAIRS_BY_REPLICATE,
    DOSE_RESPONSE_LAYOUT_ORDER,
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



def _prepare_dose_response_results_frame(data_1rep, data_2rep, data_3rep):
    columns = [
        "layout", "compound", "MSE", "error type", "Error", "E", "rows lost",
        "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"
    ]
    frame1 = pd.DataFrame(data_1rep, columns=columns)
    frame1.insert(0, "replicates", 1)
    frame2 = pd.DataFrame(data_2rep, columns=columns)
    frame2.insert(0, "replicates", 2)
    frame3 = pd.DataFrame(data_3rep, columns=columns)
    frame3.insert(0, "replicates", 3)
    df = pd.concat([frame1, frame2, frame3], ignore_index=True)
    df["layout"] = _classify_dose_response_layout_series(df["layout"])
    return df



def _prepare_dose_response_residuals_frame(residuals_1rep, residuals_2rep, residuals_3rep):
    columns = ["layout", "error_type", "Error", "E", "rows lost", "residuals", "true_residuals"]
    frame1 = pd.DataFrame(residuals_1rep, columns=columns)
    frame1.insert(0, "replicates", 1)
    frame2 = pd.DataFrame(residuals_2rep, columns=columns)
    frame2.insert(0, "replicates", 2)
    frame3 = pd.DataFrame(residuals_3rep, columns=columns)
    frame3.insert(0, "replicates", 3)
    df = pd.concat([frame1, frame2, frame3], ignore_index=True)
    df["layout"] = _classify_dose_response_layout_series(df["layout"])
    df["residuals"] = pd.to_numeric(df["residuals"], errors="coerce")
    return df


def plot_plate(plate_array, title="", mask=None, filename=None, vmin=None, vmax=None):
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.xaxis.tick_top()
    plt.title(title, fontsize = 15) 
    sns.heatmap(plate_array,linewidth=0.3,square=True,mask=mask,vmin=vmin,vmax=vmax)
    if filename:
        fig.savefig(filename,bbox_inches='tight')
    plt.close(fig)
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



def __shape_layout(layout, num_rows, num_columns, size_empty_edge):
    layout = np.reshape(layout,(-1, num_columns-2*size_empty_edge))
    
    if size_empty_edge > 0:
        vertical_edge = np.reshape(np.full(size_empty_edge*(num_rows-2*size_empty_edge),0), (-1,size_empty_edge))
                         
        layout = np.hstack((vertical_edge,layout))
    
        layout = np.hstack((layout,vertical_edge))
    
        horizontal_edge = np.reshape(np.full(size_empty_edge*num_columns,0), (-1,num_columns))
    
        layout = np.vstack((horizontal_edge,layout))
    
        layout = np.vstack((layout,horizontal_edge))
    
    return layout


def check_duplicated_layouts(layout_dir = 'screening_manual_layouts/'):

    layouts = os.listdir(layout_dir)
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
    unstacked_df = pd.pivot_table(unstack_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)

    #if filename:
     #   plot_plate(unstacked_df, title="Input",filename=filename+'heatmap-before')
    #else:
     #   plot_plate(unstacked_df, title="Input")
    
    
    ### Plot heatmap after normalization
    unstack_adjusted_df = n_combined_df[["Rows","Columns","Intensity"]].copy()
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)
    
  #  if filename:
   #     plot_plate(unstacked_adjusted_df, title="Normalized",vmin=vmin,vmax=vmax,filename=filename+'heatmap-after')
    #else:
     #   plot_plate(unstacked_adjusted_df, title="Normalized",vmin=vmin,vmax=vmax)
    
    
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
    
    
    

    
    
def plot_barplot_residuals_data(residuals_1rep, residuals_2rep, residuals_3rep, fig_name, y_max=None, leg_loc="lower center", leg_ncol=3, leg_fontsize=8, pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]], hue_order=DOSE_RESPONSE_LAYOUT_ORDER, box_pairs=None, fig_dir=''):
    """ Plots residual plots for dose response experiments as in the manuscript. """
    residuals_df = _prepare_dose_response_residuals_frame(
        residuals_1rep, residuals_2rep, residuals_3rep
    )
    residuals_df = residuals_df.rename(columns={"replicates": "Replicate", "layout": "Layout type", "residuals": "Residuals"})
    comparison_labels = list(hue_order)
    if box_pairs is None:
        box_pairs = DOSE_RESPONSE_LAYOUT_BOX_PAIRS_BY_REPLICATE

    fig,ax = plt.subplots(figsize=(4,5))
    palette = ['#4c72b0', '#55a868', '#c44e52']
    sns.boxplot(data=residuals_df, x='Replicate', y='Residuals', hue='Layout type', ax=ax, palette=palette, hue_order=comparison_labels)
    _apply_boxplot_annotations(
        ax,
        data=residuals_df,
        x='Replicate',
        y='Residuals',
        pairs=box_pairs,
        order=[1, 2, 3],
        hue='Layout type',
        hue_order=comparison_labels,
    )
    ax.legend(loc=leg_loc, ncol=leg_ncol, fontsize=leg_fontsize)
    ax.set_ylim(top=y_max)
    fig.savefig(fig_dir + fig_name + '.png', bbox_inches='tight')
    plt.close(fig)

def plot_barplot_replicate_data(data_1rep, data_2rep, data_3rep, fig_name='', fig_dir='', fig_type='', plot_mse=True, y_max=None, leg_ncol=1, leg_loc="best", leg_fontsize=8, box_pairs3=None, pvalue_thresholds=None, hue_order=DOSE_RESPONSE_LAYOUT_ORDER):
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
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = pd.concat([results_df,results_df_2rep])
    results_df = pd.concat([results_df,results_df_3rep])

    results_df.MSE = pd.to_numeric(results_df.MSE, errors='coerce')
    results_df.E = pd.to_numeric(results_df.E, errors='coerce')
    results_df.r2_score = pd.to_numeric(results_df.r2_score, errors='coerce')
    results_df.d = pd.to_numeric(results_df.d, errors='coerce')
    results_df.fit_d = pd.to_numeric(results_df.fit_d, errors='coerce')

    results_df.insert(0, 'diff_d', 0)
    results_df.diff_d = abs(results_df.d - results_df.fit_d)

    results_df = results_df[np.logical_not(np.isnan(results_df['MSE']))]


    results_df['layout'] = _classify_dose_response_layout_series(results_df['layout'])
    
    results_df = results_df.sort_values('layout', key = lambda s: s.apply(['Random','PLAID','COMPD'].index))
#df = df.sort_values('A', key=lambda s: s.apply(['July', 'August', 'Sept'].index), ignore_index=True)

    fig, ax = plt.subplots(figsize=(4, 3))

    if pvalue_thresholds is None:
        # * indicates p < 10−4, ** indicates p < 10−12, *** indicates p < 10−43.
        pvalue_thresholds = [[1e-43, "***"], [1e-12, "**"], [1e-4, "*"], [1, "ns"]] #[1e-64, "****"], 

    box_pairs = DOSE_RESPONSE_LAYOUT_BOX_PAIRS_BY_REPLICATE
    hue_order = DOSE_RESPONSE_LAYOUT_ORDER

    if y_max:
        ax.set_ylim(top = y_max)
    
    ## Plotting
    if fig_type == "relic50":
        relic50_palette = ["#91d1c2", "#00A087", "#236e56"] #"#3bccaa", 
        ax = sns.barplot(x='replicates', y="MSE", data=results_df[results_df['MSE']!=np.inf], hue="layout", hue_order=hue_order, palette=relic50_palette)
        plt.ylabel("Mean absolute log10 difference", fontsize = 10)
               
    elif fig_type == "absic50":
        ax = sns.barplot(x='replicates', y="MSE", data=results_df[results_df['MSE']!=np.inf], hue="layout", hue_order=hue_order, palette='YlOrBr')
        plt.ylabel("Mean absolute log10 difference", fontsize = 10)
        
    else:
        ax = sns.barplot(x='replicates', y="diff_d", data=results_df, hue="layout", hue_order=hue_order, palette = "GnBu")#, palette='YlOrBr')
        plt.ylabel("Mean absolute d difference", fontsize = 10)
        fig_type = "d_diff"
        

    plt.legend(fontsize = leg_fontsize, loc = leg_loc, ncol = leg_ncol)

    #annotator = Annotator(ax, pairs=[((2,"Random"),(2,"PLAID"))], data=results_df[results_df['MSE']!=np.inf], x='replicates', y="MSE",hue='layout', order=[1,2,3],hue_order=hue_order)
    #annotator.configure(test='t-test_ind', text_format='star', loc='inside',pvalue_thresholds=pvalue_thresholds, text_offset=-1)
    #annotator.apply_and_annotate()

    annotator = Annotator(ax, pairs=box_pairs, data=results_df[results_df['MSE']!=np.inf], x='replicates', y="MSE",hue='layout', order=[1,2,3],hue_order=hue_order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside',pvalue_thresholds=pvalue_thresholds, text_offset=-1)
    annotator.apply_and_annotate()


    #plt.legend().set_title(None)
    fig.savefig(fig_dir+"dose-response-"+fig_type+fig_name+".png",bbox_inches='tight',dpi=800)
    plt.close(fig)
    
    
    
    
    
    
    
def plot_r2_percentage(data_1rep, data_2rep, data_3rep, fig_name='', fig_dir='', y_max=None, leg_loc="upper left", leg_ncol=1, leg_fontsize=8, hue_order=DOSE_RESPONSE_LAYOUT_ORDER):
    """
    Plotting the percentage of low-quality curves for dose-response simulations as in the manuscript.
    """
    results_df = _prepare_dose_response_results_frame(data_1rep, data_2rep, data_3rep)
    results_df = _coerce_numeric_column(results_df, "r2_score")
    results_df["low_quality_curve"] = results_df["r2_score"] < 0.8
    low_r2 = (
        results_df.groupby(["replicates", "layout"], as_index=False)["low_quality_curve"]
        .mean()
        .rename(columns={"replicates": "Replicate", "layout": "Layout type", "low_quality_curve": "Residuals"})
    )
    low_r2["Residuals"] = low_r2["Residuals"] * 100.0
    comparison_labels = list(hue_order)

    fig,ax = plt.subplots(figsize=(3.2,3.2))
    palette = ['#4c72b0', '#55a868', '#c44e52']
    sns.boxplot(data=low_r2, x='Replicate', y='Residuals', hue='Layout type', ax=ax, palette=palette, hue_order=comparison_labels)
    ax.legend(loc=leg_loc, ncol=leg_ncol, fontsize=leg_fontsize)
    ax.set_ylim(top=y_max)
    fig.savefig(fig_dir + fig_name + '.png', bbox_inches='tight')
    plt.close(fig)

def create_latex_table(data, tex_filename, column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df.MSE = pd.to_numeric(results_df.MSE, errors='coerce')
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
    
    
def create_latex_table_wide(data_1rep, data_2rep, data_3rep, tex_filename, table_text = "Relative \\ECIC", column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = pd.concat([results_df,results_df_2rep])
    results_df = pd.concat([results_df,results_df_3rep])

    results_df.MSE = pd.to_numeric(results_df[column_name], errors='coerce')
    results_df = results_df.sort_values(column_name)
    results_df = results_df[np.logical_not(np.isnan(results_df[column_name]))]
    
    results_df['layout'] = _classify_dose_response_layout_series(results_df['layout'])
    
#    plaid_description = results_df[results_df['layout']=='Effective'].describe()
 #   random_description = results_df[results_df['layout']=='Random'].describe()
  #  border_description = results_df[results_df['layout']=='Border'].describe()

#    latex_f.write(" & Effective & Random & Border \\\\ ")
#    latex_f.write("\n\\hline\n")

    layouts = [s.display_type for s in sorted(DOSE_RESPONSE_LAYOUT_SPECS, key=lambda s: s.plot_order)]
    
    latex_f.write("\\multirow{4}{*}{"+table_text+"}")
    
    for layout in layouts:
        latex_f.write(" & "+layout)
        
        for rep in range(1,4):
            description = results_df[(results_df['layout']==layout) & (results_df['replicates']==rep)].describe()
            latex_f.write(" & "+str(round(description.loc['mean',column_name],2))+" $\\pm$ ("+str(round(description.loc['std',column_name],2))+")")
            
        latex_f.write("\\\\ \n")
    
 #   for row in rows:
  #      latex_f.write(row['row_name']+" & "+str(round(plaid_description.loc[row['row_id'],column_name],2))+" & "+str(round(random_description.loc[row['row_id'],column_name],2))+" & "+str(round(border_description.loc[row['row_id'],column_name],2))+"\\\\ \n")
    
    latex_f.write("\\hline \n")
        
    # Close file
    latex_f.close()

    
    
    
    
    
def create_latex_table_pvalues_wide(data_1rep, data_2rep, data_3rep, tex_filename, table_text = "Relative \\ECIC", column_name="MSE"):
    # Open file
    latex_f=open(tex_filename,'w')
    
    results_df = pd.DataFrame(data_1rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_2rep = pd.DataFrame(data_2rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])
    results_df_3rep = pd.DataFrame(data_3rep, columns=["layout", "compound", "MSE", "error type", "Error", "E", "rows lost", "r2_score", "b", "c", "d", "e", "fit_b", "fit_c", "fit_d", "fit_e"])

    results_df.insert(0, 'replicates', 1)
    results_df_2rep.insert(0, 'replicates', 2)
    results_df_3rep.insert(0, 'replicates', 3)

    results_df = pd.concat([results_df,results_df_2rep])
    results_df = pd.concat([results_df,results_df_3rep])

    results_df[column_name] = pd.to_numeric(results_df[column_name], errors='coerce')
    results_df = results_df.sort_values(column_name)
    results_df = results_df[np.logical_not(np.isnan(results_df[column_name]))]
    results_df = results_df[np.logical_not(np.isinf(results_df[column_name]))]
    
    results_df['layout'] = _classify_dose_response_layout_series(results_df['layout'])
    
    layouts = ['Random','PLAID','COMPD']
    
    latex_f.write("\\multirow{4}{*}{"+table_text+"}")
    
    for layout_1 in range(3):
        for layout_2 in range(layout_1+1,3):
            latex_f.write(" & "+layouts[layout_1]+" -- "+layouts[layout_2])

            for rep in range(1,4):
                results_array_1 = results_df.loc[(results_df.layout==layouts[layout_1]) & (results_df.replicates==rep),column_name]
                results_array_2 = results_df.loc[(results_df.layout==layouts[layout_2]) & (results_df.replicates==rep),column_name]

                _, pvalue = stats.ttest_ind(results_array_1,results_array_2,equal_var = False)
                latex_f.write(" & "+'{:.2e}'.format(pvalue))

            latex_f.write("\\\\ \n")
    
    latex_f.write("\\hline \n")
        
    # Close file
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

    screening_scores_df[metric+'_expected'] = pd.to_numeric(screening_scores_df[metric+'_expected'], errors='coerce')
    screening_scores_df[metric] = pd.to_numeric(screening_scores_df[metric], errors='coerce')

    results_df = pd.DataFrame(np.square(screening_scores_df[screening_scores_df['plate_type'] == order[0]][metric+'_expected'] - 
                                        screening_scores_df[screening_scores_df['plate_type'] == order[0]][metric]),
                              columns = ['MSE'])
    results_df.insert(0, 'layout', order[0])

    for type in order[1:]:
        temp_df = pd.DataFrame(np.square(screening_scores_df[screening_scores_df['plate_type'] == type][metric+'_expected'] -
                                                             screening_scores_df[screening_scores_df['plate_type'] == type][metric]),
                                                             columns = ['MSE'])
        temp_df.insert(0, 'layout', type)
        results_df = pd.concat([results_df, temp_df])
    
    sns.set_style("whitegrid", {'axes.grid' : True})

    if palette is None:
        palette = sns.color_palette("Greens",5)

    fig, ax = plt.subplots(figsize=(4,3))
    
    if y_min:
        ax.set_ylim(bottom = y_min)
    if y_max:
        ax.set_ylim(top = y_max)


    ax = sns.barplot(x='layout', y="MSE", data=results_df, palette=palette, order=order)
    plt.tick_params(axis='both', which='major', labelsize=10)
    plt.ylabel("MSE", fontsize = 10)

    #pvalue_thresholds = [[1e-4, "***"], [1e-2, "**"], [0.05, "*"],[1, "ns"]]
    

    annotator = Annotator(ax, pairs=box_pairs, data=results_df, x='layout', y="MSE", order=order)
    annotator.configure(test='t-test_ind', text_format='star', loc='inside', text_offset=-1)
    annotator.apply_and_annotate()

    if fig_name:
        fig.savefig(plots_directory+"screening-"+metric+"-mse-"+fig_name+".png",bbox_inches='tight',dpi=800)
    plt.close(fig)


        
def plot_roc_curves(residuals_filename, fig_name=None, fig_dir='', batch=0, batches=10):

    screening_residuals_df = pd.read_csv(residuals_filename)
    screening_residuals_df['layout'] = screening_residuals_df['display_type']

    screening_residuals_df = screening_residuals_df[screening_residuals_df.lost_rows<1]

    screening_residuals_df['layout'] = screening_residuals_df['display_type']

    screening_residuals_df['obtained_result_inv'] = -screening_residuals_df.obtained_result

    colors = ['#59296e','#cc0253','#e68302']

    results_plaid = screening_residuals_df[(screening_residuals_df.layout=='PLAID') ]
    results_random = screening_residuals_df[(screening_residuals_df.layout=='Random') ]
    results_border = screening_residuals_df[(screening_residuals_df.layout=='COMPD') ]


    fig, ax = plt.subplots(figsize=(3, 2))

    ## Random
    fpr, tpr, thresholds = metrics.roc_curve(results_random.loc[results_random.batch==batch,'activity'],  results_random.loc[results_random.batch==batch,'obtained_result_inv'])
    auc_random = metrics.roc_auc_score(results_random.loc[results_random.batch==batch,'activity'],  results_random.loc[results_random.batch==batch,'obtained_result_inv'])

    #create ROC curve
    plt.plot(fpr,tpr,color=colors[0])


    ## PLAID
    fpr, tpr, thresholds = metrics.roc_curve(results_plaid.loc[results_plaid.batch==batch,'activity'],  results_plaid.loc[results_plaid.batch==batch,'obtained_result_inv'])
    auc_plaid = metrics.roc_auc_score(results_plaid.loc[results_plaid.batch==batch,'activity'],  results_plaid.loc[results_plaid.batch==batch,'obtained_result_inv'])
    
    
    #create ROC curve
    plt.plot(fpr,tpr,color=colors[1])

    ## Border
    fpr, tpr, thresholds = metrics.roc_curve(results_border.loc[results_border.batch==batch,'activity'],  results_border.loc[results_border.batch==batch,'obtained_result_inv'])
    auc_border = metrics.roc_auc_score(results_border.loc[results_border.batch==batch,'activity'],  results_border.loc[results_border.batch==batch,'obtained_result_inv'])

    #create ROC curve
    plt.plot(fpr,tpr,color=colors[2])


    plt.ylabel('True Positive Rate', fontsize = 10)
    plt.xlabel('False Positive Rate', fontsize = 10)
    plt.legend(labels=['Random (AUC = '+str(round(auc_random,2))+')',
                       'PLAID (AUC = '+str(round(auc_plaid,2))+')',
                       'COMPD (AUC = '+str(round(auc_border,2))+')'],
               loc='lower right', fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if fig_name:
        fig.savefig(fig_dir+fig_name,bbox_inches='tight',dpi=300)
    plt.close(fig)
    

    
def plot_pr_curves(residuals_filename, fig_name=None, fig_dir='', batch=0, batches=10):

    #plt.rcParams['text.usetex'] = True

    screening_residuals_df = pd.read_csv(residuals_filename)
    screening_residuals_df['layout'] = screening_residuals_df['display_type']

    screening_residuals_df = screening_residuals_df[screening_residuals_df.lost_rows<1]

    screening_residuals_df['layout'] = screening_residuals_df['display_type']

    screening_residuals_df['obtained_result_inv'] = -screening_residuals_df.obtained_result

    colors = ['#59296e','#cc0253','#e68302']

    results_plaid = screening_residuals_df[(screening_residuals_df.layout=='PLAID') ]
    results_random = screening_residuals_df[(screening_residuals_df.layout=='Random') ]
    results_border = screening_residuals_df[(screening_residuals_df.layout=='COMPD') ]


    fig, ax = plt.subplots(figsize=(3, 2))


     ## Random
    precision, recall, thresholds = metrics.precision_recall_curve(results_random.loc[results_random.batch==batch,'activity'],  results_random.loc[results_random.batch==batch,'obtained_result_inv'])
    auc_random = metrics.auc(recall,precision)

    #draw PR curve
    plt.plot(recall,precision,color=colors[0])


    ## PLAID
    precision, recall, thresholds = metrics.precision_recall_curve(results_plaid.loc[results_plaid.batch==batch,'activity'],  results_plaid.loc[results_plaid.batch==batch,'obtained_result_inv'])
    auc_plaid = metrics.auc(recall,precision)
    auc_plaid_str = '{0:.2g}'.format(auc_plaid)

    #draw PR curve
    plt.plot(recall,precision,color=colors[1])

    ## Border
    precision, recall, thresholds = metrics.precision_recall_curve(results_border.loc[results_border.batch==batch,'activity'],  results_border.loc[results_border.batch==batch,'obtained_result_inv'])
    auc_border = metrics.auc(recall,precision)

    #draw PR curve
    plt.plot(recall,precision,color=colors[2])


    plt.ylabel('Precision', fontsize = 10)
    plt.xlabel('Recall', fontsize = 10)
    plt.legend(labels=['Random (AUC = '+str(round(auc_random,2))+')',
                       'PLAID (AUC = '+str(round(auc_plaid,2))+')',
                       'COMPD (AUC = '+str(round(auc_border,2))+')'],
               loc='lower right', fontsize = 8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    if fig_name:
        fig.savefig(fig_dir+fig_name,bbox_inches='tight',dpi=300)
    plt.close(fig)
    

    
def pr_auc_score(y_true, y_score):
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_score)
    
    return metrics.auc(recall,precision)

def plot_screening_plates(residuals_filename, fig_name=None, fig_dir='',max_value=300):
    screening_residuals_df = pd.read_csv(residuals_filename)

    screening_residuals_df = screening_residuals_df[screening_residuals_df.lost_rows<1]

    screening_residuals_df['layout'] = screening_residuals_df['display_type']

    neg_control_id = np.max(screening_residuals_df.comp_id)
    pos_control_id = neg_control_id -1 

    sns.set_theme(style="whitegrid")

    #max_value = max(screening_residuals_df.obtained_result)+45
    #max_value = 300
    min_value = min(screening_residuals_df.obtained_result)-10

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set(ylim=(min_value,max_value))
    #plt.yscale('log', base=2)

    #'expected_result','obtained_result'
    ax = sns.scatterplot(x="plate_id", y="expected_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.activity <1)], color='#d998b1', s=14)
    ax = sns.scatterplot(x="plate_id", y="expected_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.comp_id==neg_control_id)],color='#bf1f5f', s=14)
    ax = sns.scatterplot(x="plate_id", y="expected_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.activity > 0) & (screening_residuals_df.comp_id<pos_control_id)],color='#a8bfe6', s=14)
    ax = sns.scatterplot(x="plate_id", y="expected_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.comp_id==pos_control_id)], color='#3c7ef0', s=14)
    plt.xlabel('Plate number', fontsize = 10)
    plt.ylabel('Response', fontsize = 10)
    plt.xticks([i for i in range(5,41,5)])
    plt.legend(labels=['Negative samples','Negative control','Positive samples','Positive control'],ncol=2, loc="upper center", fontsize = 8)
    if fig_name:
        fig.savefig(fig_dir+"screening-bowl-"+fig_name+"-expected.png",bbox_inches='tight',dpi=1200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set(ylim=(min_value,max_value))
    #plt.yscale('log', base=2)

    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.activity <1)], color='#d998b1', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.comp_id==neg_control_id)],color='#bf1f5f', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.activity > 0) & (screening_residuals_df.comp_id<pos_control_id)],color='#a8bfe6', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='PLAID') & (screening_residuals_df.comp_id==pos_control_id)], color='#3c7ef0', s=14)
    plt.xlabel('Plate number', fontsize = 10)
    plt.ylabel('Response', fontsize = 10)
    plt.xticks([i for i in range(5,41,5)])
    plt.legend(labels=['Negative samples','Negative control','Positive samples','Positive control'],ncol=2, loc="upper center", fontsize = 8)
    if fig_name:
        fig.savefig(fig_dir+"screening-bowl-"+fig_name+"-plaid.png",bbox_inches='tight',dpi=1200)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(4,3))
    ax.set(ylim=(min_value,max_value))
    #plt.yscale('log', base=2)

    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='Random') & (screening_residuals_df.activity <1)], color='#d998b1', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='Random') & (screening_residuals_df.comp_id==neg_control_id)],color='#bf1f5f', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='Random') & (screening_residuals_df.activity > 0) & (screening_residuals_df.comp_id<pos_control_id)],color='#a8bfe6', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='Random') & (screening_residuals_df.comp_id==pos_control_id)], color='#3c7ef0', s=14)
    plt.xlabel('Plate number', fontsize = 10)
    plt.ylabel('Response', fontsize = 10)
    plt.xticks([i for i in range(5,41,5)])
    plt.legend(labels=['Negative samples','Negative control','Positive samples','Positive control'],ncol=2, loc="upper center", fontsize = 8)
    if fig_name:
        fig.savefig(fig_dir+"screening-bowl-"+fig_name+"-random.png",bbox_inches='tight',dpi=1200)
    plt.close(fig)


    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set(ylim=(min_value,max_value))

    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='COMPD') & (screening_residuals_df.activity <1)], color='#d998b1', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='COMPD') & (screening_residuals_df.comp_id==neg_control_id)],color='#bf1f5f', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='COMPD') & (screening_residuals_df.activity > 0) & (screening_residuals_df.comp_id<pos_control_id)],color='#a8bfe6', s=14)
    ax = sns.scatterplot(x="plate_id", y="obtained_result", data=screening_residuals_df.loc[(screening_residuals_df.layout=='COMPD') & (screening_residuals_df.comp_id==pos_control_id)], color='#3c7ef0', s=14)
    plt.xlabel('Plate number', fontsize = 10)
    plt.ylabel('Response', fontsize = 10)
    plt.xticks([i for i in range(5,41,5)])
    plt.legend(labels=['Negative samples','Negative control','Positive samples','Positive control'],ncol=2, loc="upper center", fontsize = 8)
    if fig_name:
        fig.savefig(fig_dir+"screening-bowl-"+fig_name+"-compd.png",bbox_inches='tight',dpi=1200)
    plt.close(fig)
        


def plot_well_series_lowess_internal(plate_array, layout, neg_control_id=-1, pos_control_id=-1, order=0, vmin=None, vmax=None, filename=None):
    
    plate_df = pd.DataFrame(plate_array)
    
    intensity_df = plate_df.stack(future_stack=True).reset_index() ##
    intensity_df.columns = ["Rows","Columns","Intensity"] ##
    
    types_df = pd.DataFrame(layout).stack(future_stack=True).reset_index() ##
    types_df.columns = ["Rows","Columns","Type"] ##
    
    combined_df = pd.merge(intensity_df, types_df,  how='left', on=['Rows','Columns']) ##
    
    
    #### Test unstack
    
    unstack_df = combined_df[["Rows","Columns","Intensity"]].copy()
    
    unstacked_df = pd.pivot_table(unstack_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)
    
    plot_plate(unstacked_df, title="Input",filename=filename+'heatmap-before')
    
    ####
        
    ## Before ###
    
    y_adjusted = combined_df.copy()
    y_adjusted.reset_index() ##
    
    
    ## CALL NORM
    
    ### Adjust rows
    lowess_rows_model = lowess.Lowess()
    lowess_rows_model.fit(y_adjusted[y_adjusted.Type==neg_control_id].Rows.to_numpy(),y_adjusted[y_adjusted.Type==neg_control_id].Intensity.to_numpy(),frac=1,num_fits=5000)

    xnew_rows = np.array([i for i in range(0,16)])
    y_pred_rows = lowess_rows_model.predict(xnew_rows)
    
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] -= y_pred_rows[y_adjusted.loc[y_adjusted['Type']>0,['Rows']]] 
    y_adjusted.loc[y_adjusted['Type']>0, ['Intensity']] += np.nanmean(y_pred_rows)
    
    # Model fitting Columns
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
    
    #ax = sns.regplot(data=y_adjusted, x="Columns", y="Intensity", x_jitter=0.3, fit_reg=True, marker='x',scatter_kws={"color":"blue"}, truncate=False, order=order)
    
    plt.plot(xnew, y_pred, '--', label='Estimate', color='k', zorder=3)
    
    ax.set_xticks(range(1,25))
    
    if (filename):
        fig.savefig(filename)
    plt.close(fig)
        
    unstack_adjusted_df = y_adjusted[["Rows","Columns","Intensity"]].copy()
    
    unstacked_adjusted_df = pd.pivot_table(unstack_adjusted_df, values='Intensity', index=['Rows'],columns=['Columns'], aggfunc=np.sum)
    
    return unstacked_adjusted_df.to_numpy()


def plot_well_series(*args, **kwargs):
    """Backward-compatible dispatcher for the two historical plot_well_series APIs.

    Supported call shapes:
    - plot_well_series(plate_array, norm_plate, layout, neg_control_id, pos_control_id, ...)
      -> plot_well_series_precomputed_normalization
    - plot_well_series(plate_array, layout, neg_control_id=-1, pos_control_id=-1, ...)
      -> plot_well_series_lowess_internal
    """
    if len(args) >= 5:
        return plot_well_series_precomputed_normalization(*args, **kwargs)
    return plot_well_series_lowess_internal(*args, **kwargs)