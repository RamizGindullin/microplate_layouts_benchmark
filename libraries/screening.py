## Library used for the screening simulation
# https://academic.oup.com/bioinformatics/article/31/23/3815/208794

import numpy as np
from random import randrange
from datetime import date
import os
import re
import csv 

import libraries.utilities as util

def control_stats(plate_array, layout, neg_control_id, pos_control_id):
    num_rows, num_columns = layout.shape
    
    #neg_control_id = np.max(layout)
    #pos_control_id = neg_control_id -1 
    
    pos_values = np.empty(0,np.float64)
    neg_values = np.empty(0,np.float64)
    
    ## Collect controls
    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (layout[row_index][col_index] == neg_control_id):
                 neg_values = np.append(neg_values,plate_array[row_index][col_index])
            elif (layout[row_index][col_index] == pos_control_id):
                 pos_values = np.append(pos_values,plate_array[row_index][col_index])

    # neg_control_mean, pos_control_mean, neg_stdev, pos_stdev
    return np.mean(neg_values), np.mean(pos_values), np.std(neg_values), np.std(pos_values)


# Strictly Standarized Mean Difference
def ssmd(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev):
    return (pos_control_mean - neg_control_mean)/np.sqrt(pos_stdev**2+neg_stdev**2)


def zfactor(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev):  
    return 1 - (3*(pos_stdev+neg_stdev)/abs(pos_control_mean - neg_control_mean))
                
                
                
def fill_plate(layout, neg_control_id, pos_control_id, neg_control_mean = 95, pos_control_mean = 5, neg_stdev = 5, pos_stdev = 5, percent_non_active = 0.66):
    num_rows, num_columns = layout.shape
        
    plate = np.full((num_rows, num_columns), 0.0)
    
    ##At least 0%, At most 100%
    percent_non_active = min(max(percent_non_active, 0.0),1.0)
    
    activity_layout = place_active_compounds(layout, neg_control_id, pos_control_id, percent_non_active)

    for row_index in range(num_rows):
        for col_index in range(num_columns):
            if (activity_layout[row_index][col_index] == 1):
                plate[row_index][col_index] = np.random.normal(pos_control_mean, pos_stdev)
            elif (layout[row_index][col_index] > 0):
                plate[row_index][col_index] = np.random.normal(neg_control_mean, neg_stdev)    
                
    return np.abs(plate), activity_layout


def place_active_compounds(layout, neg_control_id, pos_control_id, percent_non_active):
    num_rows, num_columns = layout.shape
        
    activity_plate = util.get_controls_layout(layout,pos_control_id)
    
    nb_active_comp = np.ceil((1-percent_non_active)*(neg_control_id-2))
    
    while nb_active_comp>0:
        rand_row = randrange(num_rows)
        rand_col = randrange(num_columns)
        
        if (layout[rand_row][rand_col] > 0) and (layout[rand_row][rand_col] < pos_control_id) and (activity_plate[rand_row][rand_col] != 1):
            activity_plate[rand_row][rand_col] = 1
            nb_active_comp = nb_active_comp - 1
            
            
    return activity_plate


def plate_metrics(plate, layout, neg_control_id, pos_control_id):
    neg_control_mean, pos_control_mean, neg_stdev, pos_stdev = control_stats(plate, layout, neg_control_id, pos_control_id)

    ssmd_value = ssmd(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev)
    zfactor_value = zfactor(neg_control_mean, pos_control_mean, neg_stdev, pos_stdev)
    
    return ssmd_value, zfactor_value


# Returns the filename
def test_quality_assessment_metrics(plate_types,error_types,error,id_text,neg_controls,pos_controls,neg_control_mean,pos_control_mean,neg_stdev,pos_stdev,data_directory,run_tag=None):
    ## Results
    today = run_tag if run_tag is not None else (date.today()).strftime("%Y%m%d")+"-"+id_text
    screening_scores_data_filename = 'screening_metrics_data-'+str(neg_controls)+'-'+str(pos_controls)+'-'+str(error)+'-'+today+'.csv'
    screening_scores_data_f=open(data_directory+screening_scores_data_filename,'a')

    # Write headers
    scores_writer = csv.writer(screening_scores_data_f)

    scores_writer.writerow(['plate', 'display_type', 'error_type', 'error', 'Zfactor_expected', 'SSMD_expected', 'Zfactor', 'SSMD'])

    for plate_type in plate_types:
        layout_dir = plate_type['dir']
        layouts = os.listdir(layout_dir)

        for layout_file in layouts:
            match = re.search(plate_type['regex'],layout_file)

            if match == None: #Skip other files, including system files
                continue

            layout = np.load(layout_dir+layout_file)  

            neg_control_id = np.max(layout)
            pos_control_id = neg_control_id -1 

            for et in error_types:

                # Fill in plate with "ideal" data
                ideal_plate, activity_layout = fill_plate(layout, neg_control_id, pos_control_id, neg_control_mean, pos_control_mean, neg_stdev = neg_stdev, pos_stdev = pos_stdev, percent_non_active=0.5)

                # Apply disturbances
                plate = et['error_function'](ideal_plate, error)

                # Full plate based on PLAID
                extended_controls_layout = util.full_controls_layout(layout, activity_layout, neg_control_id, pos_control_id)
                raw_neg_control_mean, raw_pos_control_mean, raw_neg_stdev, raw_pos_stdev = control_stats(plate, extended_controls_layout, neg_control_id, pos_control_id)

                ssmd_expected = ssmd(raw_neg_control_mean, raw_pos_control_mean, raw_neg_stdev, raw_pos_stdev)
                zfactor_expected = zfactor(raw_neg_control_mean, raw_pos_control_mean, raw_neg_stdev, raw_pos_stdev)

                # All metrics
                ssmd_plate, zfactor_plate = plate_metrics(plate, layout, neg_control_id, pos_control_id)
                
                ### Write to file!
                scores_writer.writerow([layout_file, plate_type['display_type'], et['type'], error, zfactor_expected, ssmd_expected, zfactor_plate, ssmd_plate])
               #print([layout_file, plate_type['type'], et['type'], error, zfactor_expected, ssmd_expected, zfactor_plate, ssmd_plate])


    # End and close file!
    screening_scores_data_f.close()
    
    return screening_scores_data_filename
