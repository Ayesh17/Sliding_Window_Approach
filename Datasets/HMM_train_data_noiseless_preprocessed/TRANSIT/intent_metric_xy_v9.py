#! /usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
import os
from os.path import isfile, join
import csv
import glob
import matplotlib.cbook as cbook
import intents_utils as ints
import file_utils_V2 as file_utils
import HII_data_preprocess_V5 as data_preprocess

# CHANGES
# i5
#   - Considers ground truth files under the 'intent_mover' naming convention
# i6
#   - Turns the fifth plot into 'presence data'
#   - Also brought back the confusion matrix
#   - Brought back all_data_analysis.csv - BROKEN
# i8
#   - Runs all plot generating and analysis funtions independently from the online classifier
#   - Requires that the probability report HII and hmm-formatted files exist
# i9
#   - Generates the following plots (customized with command line arguments)
#       - basic position plot: file_utils.createBasicPositionPlot(hmmFormattedFile)
#       - HII prediction plot: file_utils.createHIIPredictionPlot(probReportHIIFile)
#       - color-coded position plot: file_utils.createColorCodedPositionPlot(hmmFormattedFile, probReportHIIFile)
#       - animated color-coded position plot: file_utils.createAnimatedGIF(hmmFormattedFile, probReportHIIFile)

version = 'i9'
gtFile = ''

def scientificNotation(value):
    if value == 0:
        return '0'
    else:
        e = np.log10(np.abs(value))
        m = np.sign(value) * 10 ** (e - int(e))
        return r'${:.0f} \cdot 10^{{{:d}}}$'.format(m, int(e))

def MillisecondToSecondNotation(value):
    # print value
    return r'$%d$' % (value)

def main():
    #print("Starting Program")
    fnameToSkip = []
    print()
    print("*"*100)
    # import data
    parser = argparse.ArgumentParser(description='Process Input')
    parser.add_argument('-d', '--dir', type=str, required=True,
                        help='Directory Of Probability CSV files')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Verbose output that shows each graph before saving')
    
    parser.add_argument('-ip', '--HII_intent_plot', action='store_true',
                        help='Generate the HII prediction plot.')

    parser.add_argument('-rp', '--regular_intent_plot', action='store_true',
                        help='Generate the old-version prediction plot.')

    parser.add_argument('-cp', '--xy_color_plot', action='store_true',
                        help='Generate the color-coded position plot.')

    parser.add_argument('-p', '--xy_plot', action='store_true',
                        help='Generate the regular position plot.')

    parser.add_argument('-a', '--xy_anim_plot', action='store_true',
                        help='Generate the color-coded animated position plot.')

    parser.add_argument('-cf', '--confusion_matrix', action='store_true',
                        help='Generate the confusion matrix.')

    parser.add_argument('-tcf', '--transit_confusion_matrix', action='store_true',
                        help='Generate the transit confusion matrix.')

    parser.add_argument('-mx', '--metric_calculation', action='store_true',
                        help='Metric calculation.')

    parser.add_argument('-da', '--all_data_analysis', action='store_true',
                        help='All data analysis.')

    parser.add_argument('-noise', '--HMM_train_data_noise', action='store_true',
                        help='Noise data preprocessing.')

    parser.add_argument('-noiseless', '--HMM_train_data_noiseless', action='store_true',
                        help='Noiseless data preprocessing.')


    args = parser.parse_args()
    if not os.path.exists(args.dir):
        print("error directory doesn't exist")
        return -1
    if os.path.isfile(args.dir):
        print("Error: Expected Directory not File")
        return -1
    print(args.dir)
    accuracy_list = []
    early_predict_list = []
    early_predict_list_2 = []
    changes_list = []
    filename_list = []
    hmmFormattedFile =''
    probReportHIIFile = ''
    probReportFile = ''
    test_dir = join(os.getcwd(), 'HMM_test_data')
    noise_train_dir = join(os.getcwd(), 'HMM_train_data_noise')
    noiseless_train_dir = join(os.getcwd(), 'HMM_train_data_noiseless')
    noise_train_preprocessed_dir = join(os.getcwd(), 'HMM_train_data_noise_preprocessed')
    noiseless_train_preprocessed_dir = join(os.getcwd(), 'HMM_train_data_noiseless_preprocessed')

    for path, subdirs, files in os.walk(args.dir):
        for name in files:
            filename = os.path.join(path, name)
            if '.pdf' in name or '.pkl' in name or '.json' in name or '.png' in name:
                #print ("Skip: ", name)
                continue
            #print('Filename = ', filename)
            if 'hmm_formatted.csv' in name:
                 hmmFormattedFile = os.path.join(path, name)
                 print ("HMM formatted:", hmmFormattedFile)
            if 'probability_report_HII.csv' in name:
                probReportHIIFile = os.path.join(path, name)
                print("Probability report HII: ", probReportHIIFile)
            if 'probability_report.csv' in name:
                probReportFile = os.path.join(path, name)
                print("Probability report: ", probReportFile)
            if 'h12_analysis_i7.2.csv' in name:
                dataAnalysisFile = os.path.join(path, name)
                print("Data analysis report: ", dataAnalysisFile)
            #if 'GT' in name or 'intent_mover' in name:
            #    gtFile = os.path.join(path, name)

    ##########################################################################################
    ############
    ############        Noise Data Preprocessing
    ############
    ###########################################################################################

    if args.noise_data_preprocess:
        data_preprocess.dataPreprocess(noise_train_dir, noise_train_preprocessed_dir)

    ##########################################################################################
    ############
    ############        Noiseless Data Preprocessing
    ############
    ###########################################################################################

    if args.noiseless_data_preprocess:
        data_preprocess.dataPreprocess(noiseless_train_dir, noiseless_train_preprocessed_dir)


    ##########################################################################################
    ############
    ############        Generate XY plot
    ############
    ###########################################################################################

    if args.xy_plot:
        file_utils.createBasicPositionPlot(hmmFormattedFile)


    ##########################################################################################
    ############
    ############        Generate HII prediction plot
    ############
    ###########################################################################################

    if args.HII_intent_plot:
        file_utils.createHIIPredictionPlot(probReportHIIFile)
    
    
    ##########################################################################################
    ############
    ############        Generate color-coded XY plot
    ############
    ###########################################################################################

    if args.xy_color_plot:
        file_utils.createColorCodedPositionPlot(hmmFormattedFile, probReportHIIFile)
    
    ##########################################################################################
    ############
    ############        Generate animated color-coded animated XY plot
    ############
    ###########################################################################################

    if args.xy_anim_plot:
        file_utils.createAnimatedGIF(hmmFormattedFile, probReportHIIFile)

    ##########################################################################################
    ############
    ############        Generate Confusion matrix for Execute phase
    ############
    ###########################################################################################

    if args.confusion_matrix:
        file_utils.createConfusionMatrix(probReportFile, probReportHIIFile)


    ##########################################################################################
    ############
    ############        Generate Confusion matrix for Transit Phase
    ############
    ###########################################################################################

    if args.transit_confusion_matrix:
        file_utils.createTransitConfusionMatrix(probReportFile, probReportHIIFile)

    ##########################################################################################
    ############
    ############        Metric Calculation
    ############
    ###########################################################################################

    if args.metric_calculation:
        file_utils.metric_calculation(probReportFile, probReportHIIFile)


    ##########################################################################################
    ############
    ############        ALl data analysis
    ############
    ###########################################################################################

    if args.all_data_analysis:
        file_utils.all_data_analysis(test_dir)

   
if __name__ == '__main__':
    main()