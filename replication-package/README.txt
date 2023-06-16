Replication package for "Evaluating different approaches for classifying privacy requirements in issue reports.

There are 2 folders in this replication package. 

1. Dataset - This folder contains the data files used in the study
- Google Chrome data: Chrome - Issues with labels.csv
- Moodle: Moodle - Issues with labels.csv

2. Code - This folder contains the source files

CNN
- Source file: IST22_IssueClassification_Req_Chrome_CNN.ipynb
- Data:	Chrome_reg_level.csv
	
- Source file: IST22_IssueClassification_Req_Moodle_CNN.ipynb
- Data:	Moodle_reg_level.csv

BERT
- Source file: MSR21_IssueClassification_Req_Chrome_BERT.ipynb
- Data:	Chrome - Issues with labels.csv
	
- Source file: MSR21_IssueClassification_Req_Moodle_BERT.ipynb
- Data:	Moodle - Issues with labels.csv

Code for other classification methods is in 'Classification' folder
- Source file: run_script.py is a main source file (set project data, textual feature extraction techniques and classifier in this file, it will automatically call other files as necessary)
- Data: Chrome - Chrome_reg_level.csv
	Moodle - Moodle_reg_level.csv

Wilcoxon test
- Source file: IST22_Calculate_Wilcoxon_Recall.ipynb
