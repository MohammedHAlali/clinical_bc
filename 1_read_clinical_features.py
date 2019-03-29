# -*- coding: utf-8 -*-
"""

code to read clinical data from organized txt file that was downloaded from cancer.gov portal
we only care about the cases that we've diganostic images for them.

The output is 'clean_features.csv', contains all features in binary format, ready to be used as labels for Neural Network
Mohammed Alali
March, 2019
"""
import csv

print('hello world')
clinical_file = "nationwidechildrens.org_clinical_patient_brca.txt"
features_file = "extracted_features_csv.csv"

patients = []
clinical_features_names = []

with open(clinical_file , 'r') as f:
	#print('lines in file: ', len(f.readlines()))
	clinical_features_names = f.readline().split('\t')
	print('feature_names len: ', len(clinical_features_names))
	print('features: ', clinical_features_names)
	f.readline()
	f.readline() 
	for i, line in enumerate(f.readlines()):
		line = line.split('\t')
		#print('line length: ', len(line))
		#print('{} len line: {}'.format(i, len(line)))
		patient_dic = {} #save each line in a dictionary
		if(len(clinical_features_names) != len(line)):
			raise ValueError('ERROR: two lines not equal')
		for j, value in enumerate(line):
			#print('{} - {} = {}'.format(j, clinical_features_names[j],value))
			patient_dic[clinical_features_names[j].strip()] = value.strip()
		#print('patient dictionary: ', patient_dic)
		# save all patients in one list
		patients.append(patient_dic) 


#save all patients in one list
print('we have {} patients'.format(len(patients)))
#print('patients dictionary list: ', patients)

with open(features_file) as csvfile:
	csv_reader = csv.reader(csvfile)
	extracted_features_names = []
	selected_clinical_values_list = []
	#my_clinical_values = {}
	feature_names = next(csv_reader)
	print('feature names: ', feature_names, ' of length: ', len(feature_names))
	for i, row in enumerate(csv_reader):
		img_id = row[0].strip()	
		print('{} id:{} len:{}'.format(i, img_id, len(img_id)))
		#get patient clinical data 	
		for p in patients:
			if p['bcr_patient_barcode'] == img_id:
				my_clinical_values = {}
				my_clinical_values['id'] = img_id
				print('img id: ', my_clinical_values['id'])
				# margin_status: Close, Negative, Positive, [Unknown]
				if p['margin_status'] == 'Negative' or p['margin_status'] == '[Unknown]':
					my_clinical_values['margin_status_neg'] = 1
				else:
					my_clinical_values['margin_status_neg'] = 0
				if p['margin_status'] == 'Positive':
					my_clinical_values['margin_status_pos'] = 1
				else:
					my_clinical_values['margin_status_pos'] = 0
				if p['margin_status'] == 'Close':
					my_clinical_values['margin_status_close'] = 1
				else:
					my_clinical_values['margin_status_close'] = 0

				#print('my clinical features dict: ', my_clinical_values)
				
				# race, not needed
				# age at dignosis, not needed
				# gender, not needed 
				# cancer type = classification
				if p['histological_type'] == 'Infiltrating Ductal Carcinoma':
					my_clinical_values['idc_pos'] = 1 #yes for IDC
				else:
					my_clinical_values['idc_pos'] = 0
				if p['histological_type'] == 'Infiltrating Lobular Carcinoma':
					my_clinical_values['ilc_pos'] = 1 #yes for ILC
				else:
					my_clinical_values['ilc_pos'] = 0
		
				
				if row[2] == 'yes': #2nd column: DCIS
					my_clinical_values['dcis_pos'] = 1
				else:
					my_clinical_values['dcis_pos'] = 0
                    
				if row[4] == 'yes': # column for: LCIS
					my_clinical_values['lcis_pos'] = 1
				else:
					my_clinical_values['lcis_pos'] = 0                    

				#LCIS type
				if row[5] == 'classic':
					my_clinical_values['lcis_type_classic'] = 1
				else:
					my_clinical_values['lcis_type_classic'] = 0
				if row[5] == 'multifocal':
					my_clinical_values['lcis_type_multifocal'] = 1
				else:
					my_clinical_values['lcis_type_multifocal'] = 0
		 
				#DCIS_type: combination of overlapping four types:
				# row[6] papillary type
				# row[7] cribriform type
				# row[8] solid type
				# row[9] comedo
				if row[6] == 'yes':
					my_clinical_values['dcis_type_papillary'] = 1
				else:
					my_clinical_values['dcis_type_papillary'] = 0
				if row[7] == 'yes':
					my_clinical_values['dcis_type_cribriform'] = 1
				else:
					my_clinical_values['dcis_type_cribriform'] = 0
				if row[8] == 'yes':
					my_clinical_values['dcis_type_solid'] = 1
				else:
					my_clinical_values['dcis_type_solid'] = 0
				if row[9] == 'yes':
					my_clinical_values['dcis_type_comedo'] = 1
				else:
					my_clinical_values['dcis_type_comedo'] = 0
	
				#row[11]: DCIS nuclear grade
				if row[10] == '--' or row[11] == '2': #unknown/unmentioned
					my_clinical_values['dcis_grade_med'] = 1 #medium (default)
				else:
					my_clinical_values['dcis_grade_med'] = 0
				if row[10] == '1':
					my_clinical_values['dcis_grade_low'] = 1 #low
				else:
					my_clinical_values['dcis_grade_low'] = 0
				if row[10] == '3':
					my_clinical_values['dcis_grade_high'] = 1
				else:
					my_clinical_values['dcis_grade_high'] = 0           

				#Necrosis in DCIS
				if row[11] == '--' or row[11] == 'no':
					my_clinical_values['dcis_Necrosis'] = 0 #not present/negative
				else:
					my_clinical_values['dcis_Necrosis'] = 1 #present/positive
                    
				#Microcalcifications
				if row[12] == '--' or row[12] == 'no':
					my_clinical_values['calcifications'] = 0 #not present/negative
				else:
					my_clinical_values['calcifications'] = 1 #present/positive
		                  
 
				#13: histologic grade, differentiated: well/low=1, moderately=2, poor/high=3
				if row[13] == '--' or row[13] == '2':
					my_clinical_values['histological_grade_moder'] = 1
				else:
					my_clinical_values['histological_grade_moder'] = 0
				if row[13] == '1':
					my_clinical_values['histological_grade_well'] = 1
				else:
					my_clinical_values['histological_grade_well'] = 0
				if row[13] == '3':
					my_clinical_values['histological_grade_poor'] = 1
				else:
					my_clinical_values['histological_grade_poor'] = 0
                
 
				#row[14]: invasive tumor nuclear grade (1-3, low-high)
				if row[14] == '--' or row[14] == '2': #med/default
					my_clinical_values['invasive_grade_med'] = 1
				else:
					my_clinical_values['invasive_grade_med'] = 0
				if row[14] == '1': #low
					my_clinical_values['invasive_grade_low'] = 1
				else:
					my_clinical_values['invasive_grade_low'] = 0
				if row[14] == '3': #high
					my_clinical_values['invasive_grade_high'] = 1
				else:
					my_clinical_values['invasive_grade_high'] = 0

				#row[15]: Tubule/papilla formation (1-3, low-high)
				if row[15] == '--' or row[15] == '2': #med/default
					my_clinical_values['tubule_index_med'] = 1
				else:
					my_clinical_values['tubule_index_med'] = 0
				if row[15] == '1': #low
					my_clinical_values['tubule_index_low'] = 1
				else:
					my_clinical_values['tubule_index_low'] = 0
				if row[15] == '3': #high
					my_clinical_values['tubule_index_high'] = 1
				else:
					my_clinical_values['tubule_index_high'] = 0

				#row[16]: mitotic index (1-3, low-high)
				if row[16] == '--' or row[16] == '2': #med/default
					my_clinical_values['mitotic_index_med'] = 1
				else:
					my_clinical_values['mitotic_index_med'] = 0
				if row[16] == '1': #low
					my_clinical_values['mitotic_index_low'] = 1
				else:
					my_clinical_values['mitotic_index_low'] = 0
				if row[16] == '3': #high
					my_clinical_values['mitotic_index_high'] = 1
				else:
					my_clinical_values['mitotic_index_high'] = 0
                    
				# invasive tumor necrosis
				if row[17] == '--' or row[17] == 'no':
					my_clinical_values['invasive_necrosis'] = 0
				else:
					my_clinical_values['invasive_necrosis'] = 1
				
				#18: perineural invasion
				if row[18] == '--' or row[18] == 'no':
					my_clinical_values['perineural_invasion'] = 0
				else:
					my_clinical_values['perineural_invasion'] = 1
				
				# 19: venous blood invasion
				if row[19] == '--' or row[19] == 'no':
					my_clinical_values['Venous_blood_invasion'] = 0
				else:
					my_clinical_values['Venous_blood_invasion'] = 1

				#20: lymphatic-vascular invasion
				if row[20] == '--' or row[20] == 'no':
					my_clinical_values['lymphatic_vascular_invasion'] = 0
				else:
					my_clinical_values['lymphatic_vascular_invasion'] = 1
				#print(my_clinical_values, ' of length: ', len(my_clinical_values))
 
				#row26: AJCC tumor pathology primary tumor (AJCC_pt)
				if p['ajcc_tumor_pathologic_pt'] == 'T1' or p['ajcc_tumor_pathologic_pt'] == 'T1b' or p['ajcc_tumor_pathologic_pt'] == 'T1c':
					my_clinical_values['ajcc_pt_t1'] = 1
				else:
					my_clinical_values['ajcc_pt_t1'] = 0
			
				if p['ajcc_tumor_pathologic_pt'] == 'T2':
					my_clinical_values['ajcc_pt_t2'] = 1
				else:
					my_clinical_values['ajcc_pt_t2'] = 0
				if p['ajcc_tumor_pathologic_pt'] == 'T3':
					my_clinical_values['ajcc_pt_t3'] = 1
				else:
					my_clinical_values['ajcc_pt_t3'] = 0
				if p['ajcc_tumor_pathologic_pt'] == 'T4' or p['ajcc_tumor_pathologic_pt'] == 'T4b':
					my_clinical_values['ajcc_pt_t4'] = 1
				else:
					my_clinical_values['ajcc_pt_t4'] = 0
				#print(my_clinical_values, ' of length: ', len(my_clinical_values))
				#print('count = ', count)
 
				#AJJC tumor pathology lymph node envolvement (ajcc_pn)
				if p['ajcc_nodes_pathologic_pn'] == 'N0' or p['ajcc_nodes_pathologic_pn'] == 'N0 (i+)' or p['ajcc_nodes_pathologic_pn'] == 'N0 (i-)':
					my_clinical_values['ajcc_pn_n0'] = 1
				else:
					my_clinical_values['ajcc_pn_n0'] = 0
				if p['ajcc_nodes_pathologic_pn'] == 'N1' or p['ajcc_nodes_pathologic_pn'] == 'N1a' or p['ajcc_nodes_pathologic_pn'] == 'N1b' or p['ajcc_nodes_pathologic_pn'] == 'N1mi':
					my_clinical_values['ajcc_pn_n1'] = 1
				else:
					my_clinical_values['ajcc_pn_n1'] = 0
				if p['ajcc_nodes_pathologic_pn'] == 'N2' or p['ajcc_nodes_pathologic_pn'] == 'N2a':
					my_clinical_values['ajcc_pn_n2'] = 1
				else:
					my_clinical_values['ajcc_pn_n2'] = 0
				if p['ajcc_nodes_pathologic_pn'] == 'N3' or p['ajcc_nodes_pathologic_pn'] == 'N3a' or p['ajcc_nodes_pathologic_pn'] == 'N3b' or p['ajcc_nodes_pathologic_pn'] == 'N3c' or p['ajcc_nodes_pathologic_pn'] == 'N3mi':
					my_clinical_values['ajcc_pn_n3'] = 1
				else:
					my_clinical_values['ajcc_pn_n3'] = 0
				

				#AJJC metastasis
				if p['ajcc_metastasis_pathologic_pm'] == 'MX':
					my_clinical_values['ajcc_pm_mx'] = 1
				else:
					my_clinical_values['ajcc_pm_mx'] = 0
				if p['ajcc_metastasis_pathologic_pm'] == 'M0':
					my_clinical_values['ajcc_pm_m0'] = 1
				else:
					my_clinical_values['ajcc_pm_m0'] = 0
				if p['ajcc_metastasis_pathologic_pm'] == 'M1':
					my_clinical_values['ajcc_pm_m1'] = 1
				else:
					my_clinical_values['ajcc_pm_m1'] = 0
				print(my_clinical_values, ' of length: ', len(my_clinical_values))
                    
				#AJCC staging
				if (p['ajcc_pathologic_tumor_stage'] == 'Stage I' or 
					p['ajcc_pathologic_tumor_stage'] == 'Stage IA' or 
					p['ajcc_pathologic_tumor_stage'] == 'Stage IB' or 
					p['ajcc_pathologic_tumor_stage'] == 'Stage IC'):
					my_clinical_values['ajcc_stage_I'] = 1
				else:
					my_clinical_values['ajcc_stage_I'] = 0

				if (p['ajcc_pathologic_tumor_stage'] == 'Stage II' or
					p['ajcc_pathologic_tumor_stage'] == 'Stage IIA' or
					p['ajcc_pathologic_tumor_stage'] == 'Stage IIB' or
					p['ajcc_pathologic_tumor_stage'] == 'Stage IIC'):
					my_clinical_values['ajcc_stage_II'] = 1
				else:
					my_clinical_values['ajcc_stage_II'] = 0
				if (p['ajcc_pathologic_tumor_stage'] == 'Stage III' or 
					p['ajcc_pathologic_tumor_stage'] == 'Stage IIIB' or
					p['ajcc_pathologic_tumor_stage'] == 'Stage IIIC'):
					my_clinical_values['ajcc_stage_III'] = 1
				else:
					my_clinical_values['ajcc_stage_III'] = 0
				if p['ajcc_pathologic_tumor_stage'] == 'Stage IV':
					my_clinical_values['ajcc_stage_IV'] = 1
				else:
					my_clinical_values['ajcc_stage_IV'] = 0
				selected_clinical_values_list.append(my_clinical_values)
				print('values: ', my_clinical_values)
		print('all data: ', selected_clinical_values_list)
		print('len of patients list: ', len(selected_clinical_values_list))
 
#writing binary data into csv file
with open('clean_features.csv', 'w', newline='') as csvfile:
    dict_keys = my_clinical_values.keys()
    writer = csv.DictWriter(csvfile, fieldnames=dict_keys)
    writer.writeheader()
    for data in selected_clinical_values_list:
        writer.writerow(data)
#    for item in selected_clinical_values_list:
#        for key in item:
#            csv_writer.write('{},{}\n'.format(key, selected_clinical_values[key]))
print('done')
