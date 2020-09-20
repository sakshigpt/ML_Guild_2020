import getpass

username = getpass.getuser()

encoding_columns = ['CD', 'City', 'Lender', 'NAICSCode', 'State', 'Zip']
ohe_columns = ['Veteran', 'RaceEthnicity', 'Gender', 'BusinessType', 'DateApproved_month', 'DateApproved_day_of_week', 'DateApproved_date']
path_data_raw = '/Users/' + username + '/Desktop/Guild_Competition/ML_Guild_2020/0.data/0.raw'
path_data_processed = '/Users/' + username + '/Desktop/Guild_Competition/ML_Guild_2020/0.data/1.data_prep'
path_data_master = '/Users/' + username + '/Desktop/Guild_Competition/ML_Guild_2020/0.data/2.master'
target = 'JobsRetained'