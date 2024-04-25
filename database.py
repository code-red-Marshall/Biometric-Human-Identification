# -*- coding: utf-8 -*-
"""
Created on Thu Aug 18 13:36:26 2022

@author: user
"""

import matplotlib.pyplot as plt
import wfdb
import pandas as pd
import numpy as ny
patient_names = pd.read_fwf("ptb-diagnostic-ecg-database-1.0.0/RECORDS", dtype=str)

headers = ['PATIENT_NAME',
               # time
               'Tx', 'Px', 'Qx', 'Sx', 'PQ_time', 'PT_time', 'QS_time', 'QT_time', 'ST_time', 'PS_time',
               'PQ_QS_time', 'QT_QS_time',
               # amplitude
               'Ty', 'Py', 'Qy', 'Sy', 'PQ_ampl', 'QR_ampl', 'RS_ampl', 'QS_ampl', 'ST_ampl', 'PS_ampl', 'PT_ampl',
               'QT_ampl', 'ST_QS_ampl', 'RS_QR_ampl', 'PQ_QS_ampl', 'PQ_QT_ampl', 'PQ_PS_ampl', 'PQ_QR_ampl',
               'PQ_RS_ampl', 'RS_QS_ampl', 'RS_QT_ampl', 'ST_PQ_ampl', 'ST_QT_ampl',
               # distance
               'PQ_dist', 'QR_dist', 'RS_dist', 'ST_dist', 'QS_dist', 'PR_dist', 'ST_QS_dist', 'RS_QR_dist',
               # slope
               'PQ_slope', 'QR_slope', 'RS_slope', 'ST_slope', 'QS_slope', 'PT_slope', 'PS_slope', 'QT_slope',
               'PR_slope',
               # angle
               'PQR_angle', 'QRS_angle', 'RST_angle', 'RQS_angle', 'RSQ_angle', 'RTS_angle']

df = pd.DataFrame(columns=headers)
base_path = "ptb-diagnostic-ecg-database-1.0.0/"

for i in range(patient_names.size):
    patient_id = str(patient_names['patient001/s0010_re'][i])
    path = base_path + patient_id
    
    record = wfdb.rdrecord(path, channel_names=['v4'])
    signal = record.p_signal.ravel()
