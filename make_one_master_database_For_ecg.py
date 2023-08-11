# -*- coding: utf-8 -*-
import wfdb
import matplotlib.pyplot as plt
import pandas as pd
import neurokit2 as nk

#%%
def data_ecg(ecg_size,peak,ec_data):

    if (ecg_size>=240):
        ls=[]
        for i in range(peak-120,peak+120):
            ls.append(ec_data[i])
        return ls
    
    else:
        ls=[0 for i in range(240)]
        intial=peak-(ecg_size/2)
        end=peak+ecg_size/2
        c=120
        for i in range(peak,int(end)):
            ls[c]=ec_data[i]
            c=c+1
        c=120
        for i in range(peak-1,int(intial),-1):
            c=c-1
            ls[c]=ec_data[i]
        return ls
# %%
data_path ='C:/mit-bih-arrhythmia-database-1.0.0/'
final_data=[]
l=[100,101,102,103,104,105,106,107,108,109,111,112,113,114,115
   ,116,117,118,119,121,122,123,124,200,201,202,203,205,207,208,209,210,
   212,213,214,215,217,219,220,221,222,223,228,230,231,232,233,234]
count1=0
count2=0
label_mapping =['N','L','R','B','A','a','J','S','V','r','F','e','j','n','E','/','f','Q','?']
for k in range(0,len(l),2):
    for i in [l[k],l[k+1]] :
        ch=0
        samples=650000
        sam_freq=360
        # Read the ECG record and annotations
        record, fields = wfdb.rdsamp(data_path+str(i), channels=[ch], sampfrom=0, sampto=samples)
        annotation = wfdb.rdann(data_path+str(i), 'atr', sampfrom=0, sampto=samples)
############################################################################################################
        
        ecg_data=nk.ecg_clean(record[:,ch],sampling_rate=sam_freq)
        
        rpeaks = nk.ecg_peaks(ecg_data, sampling_rate=sam_freq)
        
        # %%
        #rpeaks is a tuple 
        rpeak_samples = rpeaks[0]
        
        list_peaks=rpeaks[1]['ECG_R_Peaks'] 
        """
        THIS DATA OF LIST IN A FORM OF ARRAY FROM THE RPEAKS TUPLE(CHECK ITS CONTENT) I.E 
        rpeaks[1]['ECG_R_Peaks'] 
        """
        # for getting the each ecg signal size
        list_ecg_size=list()
        for i in range(1,len(list_peaks)):
                list_ecg_size.append(list_peaks[i]-list_peaks[i-1])
        # %%
        flag=0
        for i in range (len(list_ecg_size)):
            d=data_ecg(list_ecg_size[i], list_peaks[i], ecg_data)
            count1=count1+1
            ##annotation work
            for j in range(len(annotation.sample)):
                if abs(annotation.sample[j]-list_peaks[i])<70:
                    if (annotation.symbol[j] in label_mapping):
                        d.append(annotation.symbol[j])
                        final_data.append(d)
                        flag=1
                        count2=count2+1
                        break;
df=pd.DataFrame(final_data)
df.to_csv('ecg_all_data.csv',index=False)#it creates its own index additionally
# %%

"""
# Plot the ECG record
plt.plot(ecg_data)
########################################################################################################

for i in range(len(annotation.sample)):
     plt.annotate(annotation.symbol[i], (annotation.sample[i], record[annotation.sample[i],ch]), color='red')

# Set plot labels and title
plt.xlabel('Sample')


plt.ylabel('Amplitude')
plt.title('ECG Record with Annotations')

# Display the plot
plt.show()
"""