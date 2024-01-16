##將學生像同心圓一樣分成很多level，再將學生依序分入群裡面，並依照限制名單進行更動分群結果。(隨機)
import pandas as pd
import numpy as np
import random
import csv
from data1 import Dataset1
from data3 import Dataset3
import time

from matplotlib import pyplot as plt

##檢查這些學生是否滿足限制式的函數，輸入有限制的名單檔案。
def check_con(student_to_group, constrained_data):
    constrained_pairs = []
    for _, row in constrained_data.iterrows():
        students = row.dropna().astype(int).tolist()
        if len(students) >= 2:
            constrained_pairs.extend([(s1, s2) for i, s1 in enumerate(students) for s2 in students[i+1:]])

    for s1, s2 in constrained_pairs:
        if s1 not in student_to_group or s2 not in student_to_group:
            continue
        if student_to_group[s1] == student_to_group[s2]:
            return False

    return True
    

##生成限制學生名單    
def generate_constrained_student_list(constrained_number, N):
    student_list = {}
    for i in range(1, constrained_number+1):
        # 隨機產生一個不在學生列表中的數字
        number = random.randint(0, N-1)
        while number in student_list.values():
            number = random.randint(0, N-1)
        student_list[f"學生{i}"] = number

    with open('data_no.csv', 'w+', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        students = []
        numbers = []
        for student, number in student_list.items():
            students.append(student)
            numbers.append(number)
        writer.writerow(students)
        writer.writerow(numbers)   
    
##接下來是使用ANOVA的變異數的F值倒數來評估組間組內差異。
##msw是組內平均和，每個屬性皆有各自的F值，而我們目前先看得是整體，所以我們將其相加，但相加無法看出各自屬性對各組的差異。還是需要列出來。
def msw(data,C):
    N=data.shape[0]
    xsum_msw=0
    for i in range(C):
        group_filter = (data["group"] == i)
        group_data = data[group_filter]
        group_data_values = group_data.drop(['number', 'dealing cards', 'group'], axis=1)
        group_mean = group_data_values.mean()
        group_counts = data[data["group"] == i].count().iloc[0]
        distant_array = np.zeros(shape=(group_counts, 1))
        group_size = group_counts
    
        for j in range(group_size):
            distant_array[j] = ((group_data_values.iloc[j] - group_mean) ** 2).sum()
        xsum = np.sum(distant_array)
        xsum_msw=xsum_msw+xsum    
    mean_msw = xsum_msw / (N-C)                    
    return mean_msw

def msb(data,C):
    data_value = data.drop(['number', 'dealing cards', 'group'], axis=1)  
    datamean=data_value.mean()
    xsum_msb=0
    distant_array= np.zeros((C,1))
    for i in range(C):
       group_filter = (data["group"] ==i)
       group_data = data[group_filter]
       group_counts = data[data["group"] == i].count().iloc[0]
       group_data_values = group_data.drop(['number', 'dealing cards','group'], axis=1)
       group_mean = group_data_values.mean()
       group_size = group_counts
       distant_array[i]=group_size * ((group_mean - datamean) ** 2).sum()  
    xsum = np.sum(distant_array)
    xsum_msb=xsum_msb+xsum
    mean_msb=xsum_msb/(C-1)
    return mean_msb
##此研究評估方法是利用F值倒數，組內/組間(補充:原本F值為組間/組內)
def F_reciprocal(data,C):
    MSW=msw(data,C) 
    MSB=msb(data,C) 
    F_reciprocal=MSW/MSB
    print(MSW,MSB)
    return F_reciprocal,MSW,MSB

def random_methon(num,C,N,atti,constrained_number):
  time_Start = time.time()#計時開始
  for v in range(1):
    result1 = np.zeros(shape=(num, 4))
    for t in range(num):
        data=pd.DataFrame(Dataset3(t, N, atti))
        N = data.shape[0]
        generate_constrained_student_list(constrained_number,N)
        constrained_data = pd.read_csv('data_no.csv')
        atti = data.shape[1]
        data["number"]=range(len(data))
        data["dealing cards"]=random.sample(range(N), N)

        #分組分層
       
        data["group"]=round(data["dealing cards"])%C##使用餘數分群
        groups = data.set_index('number')['group'].to_dict()
    
        constrained_exist = check_con(groups, constrained_data)
        #會把目前未滿足限制條件的分群結果，抓出限制名單中分到同一組的樣本，並進行與不是限制名單中樣本的群別的樣本進行隨機交換。
        if constrained_exist == False:
            for _, row in constrained_data.iterrows():
                students = row.dropna().astype(int).tolist()
                constrained = data[data["number"].isin(students)].copy()
                for group, number in constrained.iterrows():
                    same_group = constrained.groupby('group').filter(lambda x: len(x) >= 2)
                    one_constrained_group = constrained.groupby('group').filter(lambda x: len(x) == 1)
                    if (same_group['group'].value_counts()>=3).any():
                        no_chosen_sample = same_group.groupby('group',group_keys=False).apply(lambda x: x.sample(n=1, random_state=np.random.choice(x.index)))
                        chosen_sample =same_group.drop(no_chosen_sample.index)                       
                    else:
                        chosen_sample = same_group.groupby('group').apply(lambda x: x.sample(n=1, random_state=np.random.choice(x.index)))
                       
                    mask = (~data.group.isin(one_constrained_group["group"])) & (~data.group.isin(chosen_sample["group"]))
                    filtered_data_group = data.loc[mask]
                    if len(chosen_sample) < 2 :
                        chose_number = chosen_sample.iloc[0]["number"]
                        swap_number = filtered_data_group.sample(n=1, random_state=np.random.choice(filtered_data_group.index))["number"].iloc[0]
                        chose_index = data.loc[data["number"] == chose_number].index[0]
                        swap_index = data.loc[data["number"] == swap_number].index[0]
                        data.loc[chose_index, 'group'], data.loc[swap_index, 'group'] = data.loc[swap_index, 'group'], data.loc[chose_index, 'group']
                    else:    
                        for i in range(chosen_sample.count().iloc[0]):
                            chose_number = chosen_sample.iloc[i]["number"]
                            swap_number = filtered_data_group.sample(n=1, random_state=np.random.choice(filtered_data_group.index))["number"].iloc[0]
                            chose_index = data.loc[data["number"] == chose_number].index[0]
                            swap_index = data.loc[data["number"] == swap_number].index[0]
                            data.loc[chose_index, 'group'], data.loc[swap_index, 'group'] = data.loc[swap_index, 'group'], data.loc[chose_index, 'group']
                            mask_two =~data.group.isin(data.loc[data["number"] == chose_number]["group"])
                            filtered_data_group = filtered_data_group.loc[mask_two]
        
      
        Fi_value = F_reciprocal(data, C)
        
        Filist = list(range(3))
        Fiarray = np.array(Filist)
        Fiarray = Fiarray.astype(float)
        Fiarray =Fi_value
        
        time_End = time.time()  # 計時結束
        
        result1[t, 0] = Fiarray[0]
        result1[t, 1] = Fiarray[1]
        result1[t, 2] = Fiarray[2]
        result1[t, 3] = time_End - time_Start
        
     
    end = result1.sum(axis=0) / num
    return end[0], end[1], end[2],end[3]
 

