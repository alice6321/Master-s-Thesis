##隨機產生均勻或常態資料進行實驗，並隨機生成限制名單。
##將學生像同心圓一樣分成很多level，再將學生依序分入群裡面，並依照限制名單進行更動分群結果。
import pandas as pd
import numpy as np
import math  as ma
import time
import random
import csv
from data1 import Dataset1
from data3 import Dataset3
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
    level_counts = data.groupby("group").count().iloc[:, 0]
    xsum_msw=0
    for i in range(C):
        group_filter = (data["group"] == i)
        group_data = data[group_filter]
        group_data_values = group_data.drop(['number', 'distant', 'rank', 'level', 'group'], axis=1)
        group_mean = group_data_values.mean()
        distant_array = np.zeros(shape=(level_counts[i],1))
        group_size = level_counts[i]
    
        for j in range(group_size):
            distant_array[j] = ((group_data_values.iloc[j] - group_mean) ** 2).sum()
    

        xsum = np.sum(distant_array)
        xsum_msw=xsum_msw+xsum    
    mean_msw = xsum_msw / (N-C)                     
    return mean_msw

def msb(data,C):
    data_value = data.drop(['number', 'distant', 'rank', 'level', 'group'], axis=1)  
    datamean=data_value.mean()
    level_counts =data.groupby("group").count().iloc[:, 0]
    xsum_msb=0
    distant_array= np.zeros((C,1))
    for i in range(C):
       group_filter = (data["group"] ==i)
       group_data = data[group_filter]
       group_data_values = group_data.drop(['number', 'distant', 'rank', 'level', 'group'], axis=1)
       group_mean = group_data_values.mean()
       group_size = level_counts[i]
       distant_array[i]=(group_size * ((group_mean - datamean) ** 2)).sum() 
    xsum = np.sum(distant_array)
    xsum_msb=xsum_msb+xsum
    mean_msb=xsum_msb/(C-1)
    return mean_msb
##此研究評估方法是利用F值倒數，組內/組間(補充:原本F值為組間/組內)
def F_reciprocal(data,C):
    MSW=msw(data,C) 
    MSB=msb(data,C) 
    F_reciprocal=MSW/MSB
    return F_reciprocal,MSW,MSB   

def cipple(num,C,N,atti,constrained_number,merge_output_path=None,append_to_existing=False):
  time_Start = time.time()#計時開始
  for v in range(1):
    result1 = np.zeros(shape=(num, 4))
    for t in range(num):
        data= pd.DataFrame(Dataset3(t, N, atti))
        N = data.shape[0]
        generate_constrained_student_list(constrained_number,N)
        constrained_data = pd.read_csv("data_no.csv")
       
        
        atti = data.shape[1]
    
        # print(data)

        # k=data.shape[1] #返回行數

        level= ma.ceil(N/C)
        
        datamean=data.mean() #平均值
        distant_array = np.zeros(shape=(N,1))
        
        for i in range(N):
            # distant_array[i] = (data.iloc[i, 0:atti]).sum()
            distant_array[i]=(((data.iloc[i]-datamean)**2).sum())**0.5

        distant_data=pd.DataFrame(distant_array)
        data["distant"]=distant_data
        data["number"]=range(len(data))
        data.sort_values("distant",inplace=True)
        data.index = range(len(data))
        data["rank"]=(data.distant.index)

        #分組分層
        data["level"]=((data["rank"]/C)).astype(int) # /:點數除法 浮點數轉整數 astype完成dataframe欄位型別轉換
        data["group"]=np.nan#將排名當作一個分群
        data.loc[data["level"]==0, "group"] = data["rank"]
        #print(data)

        distant = np.zeros(shape=(C))
        cos =  np.zeros(shape=(C))
        #Max = np.zeros(shape=(g,1))
        sen_cos = np.zeros(shape=(C))
        data_1=data[data["level"]==0].copy()
        cen= data_1
        swap__s= None ##交換的值
        for i in range(level-1):
             data_1=cen
             data_2 = data[data["level"]==i+1].copy()
             level_counts = data.groupby("level").count().iloc[:, 0]
             g = level_counts[i+1]
             a = C
            
             
             for j in range(g):
                 
                for l in range(a):
                    distant[l]=((data_1.iloc[l,0:atti]-data_2.iloc[j,0:atti])**2).sum()**0.5
                Max=np.max(distant) #找distant最大值來配對
                if len(np.where(distant == Max)[0]) == 1:
                    select_id = int(np.where(distant == Max)[0])
                else:
                    max_indices = np.where(distant == Max)[0]
                    cos = np.zeros(len(max_indices))
                    for k, l in enumerate(max_indices):
                        cos[k] = (data_1.iloc[l,0:atti]*data_2.iloc[j,0:atti]).sum()/((data_1.iloc[l,0:atti]**2).sum()**0.5)/(data_2.iloc[j,0:atti]**2).sum()**0.5
                    select_id = max_indices[np.argmin(cos)]
               
                # print(distant,Max)
                b=data_2.iloc[j]
                c=b.name
                d=data_1.iloc[select_id]#找最大遠and最小餘弦
                e=d.group
                data.loc[c,['group']]=e #最大值分到同群
                
                #檢查有如違反限制型
                groups = data.set_index('number')['group'].to_dict()
                constrained_exist = check_con(groups, constrained_data)
                if constrained_exist==False:
                   for _, row in constrained_data.iterrows():
                       students = row.dropna().astype(int).tolist()
                       constrained = data[data["number"].isin(students)].copy()
                       for group, number in constrained.iterrows():
                           same_group = constrained.groupby('group').filter(lambda x: len(x) >= 2).sort_values('rank')
                           one_constrained_group = constrained.groupby('group').filter(lambda x: len(x) == 1)
                       mask = (~data_1.group.isin(one_constrained_group["group"])) & (~data_1.group.isin(same_group["group"]))
                       filtered_data_group =data_1.loc[mask]
                       if  filtered_data_group.shape[0] >= 1:
                           for k in range(len(filtered_data_group)):
                               con_distant = np.zeros(len(filtered_data_group))
                               con_distant[k]=((filtered_data_group.iloc[k,0:atti]-data_2.iloc[j,0:atti])**2).sum()**0.5
                           sen=np.max(con_distant) #找distant最大值來配對
                           if len(np.where(con_distant == sen)[0]) == 1:
                               sen_select_id = int(np.where(con_distant == sen)[0])
                           else:
                               sen_indices = np.where(con_distant  == sen)[0]
                               sen_cos = np.zeros(len(sen_indices))
                               for k, l in enumerate(sen_indices):
                                   sen_cos[k] = (data_1.iloc[l,0:atti]*data_2.iloc[j,0:atti]).sum()/((data_1.iloc[l,0:atti]**2).sum()**0.5)/(data_2.iloc[j,0:atti]**2).sum()**0.5
                               sen_select_id = max_indices[np.argmin(sen_cos)]
                           b=data_2.iloc[j]                        
                           c=b.name
                           d=filtered_data_group.iloc[sen_select_id]
                           e=d.group
                           data.loc[c,['group']]=e #最大值分到同群
                           f = (filtered_data_group.iloc[sen_select_id,0:atti]+b.iloc[0:atti])/2 #已分配的層,群內取平均  
                           cen.iloc[cen["group"] == e, 0:atti] = f 
                           data_1=data_1.drop(data_1.index[data_1["group"]==filtered_data_group.iloc[sen_select_id]["group"]])        
                       else:
                           mask_2 = (~cen.group.isin(one_constrained_group["group"])) & (~cen.group.isin(same_group["group"]))
                           filtered_data_group_2 =cen.loc[mask_2]
                           b=data_2.iloc[j]                   
                           c=b.name
                           swap__s=filtered_data_group_2.iloc[0]
                           data.loc[c,['group']]=swap__s.group#跟前一筆樣本對調
                           d=data_1.iloc[select_id]#找最大值
                           e=d.group                            
                           data.loc[swap__s.name,['group']]=e #更新前一筆樣本的
                           swap__group_index = cen[cen['group'] == swap__s['group']].index[0]
                           cen.iloc[cen["group"] == swap__s.group, 0:atti] = ((((cen.iloc[swap__group_index , 0:atti])*2)-swap__s.iloc[0:atti])+b.iloc[0:atti])/2 
                           f = ((data_1.iloc[select_id,0:atti])+swap__s.iloc[0:atti])/2 #已分配的層,群內取平均  
                           cen.iloc[cen["group"] == e, 0:atti] = f 
                           data_1=data_1.drop(data_1.index[select_id])
                           
                       
                   
                else:
                   
                   f = ((data_1.iloc[select_id,0:atti])+b.iloc[0:atti])/2 #已分配的層,群內取平均  
                   cen.iloc[cen["group"] == e, 0:atti] = f 
                   data_1=data_1.drop(data_1.index[select_id])
                #分好的剔掉
                
                distant.fill(0.0)
                
                a = a-1
        F_reciprocal_value=F_reciprocal(data,C)
        # print(t, F_reciprocal_value)

        F_reciprocallist=list(range(3))
        F_reciprocal_array=np.array(F_reciprocallist)
        F_reciprocal_array=F_reciprocal_array.astype(float)
        F_reciprocal_array=F_reciprocal_value
        
        time_End = time.time() #計時結束
        result1[t,0]=F_reciprocal_array[0]
        result1[t,1]=F_reciprocal_array[1]
        result1[t,2]=F_reciprocal_array[2]
        result1[t,3]=time_End - time_Start 
    end = result1.sum(axis=0)/num          
    if merge_output_path is not None:
        combined_data = pd.concat([data.sort_values(by="group", ascending=True), cen], axis=1)
        if append_to_existing:
            with open(merge_output_path, 'a') as f:
                combined_data.to_csv(f, header=False, index=False)
        else:
            combined_data.to_csv(merge_output_path, index=False)
  
 
    
    return end[0], end[1], end[2],end[3]

