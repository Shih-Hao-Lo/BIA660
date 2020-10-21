# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:47:30 2020

@author: ME671KP
"""

from selenium import webdriver
from selenium.webdriver.support.ui import Select
import codecs
import time
from datetime import date
import re
import math
import matplotlib.pyplot as plt
import numpy 

def createarr(init):
    arr = []
    for x in range(0, 10):
        arr.append([])
        for y in range(0, 12):
            arr[x].append(init)
    return arr

def create1darr(init):
    arr = []
    for x in range(0, 12):
        arr.append(init)
    return arr

def storedata(data, dic, cntdic, year, month, targetyear):
    name = data[0]
    if data[4] == '--':
        data[4] = '0'
    else:
        data[4] = re.sub('[^\d.]','', data[4])
    #print(data[4])    
    waterLevel = float(data[4])
    #print('store at: dict[' + name + '][' + str(year-targetyear-1) + '][' + str(month-1) +'] = ' + str(waterLevel))
    print('store at: dict[' + name + '][' + str(year) + '][' + str(month) +'] = ' + str(waterLevel))
    if not name in dic:
        dic[name] = createarr(0)
    if not name in cntdic:
        cntdic[name] = create1darr(0)
    arr = dic[name]
    cntarr  = cntdic[name]
    if waterLevel != 0:
        cntarr[month-1] += 1
    arr[year-targetyear-1][month-1] += waterLevel

try:
    today = date.today()
    #today = date(2019, 5, 15)
    
    url = 'http://fhy.wra.gov.tw/ReservoirPage_2011/Statistics.aspx'
    
    driver = webdriver.Chrome('./chromedriver')
    driver.get(url)
    
    time.sleep(1)
    
    # basic option
    selectAll = Select(driver.find_element_by_name('ctl00$cphMain$cboSearch'))
    selectAll.select_by_value('防汛重點水庫')
    
    selectHour = Select(driver.find_element_by_name('ctl00$cphMain$ucDate$cboHour'))
    selectHour.select_by_value('0')
    
    selectMinute = Select(driver.find_element_by_name('ctl00$cphMain$ucDate$cboMinute'))
    selectMinute.select_by_value('0')
    
    time.sleep(1)    
    
    year = today.year-1
    month = 12
    
    startyear = today.year - 1
    targetyear = year - 10;
    
    dic = {}
    cntdic = {}
    meandic = {}
    
    tagarr = []
    fullDataList = []
    
    fw=codecs.open('rpa.txt','w',encoding='utf8')
    
    delay = 1
    
    while year > targetyear:
    #while year >= 2018:
        print("Getting Data at " + str(year) + " / " + str(month))
        # set search date to today
        selectYear = Select(driver.find_element_by_name('ctl00$cphMain$ucDate$cboYear'))
        selectYear.select_by_value(str(year))
        
        time.sleep(delay)
        
        selectMonth = Select(driver.find_element_by_name('ctl00$cphMain$ucDate$cboMonth'))
        selectMonth.select_by_value(str(month))
        
        time.sleep(delay)
        
        selectDay = Select(driver.find_element_by_name('ctl00$cphMain$ucDate$cboDay'))
        selectDay.select_by_value("1")
        
        time.sleep(1)
        
        delay = 0
        
        # start search
        searchButton = driver.find_element_by_id('ctl00_cphMain_btnQuery')
        searchButton.click()
        
        time.sleep(2)
        
        dataList = driver.find_elements_by_tag_name("tbody")
        #print(len(dataList))
        #print(dataList)
        
        fullarr = []
        
        for data in dataList:
            trList = data.find_elements_by_tag_name("tr")
            #print(len(trList))
            cnt = 0;
            for tr in trList:
                cnt+=1
                if cnt <= 4 or cnt == len(trList):
                    if cnt == 1:
                        tdList = tr.find_elements_by_tag_name("th")
                        cnt2 = 0
                        for td in tdList:
                            cnt2+=1
                            if cnt2 < len(tdList):
                                tagarr.append(td.text.replace("\n", ""))
                    elif cnt == 2:
                        tdList = tr.find_elements_by_tag_name("th")
                        cnt2 = 0
                        for td in tdList:
                            cnt2+=1
                            if cnt2 < len(tdList)-1:
                                tagarr.append(td.text.replace("\n", ""))
                    elif cnt == 3:
                        tdList = tr.find_elements_by_tag_name("th")
                        cnt2 = 0
                        for td in tdList:
                            tagarr.append(td.text.replace("\n", ""))
                        #print(tagarr)
                else:
                    #print(tr.text)
                    arr = []
                    tdList = tr.find_elements_by_tag_name("td")
                    for td in tdList:
                        arr.append(td.text)
                    #print(arr)
                    fw.write(str(arr) + '\n')
                    storedata(arr, dic, cntdic, year, month, targetyear)
                    fullarr.append(arr)
        fullDataList.append(fullarr)
        print('/////////////////////////////////////////////////')
        month -= 1
        if month == 0:
            month = 12
            year -= 1
        
    #print(tagarr)
    #print(fullarr)
    
    #calculate mean
    for key in dic:
        #for x in range(0, 10):
        #    for y in range(0, 12):
        #        dic[key][x][y] /= cntdic[key][x][y]
        meandic[key] = create1darr(0)
        for x in range(0, 10):
            for y in range(0, 12):
                meandic[key][y] += dic[key][x][y]
        
        for x in range(0, 12):
            meandic[key][x] /= cntdic[key][x]
        
        print(key + ":")
        for row in dic[key]:
            print(row)
        
    #for key in meandic:
    #    print(key + " mean:")
    #    print(meandic[key])
        
    sddic = {}
    
    #calculate standard difference
    for key in dic:
        sddic[key] = create1darr(0)
        for x in range(0, 10):
            for y in range(0, 12):
                sddic[key][y] += (dic[key][x][y] - meandic[key][y]) * (dic[key][x][y] - meandic[key][y])
        for x in range(0, 12):
            sddic[key][x] /= (cntdic[key][x] - 1)
            sddic[key][x] = math.sqrt(sddic[key][x])
            
    for key in dic:
        print(key + ":")
        for x in range(0, 12):
            print('month ' + str(x+1) + ":total = " + str(meandic[key][x]*cntdic[key][x]) + ", cnt = " + str(cntdic[key][x]) + ", mean = " + str(meandic[key][x]) + ", standard diff = " + str(sddic[key][x]))
        
    keys = ['石門水庫']#, '翡翠水庫']
    
    for key in keys:
        datas = numpy.array(dic[key])
        means = meandic[key]
        sdfs = sddic[key]
        idx = range(targetyear+1, targetyear+11)
        for x in range(12):
            data = datas[:,x]
            mean = means[x]
            sdf = sdfs[x]
            plt.plot(idx, data)
            plt.xlabel('water level') 
            plt.ylabel('year') 

            # giving a title to my graph 
            plt.title(key + 'water level history at month ' + str(x+1)) 

    fw.close()
    driver.quit()#close the browser
except Exception as e:
    fw.close()
    print(e)
    driver.quit()#close the browser