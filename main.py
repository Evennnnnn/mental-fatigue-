import os
import mne
import matplotlib.pyplot as plt
import numpy as np
import math
import pandas as pd
from datetime import datetime
dt01 = datetime.today()
from openpyxl import load_workbook

num = 26
date = '0'+str(datetime.today().month)+str(datetime.today().day)
# 额叶区的功率为F1 - 6、AF3 - 4, FP1 - 2, Fz, FPz电极的功率平均值，
# 中央区为FC1 - 6, C1 - 6, CP1 - 6, FCz, Cz电极的功率平均值，
# 顶叶区为P3 - 6, PO3 - 6, Pz, POz电极的功率平均值，
# 枕叶区为O1 - 2, Oz电极的功率平均值
channel = {
        'Frontal':['F1','F2','F3','F4','F5','F6','AF3','AF4','Fp1','Fp2','Fz','Fpz'], #12
        'Central':['FC1','FC2','FC3','FC4','FC5','FC6','C1','C2','C3','C4','C5','C6',
                   'CP1','CP2','CP3','CP4','CP5','CP6','FCz','Cz'], #20
        'Parietal':['P3','P4','P5','P6','PO3','PO4','PO5','PO6','Pz','POz'], #10
        'Occipital':['O1','O2','Oz'] #3
    }

group = ['no_rest', 'ASMR_rest', 'sitting_rest']

colorbar = {
    'AllBand': [[1, 40],[]],
    'Delta': [[1, 4],[]],
    'Theta': [[4, 8],[]],
    'LowerAlpha': [[8, 10],[]],
    'UpperAlpha': [[10, 12],[]],
    'Beta': [[12, 30],[]]
}

def get_filename(path, filetype):  # 输入路径、文件类型例如'.csv'
    file_list = []
    for root, dirs, files in os.walk(path):
        for i in files:
            if os.path.splitext(i)[1] == filetype:
                file_list.append(i)
    return file_list   # 输出由有后缀的文件名组成的列表

def find_samplepoints(raw, sfreq):
    # Extract the time_point of four events
    events = mne.events_from_annotations(raw)
    event_60 = events[1]['60']
    event_96 = 10
    if '96' in events[1]:
        event_96 = events[1]['96']
    epoch_span = 180 * sfreq  # 3分钟

    last_60 = 0
    flag_60 = 0  # do not find 60
    flag_96 = 0
    sample_point = []
    for n, event in enumerate(events[0]):
        if event[2] == event_60 and not flag_60:
            sample_point.append(event[0])
            flag_60 = 1
        elif event[2] == event_96:
            sample_point.append(events[0][n - 1][0] - epoch_span)
            flag_96 = 1
        elif event[2] == event_60 and flag_96:
            sample_point.append(event[0])
            flag_96 = 0
        elif event[2] == event_60:
            last_60 = event[0]

    if len(sample_point) == 1:
        sample_point.append(int((sample_point[0] + last_60)/2-epoch_span))
        sample_point.append(int((sample_point[0] + last_60)/2))
    sample_point.append(last_60 - epoch_span)
    return sample_point


def psd():
    # 保存4个起始时间点，判断时间段分割是否正确
    all_time = {}

    # 保存1-40Hz和五个频段的psd数据，有需要的话后面可以画psd曲线
    psd_list = {}
    raw_psd = {}

    # 计算4个频段的数据


    # 按四个频段循环
    for freq_num, freq_name in enumerate(['AllBand', 'Delta', 'Theta', 'LowerAlpha', 'UpperAlpha', 'Beta'], 1):
        psd_list[freq_name]={}

        # 按三个组别
        for group_num, group_name in enumerate(group, 1):
            group_path = f'./data/group{group_num}'
            file_list = get_filename(group_path, '.set')

            psd_list[freq_name][group_name] = [[],[],[],[],[]]
            psd_list[freq_name][group_name][0] = file_list

            # 新建文件夹保存图片
            new_file = f'{group_path}/wavespan'
            if not os.path.exists(new_file):
                os.mkdir(new_file)

            # 每个人循环
            for file_num, file_name in enumerate(file_list) :
                print(freq_name, group_name, file_num)
                real_file_name = os.path.splitext(file_name)[0]
                raw = mne.io.read_raw_eeglab(os.path.join(group_path, file_name), preload=False)
                sfreq = int(raw.info['sfreq'])
                epoch_span = 180 * sfreq  # 3分钟

                # 获得四个时间段的起始时间点
                sample_points = find_samplepoints(raw, sfreq)
                time_points = [int(points / sfreq) for points in sample_points]
                all_time[f'{real_file_name}'] = time_points

                # 四个时间段循环
                for time_loop, time_point in enumerate(time_points, 1) :
                    print(time_loop, time_point)

                    psd = raw.compute_psd(fmin=colorbar[freq_name][0][0], fmax=colorbar[freq_name][0][1],
                                          tmin=time_point, tmax=time_point + epoch_span,
                                          n_fft = 1000, n_overlap=100)
                    raw_psd = psd

                    if freq_num == 1:
                        data_list = psd._data
                    else:
                        data_list = [10 * np.log10(data * 10 ** 12) for data in psd._data.mean(1)]
                    psd_list[freq_name][group_name][time_loop].append(data_list)

                    print( id(psd_list[freq_name][group_name][time_loop]) )
                    print(id(psd_list[freq_name][group_name][time_loop+1]))
                    a=0
                    """# 画每个人的五个频段的图
                    fig = psd.plot_topomap(cmap='RdYlBu', show = False, dB=True)
                    plt.savefig(f'{new_file}/{real_file_name}__time{time_loop}.png', bbox_inches='tight', dpi = 600)
                    plt.close()
                    """

    np.save(date + 'time_points.npy', all_time)

    psd_list['raw_psd'] = raw_psd
    np.save(date + 'psd.npy', psd_list)


# 已知完整psd文件，画平均脑电图
def plot_final_psd():

    psd_list = np.load('0410psd.npy', allow_pickle=True).item()
    raw_psd = psd_list['raw_psd']
    average_psd = {}
    for group_num, group_name in enumerate(group, 1):
        group_psd = psd_list[group_name]
        average_psd[group_name] = [[], [], [], [], []]
        average_psd[group_name][0] = group_psd[0]
        for time_loop in [1, 2, 3, 4]:
            psd = np.zeros((59, 82))
            for file_num, file_name in enumerate(group_psd[0], 0):
                psd = np.add(psd, group_psd[time_loop][file_num])
            psd = [data/len(group_psd[time_loop]) for data in psd]
            psd = np.array(psd)
            average_psd[group_name][time_loop] = psd
            raw_psd._data = psd
            for freq_span in ['Delta', 'Theta', 'Alpha', 'Beta']:
                fig = raw_psd.plot_topomap(dB=True, cmap='RdYlBu', bands = {f'{freq_span}_T{time_loop}':colorbar[freq_span][1]}, show = False)
                fig.savefig(f'./data/group{group_num}/0407_{freq_span}_time{time_loop}.png', bbox_inches='tight', dpi=600)
                plt.close(fig)

    # np.save('average_psd.npy', average_psd)

    # bands = {'Delta (0-4 Hz)': (0, 4), 'Theta (4-8 Hz)': (4, 8),
    #          'Alpha (8-12 Hz)': (8, 12), 'Beta (12-30 Hz)': (12, 30),
    #          'Gamma (30-45 Hz)': (30, 45)}


# 画统计图
def plot_satistic():
    area = ['Frontal', 'Central', 'Parietal', 'Occipital']
    fig, axs = plt.subplots(nrows=4, ncols=4, figsize=(20, 20))

    raw = mne.io.read_raw_eeglab(
        'D:\\reservation\SRTP\data_process\EEG_process\ASMR_rest\\1-1.set')
    ch_names = raw.ch_names
    average = {}
    for freq_num, freq_name in enumerate(['Delta', 'Theta', 'Alpha', 'Beta'], 1):
        average[freq_name] = [[]]*4
        wave_span = np.load( freq_name + '.npy', allow_pickle=True).item()
        for time_loop in [1, 2, 3, 4]:
            barea_list = list()

            data_df = pd.DataFrame(wave_span[time_loop], columns=ch_names)  # 利用pandas库对数据进行格式转换
            for ch_num, ch_name in enumerate(area, 1):
                barea_list.append(data_df[channel[ch_name]].mean(1))

            barea_list1 = pd.concat(barea_list, axis=1)

            temp = list()
            temp.append(barea_list1.iloc[0:num].mean(axis=0))
            temp.append(barea_list1.iloc[num : num*2].mean(axis=0))
            temp.append(barea_list1.iloc[num*2 : num*3].mean(axis=0))
            temp = pd.concat(temp, axis=1)
            temp = temp.T.to_numpy()
            average[freq_name][time_loop-1]=temp

            axs[freq_num-1, time_loop-1].plot(area, temp[0], area, temp[1], area, temp[2])
            axs[freq_num-1, time_loop-1].set_title(freq_name + f'_T{time_loop}' )

    for ax in axs.flat:
        ax.set_ylabel('dB')
        ax.legend(group)
    fig.subplots_adjust(hspace=0.4)
    plt.show()


if __name__ == '__main__':
    print(date)
    # cal_psd()
    # save_psd()
    # plot_final_psd()
    # plot_satistic()
    psd()


    # title = 'psd'  # 读取的文件名
    # data = np.load(title + '.npy', allow_pickle=True).item()  # 读取numpy文件
    # data_df = pd.DataFrame(data)  # 利用pandas库对数据进行格式转换
    #
    # # create and writer pd.DataFrame to excel
    # writer = pd.ExcelWriter(title + '.xlsx')  # 生成一个excel文件
    # data_df.to_excel(writer, 'page_1')  # 数据写入excel文件
    # writer._save()  # 保存excel文件
