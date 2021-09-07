import os
import posixpath as path
from time import time
import h5py
import pandas as pd
import numpy as np
import xarray as xr
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from scipy.interpolate import griddata
from scipy.interpolate import make_interp_spline
import cmaps
import cartopy.crs as ccrs
import cartopy.mpl.ticker as cticker
import cartopy.io.shapereader as shpreader


def Get_point_index(LatLonFile_name='/Volumes/innovation/Dong_work/data/lat_lon.dat'):
    """
    读lat_lon.dat, 以便之后根据index得到该点的经纬度值, index从0开始
    """
    df = pd.read_csv(LatLonFile_name, sep='\s+', header=None, names=['lat', 'lon']).reset_index()
    df['lat'] = df['lat'].round(6)  # 统一保持7位小数
    df['lon'] = df['lon'].round(6)  # 统一保持7位小数
    # print(df)
    return df


def grid_interp_to_station(grid_var, grid_lon, grid_lat, station_lon, station_lat, method='linear'):  # 'cubic'
    """
    func: 将等经纬度网格值 插值到 离散站点。使用griddata进行插值
    inputs:
        [经度网格，纬度网格，数值网格]; shape is (x,y)=>(x*y, 1)
        station_lon: 站点经度
        station_lat: 站点纬度; shape is (m, n)=>(m*n, 1)
        method: 插值方法,默认使用 cubic
    outputs:
        var_data:[u,v,w]
    """

    grid_lon = np.array(grid_lon).reshape(-1, 1)
    grid_lat = np.array(grid_lat).reshape(-1, 1)
    grid_var = np.array(grid_var).reshape(-1, 1)

    station_lon = np.array(station_lon).reshape(-1, 1)
    station_lat = np.array(station_lat).reshape(-1, 1)

    # points = np.hstack((grid_lon, grid_lat))
    points = np.concatenate([grid_lon, grid_lat], axis=1)
    station_value = griddata(points, grid_var, (station_lon, station_lat), method=method)
    station_value = station_value[:, :, 0]
    df = pd.DataFrame(station_value)
    # print(df.shape)
    # df.fillna(method='ffill', axis=0, inplace=True)
    station_value = df  # .values.flatten()  # .reshape(96, 68)

    return station_value


def Aer_prof(aer_prof_file, point_index, point_range, plot_pic):
    """
    读取2018/11/14 00:00UTC 气溶胶廓线文件,得到与lat-lon.dat中index一致的点上的廓线数据，并画图
    :param plot_pic: 决定是否画图
    :param aer_prof_file:
    :param point_index:  取第几个点，从0开始，与lat-lon文件的index一致
    :param point_range: 一共取几个点的数据
    :return: df & plot
    """
    df_points = Get_point_index()
    point_lat = format(float(df_points['lat'][df_points['index'] == point_index].values), '.2f')
    point_lon = format(float(df_points['lon'][df_points['index'] == point_index].values), '.2f')

    Aer_prof_single_point = []
    with open(aer_prof_file, 'rb') as f:
        data = f.readlines()  # 将txt中所有字符串读入data
        for i in range(51 * point_index + 7, 51 * (point_index + point_range) + 1, 51):  # 每97行读一次
            print('i is ', i)
            print('######')
            for j in range(i, i + 48, 2):
                # print('j is ', j)
                level1_fst_line = list(map(float, data[j - 1].split()))
                level1_sec_line = list(map(float, data[j].split()))
                level_1 = level1_fst_line + level1_sec_line
                # print(level_1)
                Aer_prof_single_point.append(level_1)
        df_Aer_prof_single_point = pd.DataFrame(np.array(Aer_prof_single_point), columns=['bcar', 'dus1', 'dus2',
                                                                                          'dus3', 'sulp', 'ssa1',
                                                                                          'ssa2', 'ssa3', 'omat'])
        print(df_Aer_prof_single_point)
        # df_Aer_prof_single_point.to_excel("Aerosol_profiles_lat" + point_lat + "_lon" + point_lon + ".xlsx",
        #                                   index=False)
        if plot_pic == 'True':
            fig = plt.figure(figsize=(6, 8), dpi=300)
            ax = fig.add_subplot(111)

            df = df_Aer_prof_single_point
            df['Pressure'] = pd.DataFrame(
                {'Pressure': [1.5, 2.5, 4, 6, 8.5, 15, 25, 40, 60, 85, 125, 175, 225, 275, 350, 450,
                              550, 650, 750, 825, 875, 912.5, 937.5, 975]})

            l1 = plt.plot(df['bcar'], df['Pressure'], 'k', label="BC", linewidth=3)
            l2 = plt.plot(df['dus1'], df['Pressure'], 'y', label="DU_1", linewidth=3)
            l3 = plt.plot(df['dus2'], df['Pressure'], 'y', label="DU_2", linewidth=3)
            l4 = plt.plot(df['dus3'], df['Pressure'], 'y', label="DU_3", linewidth=3)
            l5 = plt.plot(df['sulp'], df['Pressure'], 'r', label="SU", linewidth=3)
            l6 = plt.plot(df['ssa1'], df['Pressure'], 'c', label="SS_1", linewidth=3)
            l7 = plt.plot(df['ssa2'], df['Pressure'], 'c', label="SS_2", linewidth=3)
            l8 = plt.plot(df['ssa3'], df['Pressure'], 'c', label="SS_3", linewidth=3)
            l9 = plt.plot(df['omat'], df['Pressure'], 'Purple', label="OM", linewidth=3)
            plt.legend(prop={'size': 14})
            # plt.ylim(1.5, 1000)
            # ax.invert_yaxis()
            # ax.set_yscale('symlog')
            # ax.set_yticks([10, 50, 100, 200, 300, 500, 700, 1000])
            # ax.set_yticklabels(['10', '50', '100', '200', '300', '500', '700', '1000'])

            plt.ylim(400, 1000)
            ax.invert_yaxis()
            ax.set_yscale('symlog')
            ax.set_yticks([450, 550, 650, 750, 850, 950, 1000])
            ax.set_yticklabels(['350', '450', '550', '650', '750', '850', '950', '1000'])

            plt.title(
                "Vertical profiles of aerosol chemical composition \n "
                "20181114 00:00UTC lat:" + point_lat + " lon:" + point_lon)  # 43158  29.989041  104.041768
            # plt.savefig('Aersol_profile_2018111400UTC_lat2998_lon10404'+'.png', dpi=300)  # 'Sichuan_jacobian_CH1.png'
            plt.show()
    return None


def Aer_Prof_AllPoints_VerticalIntegration(aer_prof_file, point_index, point_range):
    """
    读取2018/11/14 00:00UTC 气溶胶廓线文件,得到与lat-lon.dat中index一致的点上的廓线数据，并画图
    :param aer_prof_file:
    :param point_index:  取第几个点，从0开始，与lat-lon文件的index一致
    :param point_range: 一共取几个点的数据
    :return: df & plot
    """

    Aer_SinglePoint_VerticalIntegration = {}
    with open(aer_prof_file, 'rb') as f:
        data = f.readlines()  # 将txt中所有字符串读入data
        # for i in range(51 * point_index + 7, 51 * (point_index + point_range) + 1, 51):  # 每51行读一次
        for i in range(51 * point_index + 7, 51 * (point_index + point_range) + 1, 51):  # 每51行读一次
            print('i is ', i)
            print('######')
            Aer_prof_single_point = []
            for j in range(i, i + 48, 2):
                # print('j is ', j)
                level1_fst_line = list(map(float, data[j - 1].split()))
                level1_sec_line = list(map(float, data[j].split()))
                level_1 = level1_fst_line + level1_sec_line
                # print(level_1)
                Aer_prof_single_point.append(level_1)
            df_Aer_prof_single_point = pd.DataFrame(np.array(Aer_prof_single_point), columns=['bcar', 'dus1', 'dus2',
                                                                                              'dus3', 'sulp', 'ssa1',
                                                                                              'ssa2', 'ssa3', 'omat'])
            # print(df_Aer_prof_single_point)
            Aer_SinglePoint_VerticalIntegration[i] = pd.DataFrame(df_Aer_prof_single_point.sum()).T
    AerPoints_VerticalIntegration = pd.concat(list(Aer_SinglePoint_VerticalIntegration.values()),
                                              ignore_index=True, axis=0)
    return AerPoints_VerticalIntegration


class Jacobian(object):
    """
    """

    def __init__(self, point_index, point_range, adjoint_file, channel_num):
        """
        :param point_index:  想取哪个点，对应lat-lon文件的第几行(减一)
        :param point_range: 从选取点开始 总共要读取几个点
        :return: dataframe with 9 cols, Each col represents an aerosol category
        """
        super(Jacobian, self).__init__()
        df_points = Get_point_index()
        self.point_index = point_index
        self.point_lat = format(float(df_points['lat'][df_points['index'] == point_index].values), '.2f')
        self.point_lon = format(float(df_points['lon'][df_points['index'] == point_index].values), '.2f')
        self.point_range = point_range
        self.adjoint_file = adjoint_file
        self.channel_num = channel_num

    def Read_Jacobian_OneCH(self):
        """

        """
        Jacobian_bcar_points = []
        Jacobian_dus1_points = []
        Jacobian_dus2_points = []
        Jacobian_dus3_points = []
        Jacobian_sulp_points = []
        Jacobian_ssa1_points = []
        Jacobian_ssa2_points = []
        Jacobian_ssa3_points = []
        Jacobian_omat_points = []
        with open(self.adjoint_file, 'rb') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for i in range(97 * (self.point_index - 1) + 1, 97 * (self.point_index + self.point_range - 1) + 1,
                           97):  # 每97行读一次
                print('i is ', i)
                for j in range(0, 97 - 1, 4):
                    # print(i + j)
                    # print(data[i + j])
                    Jacobian_bcar = list(map(float, data[i + j + 1].split()))[0]
                    Jacobian_dus1 = list(map(float, data[i + j + 1].split()))[1]
                    Jacobian_dus2 = list(map(float, data[i + j + 1].split()))[2]

                    Jacobian_dus3 = list(map(float, data[i + j + 2].split()))[0]
                    Jacobian_sulp = list(map(float, data[i + j + 2].split()))[1]
                    Jacobian_ssa1 = list(map(float, data[i + j + 2].split()))[2]

                    Jacobian_ssa2 = list(map(float, data[i + j + 3].split()))[0]
                    Jacobian_ssa3 = list(map(float, data[i + j + 3].split()))[1]
                    Jacobian_omat = list(map(float, data[i + j + 3].split()))[2]

                    Jacobian_bcar_points.append(Jacobian_bcar)
                    Jacobian_dus1_points.append(Jacobian_dus1)
                    Jacobian_dus2_points.append(Jacobian_dus2)

                    Jacobian_dus3_points.append(Jacobian_dus3)
                    Jacobian_sulp_points.append(Jacobian_sulp)
                    Jacobian_ssa1_points.append(Jacobian_ssa1)

                    Jacobian_ssa2_points.append(Jacobian_ssa2)
                    Jacobian_ssa3_points.append(Jacobian_ssa3)
                    Jacobian_omat_points.append(Jacobian_omat)

            Jacobian_bcar_all = pd.DataFrame({'Jacobian_bcar': Jacobian_bcar_points})
            Jacobian_dus1_all = pd.DataFrame({'Jacobian_dus1': Jacobian_dus1_points})
            Jacobian_dus2_all = pd.DataFrame({'Jacobian_dus2': Jacobian_dus2_points})
            Jacobian_dus3_all = pd.DataFrame({'Jacobian_dus3': Jacobian_dus3_points})
            Jacobian_sulp_all = pd.DataFrame({'Jacobian_sulp': Jacobian_sulp_points})
            Jacobian_ssa1_all = pd.DataFrame({'Jacobian_ssa1': Jacobian_ssa1_points})
            Jacobian_ssa2_all = pd.DataFrame({'Jacobian_ssa2': Jacobian_ssa2_points})
            Jacobian_ssa3_all = pd.DataFrame({'Jacobian_ssa3': Jacobian_ssa3_points})
            Jacobian_omat_all = pd.DataFrame({'Jacobian_omat': Jacobian_omat_points})
            # print(Jacobian_bcar_all)
            # print(Jacobian_omat_all)
            Jacobian_aer_all = pd.concat([Jacobian_bcar_all, Jacobian_dus1_all, Jacobian_dus2_all,
                                          Jacobian_dus3_all, Jacobian_sulp_all, Jacobian_ssa1_all,
                                          Jacobian_ssa2_all, Jacobian_ssa3_all, Jacobian_omat_all], axis=1)

            # print(Jacobian_aer_all.shape)
            # print(Jacobian_aer_all)

            return Jacobian_aer_all

    def plot_Jacobian(self):
        df = self.Read_Jacobian_OneCH()
        fig = plt.figure(figsize=(6, 8), dpi=300)
        ax = fig.add_subplot(111)

        df['Pressure'] = pd.DataFrame(
            {'Pressure': [1.5, 2.5, 4, 6, 8.5, 15, 25, 40, 60, 85, 125, 175, 225, 275, 350, 450,
                          550, 650, 750, 825, 875, 912.5, 937.5, 975]})

        l1 = plt.plot(df['Jacobian_bcar'], df['Pressure'], 'k', label="BC", linewidth=3)
        l2 = plt.plot(df['Jacobian_dus1'], df['Pressure'], 'y', label="DU_1", linewidth=3)
        l3 = plt.plot(df['Jacobian_dus2'], df['Pressure'], 'y', label="DU_2", linewidth=3)
        l4 = plt.plot(df['Jacobian_dus3'], df['Pressure'], 'y', label="DU_3", linewidth=3)
        l5 = plt.plot(df['Jacobian_sulp'], df['Pressure'], 'r', label="SU", linewidth=3)
        l6 = plt.plot(df['Jacobian_ssa1'], df['Pressure'], 'c', label="SS_1", linewidth=3)
        l7 = plt.plot(df['Jacobian_ssa2'], df['Pressure'], 'c', label="SS_2", linewidth=3)
        l8 = plt.plot(df['Jacobian_ssa3'], df['Pressure'], 'c', label="SS_3", linewidth=3)
        l9 = plt.plot(df['Jacobian_omat'], df['Pressure'], 'Purple', label="OM", linewidth=3)
        plt.legend(prop={'size': 14})
        plt.ylim(1.5, 1000)
        ax.invert_yaxis()
        # ax.set_yscale('s/ymlog')  # 由于气溶胶集中在边界层内，故不取log
        ax.set_yticks([10, 50, 100, 200, 300, 500, 700, 1000])
        ax.set_yticklabels(['10', '50', '100', '200', '300', '500', '700', '1000'])

        # plt.ylim(400, 1000)
        # ax.invert_yaxis()
        # ax.set_yscale('symlog')
        # ax.set_yticks([450, 550, 650, 750, 850, 950, 1000])
        # ax.set_yticklabels(['350', '450', '550', '650', '750', '850', '950', '1000'])
        # plt.title('CH' + channel_num + ' 20181114 00:00UTC lat:39.20 lon:117.35')  # 173629  39.201607  117.351584
        plt.title('CH' + self.channel_num + " 20181114 00:00UTC lat:" + self.point_lat + " lon:" + self.point_lon)
        # plt.savefig('JJJ_jacobian_CH' + self.channel_num + '.png', dpi=300)  # 'JJJ_jacobian_CH1.png'
        plt.show()
        return None


class AerosolSimulate(object):
    """docstring for ClassName"""

    def __init__(self, SimulationFile, AGRIFile, CLMFile, LatLonFile, CAMSAODFile, FY4RawFile,
                 SouthLat, NorthLat, WestLon, EastLon, time_select):
        super(AerosolSimulate, self).__init__()
        self.SouthLat = SouthLat
        self.NorthLat = NorthLat
        self.WestLon = WestLon
        self.EastLon = EastLon
        self.SimulationFile = SimulationFile
        self.AGRIFile = AGRIFile
        self.LatLonFile = LatLonFile
        self.FY4RawFile = FY4RawFile
        self.CLMFile = CLMFile
        self.time_select = time_select
        self.CAMSAODFile = CAMSAODFile

    def Get_BTorRadiation(self):
        len_file_lins = len(open(self.SimulationFile).readlines())  # SimulationFile lines 1568820

        BT = []
        Rad = []
        with open(self.SimulationFile, 'r') as f:
            data = f.readlines()  # 将txt中所有字符串读入data
            for i in range(1, len_file_lins, 6):  # 每6行读一次
                BT_fst_line = list(map(float, data[i].split()))  # str转化为浮点数
                BT_sec_line = list(map(float, data[i + 1].split()))
                BT_i = BT_fst_line + BT_sec_line
                # print(BT_i)
                BT.append(BT_i)

                Rad_fst_line = list(map(float, data[i + 3].split()))
                Rad_sec_line = list(map(float, data[i + 4].split()))
                Rad_i = Rad_fst_line + Rad_sec_line
                Rad.append(Rad_i)
            BT_obs = np.array(BT)  # (261470, 13) 13个Channel 每个channel 261470个数据
            Rad_obs = np.array(Rad)

        obs_data = np.hstack((Rad_obs[:, :6], BT_obs[:, 6:]))  # Channel 1~6 是反射值， Channel 7~14是亮度温度
        df1 = pd.DataFrame(obs_data, columns=["Channel_simu_1", "Channel_simu_2", "Channel_simu_3",
                                              "Channel_simu_4", "Channel_simu_5", "Channel_simu_6",
                                              "Channel_simu_7_8", "Channel_simu_9", "Channel_simu_10",
                                              "Channel_simu_11", "Channel_simu_12", "Channel_simu_13",
                                              "Channel_simu_14"])

        df2 = pd.read_csv(self.LatLonFile, sep='\s+', header=None, names=['lat', 'lon'])
        df2 = df2.round(6)  # 统一保持7位小数
        df_result = pd.concat([df1, df2], axis=1)
        return df_result

    def Get_AGRI_LatLon_region(self):
        LatLonFile = open(self.FY4RawFile, "rb")
        geo = np.fromfile(LatLonFile, dtype='<f8')  # (15103008,) = 2748 x 2748 x 2
        lat_geo, lon_geo = geo[::2], geo[1::2]
        return lat_geo, lon_geo

    def Get_AGRI_data_OneChannel(self, Channel):
        file = h5py.File(self.AGRIFile, "r")
        IR_channel_grey = file['NOMChannel' + str(Channel)][:].flatten()  # (2748, 2748) => (7551504,)
        IR_channel_cal = file['CALChannel' + str(Channel)][:]  # (4096,)
        index_ch = np.where(IR_channel_grey == 65535, 0, IR_channel_grey)  # (7551504,)
        obs_data = np.where(IR_channel_grey == 65535, np.nan, IR_channel_cal[index_ch])

        lat_geo, lon_geo = self.Get_AGRI_LatLon_region()
        a = np.array([obs_data, lat_geo, lon_geo]).transpose()
        df = pd.DataFrame(a, columns=["obs_value", "lat", "lon"])
        df_new = df[(df['lat'] > self.SouthLat) & (df['lat'] < self.NorthLat)
                    & (df['lon'] > self.WestLon) & (df['lon'] < self.EastLon)]
        return df_new["obs_value"]

    def Get_AGRI_data_ALLChannel(self):
        obs_all_channel = []
        for i in range(1, 14 + 1):
            Channel_i = str(i).rjust(2, '0')
            obs_channel_i = self.Get_AGRI_data_OneChannel(Channel_i)
            obs_all_channel.append(obs_channel_i)

        obs_all_channel = np.array(obs_all_channel).transpose()
        df1 = pd.DataFrame(obs_all_channel,
                           columns=["Channel_AGRI_1", "Channel_AGRI_2", "Channel_AGRI_3", "Channel_AGRI_4",
                                    "Channel_AGRI_5", "Channel_AGRI_6", "Channel_AGRI_7", "Channel_AGRI_8",
                                    "Channel_AGRI_9", "Channel_AGRI_10", "Channel_AGRI_11", "Channel_AGRI_12",
                                    "Channel_AGRI_13", "Channel_AGRI_14"])
        lat_geo, lon_geo = self.Get_AGRI_LatLon_region()
        a = np.array([lat_geo, lon_geo]).transpose()
        df2 = pd.DataFrame(a, columns=["lat", "lon"])
        df2_ = df2[(df2['lat'] > self.SouthLat) & (df2['lat'] < self.NorthLat)
                   & (df2['lon'] > self.WestLon) & (df2['lon'] < self.EastLon)].reset_index(
            drop=True)
        df2_ = df2_.round(6)  # 统一保持7位小数
        df_result = pd.concat([df1, df2_], axis=1)
        return df_result

    def Get_AGRI_CLM(self):
        file = xr.open_dataset(self.CLMFile)
        CLM = file["CLM"].values.flatten()
        CLM[CLM == 126] = 4  # space:126 => 4 外太空
        CLM[CLM == 127] = 5  # fillvalue:127 => 5 无数据区
        lat_geo, lon_geo = self.Get_AGRI_LatLon_region()

        a = np.array([CLM, lat_geo, lon_geo]).transpose()
        df = pd.DataFrame(a, columns=["CLM", "lat", "lon"])
        df_new = df[(df['lat'] > self.SouthLat) & (df['lat'] < self.NorthLat)
                    & (df['lon'] > self.WestLon) & (df['lon'] < self.EastLon)]
        df_new = df_new.round({'lat': 6, 'lon': 6})  # lat,lon列保持7位小数
        return df_new

    def Get_CAMS_AOD_data(self, var_name):
        file = xr.open_dataset(self.CAMSAODFile)
        file_s = file.sel(time=self.time_select)
        data = file_s[var_name]
        Lon = data.longitude.values
        Lat = data.latitude.values
        lon_, lat_ = np.meshgrid(Lon, Lat)
        data_ = data.values
        return lon_, lat_, data_

    def Get_CAMS_Interpolation(self):
        df = pd.read_excel('CMASS_Aer_AllPoints_VerticalIntegration_Frac.xlsx')
        return df

    def Merge_AGRI_AerSimulate_CLMMask_AOD(self):
        df_CLM = self.Get_AGRI_CLM()
        df_Aerosol = self.Get_CAMS_Interpolation()
        df_SimulateRadBT = self.Get_BTorRadiation()
        df_AGRI = self.Get_AGRI_data_ALLChannel()
        df_Aer = pd.concat([df_Aerosol, df_SimulateRadBT], axis=1)
        df_Aer_CLM = pd.merge(df_CLM, df_Aer, on=['lat', 'lon'])
        df_AGRI_Aer_CLM = pd.merge(df_Aer_CLM, df_AGRI, on=['lat', 'lon'])
        AGRI_Aer_CLM_Mask = df_AGRI_Aer_CLM # .loc[(df_AGRI_Aer_CLM['CLM'] > 1)]  # !!!!!!
        # AGRI_Aer_CLM_Mask.to_excel('AGRI_Simu_CLM.xlsx', index=False)

        lon_AOD, lat_AOD, data_AOD = self.Get_CAMS_AOD_data(var_name='aod550')

        AOD_AGRI_Point = grid_interp_to_station(grid_var=data_AOD, grid_lon=lon_AOD, grid_lat=lat_AOD,
                                                station_lon=AGRI_Aer_CLM_Mask['lon'].values,
                                                station_lat=AGRI_Aer_CLM_Mask['lat'].values)
        AGRI_Aer_CLM_Mask.loc[:, 'AOD'] = AOD_AGRI_Point.values

        AGRI_Aer_CLM_Mask_AOD = AGRI_Aer_CLM_Mask#.loc[(AGRI_Aer_CLM_Mask['AOD'] > 0.8)]  # 0.6
        return AGRI_Aer_CLM_Mask_AOD  # AGRI_Aer_CLM_Mask_AOD

    # def plt_data_on_map(self, data_, lat_, lon_, title, colorbar_sel, Vmin, Vmax):
    #     lon_AOD, lat_AOD, data_AOD = self.Get_CAMS_AOD_data(var_name='aod550')
    #     map_ = Basemap(projection='cyl', llcrnrlon=self.WestLon, llcrnrlat=self.SouthLat,
    #                    urcrnrlon=self.EastLon, urcrnrlat=self.NorthLat, resolution='l')
    #     map_.readshapefile('/Users/sunhaofei/Code/plot_use/china1', 'whatevername', color='k', linewidth=0.8)
    #     map_.drawmapboundary(fill_color='white')
    #     map_.drawparallels(np.arange(0, 44, 5.), color='black', dashes=[5, 5], fontsize=11,
    #                        linewidth=0.4, labels=[True, False, False, False])
    #     map_.drawmeridians(np.arange(103, 180, 5.), color='black', dashes=[5, 5], fontsize=11,
    #                        linewidth=0.4, labels=[False, False, False, True])
    #     plt.title(title, loc='left', fontsize=11)  # , pad=20
    #
    #     if colorbar_sel == "jet":
    #         plot_ = map_.scatter(lon_, lat_, c=data_, s=2, vmin=Vmin, vmax=Vmax, cmap="jet")  #
    #         C = plt.contour(lon_AOD, lat_AOD, data_AOD, 6, colors='black')  # 8表示等高线的密集程度，最大是10
    #         plt.clabel(C, inline=1, fontsize=10, fmt='%1.1f')
    #         bar = plt.colorbar(plot_, orientation='vertical', fraction=0.034, pad=0.04)  # , format='%1.3f'
    #         bar.ax.tick_params(labelsize=11)
    #     elif colorbar_sel == "seismic":
    #         plot_ = map_.scatter(lon_, lat_, c=data_, s=2, vmin=Vmin, vmax=Vmax, cmap="seismic")  #
    #         C = plt.contour(lon_AOD, lat_AOD, data_AOD, 6, colors='black')
    #         plt.clabel(C, inline=1, fontsize=10, fmt='%1.1f')
    #         bar = plt.colorbar(plot_, orientation='vertical', fraction=0.034, pad=0.04)  # , format='%1.3f'
    #         bar.ax.tick_params(labelsize=11)
    #     else:
    #         color_level = [0, 1, 2, 3, 4, ]
    #         color_dict = ['silver', 'Gray', '#33A2FF', 'Navy', ]  # 颜色列表
    #         cmap_s = mcolors.ListedColormap(color_dict)  # 产生颜色映射
    #         norm = mcolors.BoundaryNorm(color_level, cmap_s.N)  # 生成索引
    #         map_.scatter(lon_, lat_, latlon=True, s=2, c=data_, cmap=cmap_s)
    #         bar = plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap_s),
    #                            orientation='vertical', fraction=0.043, pad=0.04)
    #         bar.set_ticks([0.5, 1.5, 2.5, 3.5, ])
    #         bar.set_ticklabels(['cloud', 'probably cloud', 'probably clear', 'clear', ])
    #         bar.ax.tick_params(length=0)
    #     return None

    def plot_simu_AGRI_bias(self, AGRI_data, CTL_data, aer_data, LON, LAT, Channel, Aer_name):
        fig = plt.figure(figsize=(12, 7.8), dpi=300)
        plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['font.size'] = 11
        #     plt.subplots_adjust(wspace=-0.2,hspace=-0.35)#调整子图间距
        # print(Channel)
        if Channel == 1:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [0, 0.5, -0.3, 0.3, -0.05, 0.05]
        elif Channel == 2:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [0, 0.5, -0.3, 0.3, -0.1, 0.1]
        elif Channel == 3:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [0, 0.5, -0.3, 0.3, -0.1, 0.1]
        elif Channel == 4:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [0, 0.05, -0.3, 0.3, -0.01, 0.01]
        elif Channel == 5:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [0, 0.5, -0.3, 0.3, -0.05, 0.05]
        elif Channel == 6:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [0, 0.3, -0.3, 0.3, -0.05, 0.05]
        elif Channel == 8:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [260, 310, -10, 10, -0.3, 0.3]
        elif Channel == 9:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [220, 260, -10, 10, -0.01, 0.01]
        elif Channel == 10 or Channel == 14:
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [230, 270, -10, 10, -0.1, 0.1]
        else:  # Channel == 11 or 12 or 13
            vmin_1, vmax_1, vmin_2, vmax_2, vmin_3, vmax_3 = [250, 290, -10, 10, -0.1, 0.1]
        # print(vmin_1, vmax_1, vmin_2, vmax_2)
        # fig.add_subplot(231)
        # self.plt_data_on_map(data_=AGRI_data, lat_=LAT, lon_=LON,
        #                      colorbar_sel='jet', Vmin=vmin_1, Vmax=vmax_1,
        #                      title="(a) AGRI observation of channel " + str(Channel))
        # fig.add_subplot(232)
        # self.plt_data_on_map(data_=CTL_data, lat_=LAT, lon_=LON,
        #                      colorbar_sel='jet', Vmin=vmin_1, Vmax=vmax_1,
        #                      title="(b) CTL of channel " + str(Channel))
        # fig.add_subplot(233)
        # self.plt_data_on_map(data_=aer_data, lat_=LAT, lon_=LON,
        #                      colorbar_sel='jet', Vmin=vmin_1, Vmax=vmax_1,
        #                      title="(c) " + Aer_name + " of channel" + str(Channel))
        # fig.add_subplot(234)
        # self.plt_data_on_map(data_=CTL_data - AGRI_data, lat_=LAT, lon_=LON,
        #                      colorbar_sel='seismic', Vmin=vmin_2, Vmax=vmax_2,
        #                      title="(d) CTL-AGRI of channel " + str(Channel))
        # fig.add_subplot(235)
        # self.plt_data_on_map(data_=aer_data - AGRI_data, lat_=LAT, lon_=LON,
        #                      colorbar_sel='seismic', Vmin=vmin_2, Vmax=vmax_2,
        #                      title="(e) Aer-AGRI of channel " + str(Channel))
        # fig.add_subplot(236)
        # self.plt_data_on_map(data_=aer_data - CTL_data, lat_=LAT, lon_=LON,
        #                      colorbar_sel='seismic', Vmin=vmin_3, Vmax=vmax_3,
        #                      title="(f) Aer-CTL of channel " + str(Channel))
        #
        # fig.tight_layout()  # 调整整体空白
        # plt.savefig(Aer_name + '_CH' + str(Channel) + '.png')
        # # plt.show()
        return None


class CAMSData(object):
    """docstring for CAMSData"""
    def __init__(self):
        super(CAMSData, self).__init__()

    def AerosolMixingRatio_Time_i(self, index, filename):
        FileRegionTime = filename.sel(time=index)
        dus = FileRegionTime['aermr04']
        dum = FileRegionTime['aermr05']
        dul = FileRegionTime['aermr06']

        bchphil = FileRegionTime['aermr09']
        bchphob = FileRegionTime['aermr10']

        omhphil = FileRegionTime['aermr07']
        omhphob = FileRegionTime['aermr08']

        ssl = FileRegionTime['aermr01']
        ssm = FileRegionTime['aermr02']
        sss = FileRegionTime['aermr03']

        su = FileRegionTime['aermr11']

        BC = bchphil + bchphob
        OM = omhphil + omhphob
        SS = sss + ssm + ssl
        DU = dus + dum + dul
        SU = su
        # print(sulphate.shape)
        return BC, OM, SS, DU, SU

    def VerticalIntegration_time_i(self, index, filename):
        file_region_Time = filename.sel(time=index)
        aermssdus = file_region_Time['aermssdus']
        aermssdum = file_region_Time['aermssdum']
        aermssdul = file_region_Time['aermssdul']

        aermssbchphil = file_region_Time['aermssbchphil']
        aermssbchphob = file_region_Time['aermssbchphob']

        aermssomhphil = file_region_Time['aermssomhphil']
        aermssomhphob = file_region_Time['aermssomhphob']

        aermssssl = file_region_Time['aermssssl']
        aermssssm = file_region_Time['aermssssm']
        aermsssss = file_region_Time['aermsssss']

        aermsssu = file_region_Time['aermsssu']

        BC = aermssbchphil + aermssbchphob
        OM = aermssomhphil + aermssomhphob
        SS = aermsssss + aermssssm + aermssssl
        DU = aermssdus + aermssdum + aermssdul
        SU = aermsssu
        # print(sulphate.shape)
        return BC, OM, SS, DU, SU

    def Contour_map(self, leftlon, rightlon, lowerlat, upperlat, ax, Img_Extent, spec, data, ColorMap, SumOperate):
        ax.set_extent(Img_Extent, crs=ccrs.PlateCarree())
        ax.set_xticks(np.arange(leftlon, rightlon + spec - 4, spec), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(lowerlat, upperlat + spec - 4, spec), crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(cticker.LongitudeFormatter())
        ax.yaxis.set_major_formatter(cticker.LatitudeFormatter())
        ax.grid(linewidth=0.6, color='black', alpha=0.5, linestyle='--')
        # 读取shp文件
        china = shpreader.Reader('/Users/sunhaofei/Code/plot_use/bou2_4l.dbf').geometries()
        # 绘制中国国界省界九段线等等
        ax.add_geometries(china, ccrs.PlateCarree(), facecolor='none', edgecolor='black', zorder=1, lw=0.75)

        Lon = data.longitude.values
        Lat = data.latitude.values
        lon_, lat_ = np.meshgrid(Lon, Lat)
        if SumOperate == "True":
            data_ = np.array(data).sum(0)
            print(data_.shape)
            plot_ = ax.contourf(lon_, lat_, data_, levels=10, extend='both', zorder=0,
                                transform=ccrs.PlateCarree(),
                                cmap=ColorMap)  # levels=np.arange(-0.9, 1.0, 0.1),
            bar = plt.colorbar(plot_, orientation='vertical', fraction=0.034, pad=0.04)
            bar.ax.tick_params(labelsize=11)
            bar.formatter.set_powerlimits((-1, 1))
            bar.ax.yaxis.set_offset_position('left')
            bar.update_ticks()
        else:
            plot_ = ax.contourf(lon_, lat_, data, levels=10, extend='both', zorder=0,
                                transform=ccrs.PlateCarree(),
                                cmap=ColorMap)  # levels=np.arange(-0.9, 1.0, 0.1),
            bar = plt.colorbar(plot_, orientation='vertical', fraction=0.034, pad=0.04)
            bar.ax.tick_params(labelsize=11)
            bar.formatter.set_powerlimits((-1, 1))
            bar.ax.yaxis.set_offset_position('left')
            bar.update_ticks()

    def LongitudeElevation(self, ax, lon_, level_, data_, ColorMap):
        plot_ = plt.contourf(lon_, level_, data_, levels=10, cmap=ColorMap, extend='max')
        plt.ylim(700, 1000)
        ax.invert_yaxis()
        ax.set_xticks(np.arange(102, 126, 5))
        ax.set_xticklabels([r'$120^{^\circ}E$', r'$107^{^\circ}E$',
                            r'$112^{^\circ}E$', r'$117^{^\circ}E$', r'$122^{^\circ}E$'])
        bar = plt.colorbar(plot_, orientation='vertical', fraction=0.07, pad=0.04)
        bar.ax.tick_params(labelsize=12)
        bar.formatter.set_powerlimits((-1, 1))
        bar.ax.yaxis.set_offset_position('left')
        bar.update_ticks()
        plt.ylabel("Pressure (hPa)")
        return None