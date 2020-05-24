close all, clear all, clc, dbstop if error
fpni='C:\Users\zsq55\Desktop\MATLAB\shp_map_file\shp_map_file\China\bou2_4l.shp';%中国地图shp文件所在路径
China=shaperead(fpni);
boux=[China(:).X];bouy=[China(:).Y];%分别是获取经度X信息和纬度Y信息
set(gcf,'position',[0 0 1440 780]);%设置图形窗口位置和大小
m_proj('Mercator','lon',[72 137],'lat',[15 55]);%设置投影方式为：墨卡托，地图显示范围
m_plot(boux,bouy,'k');%最关键的一句，绘制地图
%下面这句设置图形横纵坐标为经纬度格式
m_grid('linestyle','none','linewidth',2,'tickdir','out','xaxisloc','bottom','yaxisloc','left','fontsize',15);
hold on;
load filename2_1979_2018.mat; %加载散点数据
load a_trend_TX_Pr_WS_1979_1988.mat
load a_trend_TX_Pr_WS_1989_1998.mat
load a_trend_TX_Pr_WS_1999_2008.mat
load a_trend_TX_Pr_WS_2009_2018.mat
% load a_trend_TX_Pr_WS_1961_2015.mat
lon=(file(:,3)*0.01);lat=(file(:,2)*0.01);dataco=trend_ktal_prsum2(:,3)-trend_ktal_prsum1(:,3)
m_scatter(lon,lat,80, dataco,'filled', 'MarkerFaceColor', 'flat', 'MarkerEdgeColor', 'w','linewi',1) ;%画实心点图
set(gca,'FontSize', 30)
% m_scatter(lon,lat,50, dataco, 'MarkerFaceColor', w','linewi',2) ;%画空心点图
mm=[7/255,  30/255,  70/255
   7/255,  47/255, 107/255
   8/255,  82/255, 156/255
  33/255, 113/255, 181/255
  66/255, 146/255, 199/255
  90/255, 160/255, 205/255
 120/255, 191/255, 214/255
 170/255, 220/255, 230/255
 219/255, 245/255, 255/255
 240/255, 252/255, 255/255
 255/255, 240/255, 245/255
 255/255, 224/255, 224/255
 252/255, 187/255, 170/255
 252/255, 146/255, 114/255
 251/255, 106/255,  74/255
 240/255,  60/255,  43/255
 204/255,  24/255,  30/255
 166/255,  15/255,  20/255
 120/255,  10/255,  15/255
  95/255,   0/255,   0/255]
colormap(mm)
caxis([-100 100])
c = colorbar
set(c,'ytick',[-100:50:100],'fontsize',25,'fontname','times new roman')
% colormap( mm,[-1:0.25:1], 'colorbar', 'on','location', 'vertical', 'fontsize', 13);%显示图例
 title('1979-1998', 'Rotation', 0, 'FontSize', 30);%显示图标题
saveas(gcf,'trend_ktal_prsum_w1.png'); 
