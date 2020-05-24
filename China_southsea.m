subplot('position',[0.,0.0,0.97,0.9])
path(path,'G:\CLIM_CHANGE\CHINA\shp_map_file\m_map');
path(path,'G:\CLIM_CHANGE\CHINA\shp_map_file\China');
lonyou=138;
lonzuo=70;
latshang=55;
latxia=15;
%大地图的上下左右经纬度

a=shaperead('G:\CLIM_CHANGE\CHINA\shp_map_file\China\bou2_4m\bou2_4l.shp');
boul_4lx=[a(:).X];
boul_4ly=[a(:).Y];
m_proj('Equidistant Cylindrical','lon',[lonzuo,lonyou],'lat',[latxia,latshang])

m_plot(boul_4lx,boul_4ly,'k')

set(gca,'ticklength',[0,0])
m_grid('xtick',[70:68:138],'ytick',[15:40:55],'linestyle','none','box','off','xti',[],'yti',[])%,'tic')

hold on
%到这中国地图完成
%%%%%%%%%%%   south sea    %%%%%%%%%%%

ssyou=123;
sszuo=106;
ssshang=24;
ssxia=4;
%south sea上下左右的经纬度
k=0.7;%南海地区的地图和全国地图的比率


j=1;
fx(1)=lonyou-(ssyou-sszuo)*k;
fx(2)=lonyou-(ssyou-sszuo)*k;
fx(3)=lonyou-0.05;
fx(4)=lonyou-0.05;
fx(5)=lonyou-(ssyou-sszuo)*k;
fy(1)=latxia+10;
fy(2)=latxia+(ssshang-ssxia)*k;
fy(3)=latxia+(ssshang-ssxia)*k;
fy(4)=latxia+0.05;
fy(5)=latxia+0.05;
for i=1:4
[lx(i),ly(i)]=m_ll2xy(fx(i),fy(i));
end

bar((lx(1)+lx(3))/2,ly(2),lx(3)-lx(1),'w')
hold on
%以上是将南海要画图的地区涂白，不让背景中的经纬度线在小地图中出现。
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n,l]=size(boul_4lx);


for i=1:l
    if isnan(boul_4lx(i)) && isnan(boul_4ly(i))
        lx2(j)=nan;
        ly2(j)=nan;
        j=j+1;
    end
    if (boul_4lx(i)<=ssyou && boul_4lx(i)>=sszuo) 
        if (boul_4ly(i)<=ssshang && boul_4ly(i)>=ssxia)
            lx2(j)=lonyou-(ssyou-boul_4lx(i))*k;
            ly2(j)=latxia+(boul_4ly(i)-ssxia)*k;
            j=j+1;
        end
    end
end

m_plot(lx2,ly2,'k-','markersize',0.2)
hold on

m_plot(fx,fy,'k-','linewidth',0.2)
fy2(1:5)=latxia;
hold on
m_plot(fx,fy2,'k','linewidth',0.2)
