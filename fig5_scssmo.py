#%%
import xarray as xr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cmaps
import matplotlib as mpl
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
#colorbar配色
# Create a colormap with truncated range
cmap_design = mpl.colormaps.get_cmap(cmaps.BlueWhiteOrangeRed)#得到一个颜色映射
warm = mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_design.name, a=0, b=1), cmap_design(np.linspace(0.53, 0.9, 256))[0:250], N=255)
cold = mcolors.LinearSegmentedColormap.from_list('trunc({n},{a:.2f},{b:.2f})'.format(n=cmap_design.name, a=0, b=1), cmap_design(np.linspace(0.1, 0.47, 256))[0:250], N=255)

all_colors = np.vstack((cold(np.linspace(0, 1, 256)),warm(np.linspace(0, 1, 256))))
cold_warm = mcolors.LinearSegmentedColormap.from_list('new_cmap', all_colors)
#%%u850
u850 = xr.open_dataset('../DATA/DATA/ERA5_native_SCSSM/U_daliy/ERA5_ecmwf_U_Y1940-2025_M04-06_daily.nc').u
#scssm区域
scss = u850.sel(latitude=slice(15, 5), longitude=slice(110, 120))
scssm = scss.mean(dim={'latitude','longitude'})

scssm = scssm.sel(valid_time=slice('1979-01-01', '2025-06-30'))

# %%
def get_pentad_indices(times):
    """
    返回每个时间点属于哪一个pentad,每年从1开始计算
    闰年的2月29日不参与计算
    """
    times = pd.to_datetime(times)
    # 去掉2月29日
    mask = ~((times.month == 2) & (times.day == 29))
    times = times[mask]
    doy = times.dayofyear

    # 需要重新计算dayofyear（去掉2月29日后，后面日期的doy都要-1）
    doy_new = []
    for t in times:
        if t.month > 2 or (t.month == 2 and t.day > 28):
            # 闰年且当前日期在3月1日及以后，dayofyear要-1
            if (t.year % 4 == 0 and (t.year % 100 != 0 or t.year % 400 == 0)):
                doy_new.append(t.dayofyear - 1)
            else:
                doy_new.append(t.dayofyear)
        else:
            doy_new.append(t.dayofyear)
    doy = np.array(doy_new)

    pentad = ((doy - 1) // 5) + 1
    year = times.year
    pentad_idx = pd.MultiIndex.from_arrays([year, pentad], names=('year', 'pentad'))
    return pentad_idx, times

#索引日期对应的侯
times = pd.to_datetime(scssm['valid_time'].values)
pentad_idx = get_pentad_indices(times)
#取4月25日对应的索引
index_s = pentad_idx[0][(times.month==5) & (times.day==1)][0]
#%%
def detect_scssm_onset(scssm):
    """
    输入:
    scssm -- xarray.DataArray, 时间维度为 'valid_time', 内容为850hPa zonal wind
    输出:
    onset_dates -- {year: onset_pentad}, onset_pentad为严格/宽松条件下首次满足的pentad
    scssm_pentad -- shape(year, pentad),每pentad平均风速
    严格条件:
    - 首个pentad(p)及其后连续3个pentad(共4个,包括p),至少有3个为正
    - 这4个pentad的均值 > 1 m/s
    宽松条件:
    - 首个pentad(p)及其后连续3个pentad(共4个),至少有2个为正
    - 这4个pentad的均值 >= 0.5 m/s
    """
    times = pd.to_datetime(scssm['valid_time'].values)
    pentad_idx = get_pentad_indices(times)
    # 按pentad取均值
    scssm_pentad = pd.Series(scssm.values, index=pentad_idx[0]).groupby(['year','pentad']).mean()
    scssm_pentad = scssm_pentad.unstack(fill_value=np.nan)  # shape: (year, pentad)
    scssm_pentad = xr.DataArray(
        data = scssm_pentad.values,
        coords = {'year': scssm_pentad.index, 'pentad': scssm_pentad.columns},
        dims = ['year', 'pentad']
    )
    onset_dates = {}
    for year in scssm_pentad.year.values:
        pentads_to_check = [p for p in scssm_pentad.pentad.values if p >= 24]
        found = False
        # Step 1: 严格条件
        for p in pentads_to_check:
            p_next = [pp for pp in range(p, p+4) if pp in scssm_pentad.pentad.values]
            if len(p_next) < 4:
                continue
            wnd4 = scssm_pentad.sel(year=year, pentad=p_next).values
            if wnd4[0] <= 0:
                continue
            if np.sum(wnd4 > 0) < 2:
                continue
            if np.mean(wnd4) <= 1:
                continue
            onset_dates[year] = p
            found = True
            break
        # Step 2: 宽松条件
        if not found:
            for p in pentads_to_check:
                wnd0 = scssm_pentad.sel(year=year, pentad=p).item()
                if wnd0 <= 0:
                    continue
                wnd4 = scssm_pentad.sel(year=year, pentad=slice(p+1,p+4)).values
                if np.sum(wnd4 > 0) < 2:
                    continue
                if np.mean(wnd4) <= 0.6:
                    continue
                onset_dates[year] = p
                found = True
                break
        if not found:
            onset_dates[year] = None
    return onset_dates, scssm_pentad
# %%
scssm_onset,scssm_pentad = detect_scssm_onset(scssm)

#%%
#取字典的值
scssm_onset_values = np.array(list(scssm_onset.values()))
#赋予年份
scssm_onset_values = pd.Series(scssm_onset_values, index=scssm_onset.keys(), name='scssm_onset')
#%%重整形(year, time)
#截取每年4月25日后的数据
scssm_after_425 = scssm.sel(valid_time=times[(times.month==4) & (times.day>=26) | (times.month==5) | (times.month==6)])
scssm_after_425 = scssm_after_425.values.reshape(47, -1)  # (year, time)

#设置其为dataframe，从4月26日开始到6月30日
scssm_each_year = pd.DataFrame(scssm_after_425, index=np.arange(1979, 2026))

#%%
scssm_2005_pentad = scssm_pentad.sel(year=2005)
scssm_2005 = scssm.sel(valid_time=times[times.year==2005])
plt.plot(scssm_2005['valid_time'], scssm_2005.values, marker='o')
plt.axhline(0, color='k', linestyle='--')
#每隔5天画一条竖线
for day in pd.date_range(start='2005-04-26', end='2005-06-30', freq='5D'):
    plt.axvline(day, color='gray', linestyle='--', alpha=0.5)
    #对应侯标记
    plt.text(day, -5, f'{get_pentad_indices([day])[0][0][1]}', ha='center', va='top', fontsize=10)

#%%
#中文显示设置
# ===== 1. 取得日期序列（DatetimeIndex）与对应的日值 =====
# 这里兼容 xarray 或 pandas 的情况
try:
    # 如果 scssm_2005 是 xarray DataArray 且含 valid_time 维
    dates = pd.to_datetime(scssm_2005['valid_time'].values)
    values = np.asarray(scssm_2005.values).ravel()
except Exception:
    # 如果是 pandas Series with DatetimeIndex
    try:
        dates = pd.to_datetime(scssm_2005.index)
        values = np.asarray(scssm_2005.values).ravel()
    except Exception:
        raise RuntimeError("无法识别 scssm_2005 数据结构，请传入带日期索引的 Series 或含 valid_time 的 xarray DataArray")

# ===== 2. 计算每个日期对应的 pentad 编号（调用你的 get_pentad_indices） =====
# 如果你已有按日映射的 Series，可直接用它；否则逐日调用 get_pentad_indices
pentad_for_date = []
for d in dates:
    # get_pentad_indices 可能返回复杂结构，按你之前使用 get_pentad_indices([day])[0][0][1]
    res = get_pentad_indices([d])
    # 上面函数返回结构示例: [[(start, end, pentad_num), ...]] ；按你原先用法取[0][0][1]
    pent = res[0][0][1]
    pentad_for_date.append(int(pent))
day2pent = pd.Series(pentad_for_date, index=dates)

# ===== 3. 按侯分组，得到每个侯的起始/结束日期与中点 =====
unique_pentads = sorted(day2pent.unique())
pentad_ranges = []  # list of (pentad, start_date, end_date, mid_date)
for p in unique_pentads:
    mask = day2pent == p
    if not mask.any():
        continue
    start = mask.index[mask][0]
    end = mask.index[mask][-1]
    mid = start + (end - start) / 2
    pentad_ranges.append((p, pd.to_datetime(start), pd.to_datetime(end), mid))

# ===== 4. 绘图：主轴画日序列，top axis 显示侯编号（ticks 在 mid） =====
fig, ax = plt.subplots(figsize=(12, 4.5), dpi=300, facecolor='w')

# 主图：日序列条/线（你原来用 plot + vertical 5-day lines）
ax.plot(dates, values, marker='o', linewidth=1.6, color='#241e1e', label='SCSSM daily')
ax.axhline(0, color='k', linestyle='--', linewidth=0.8, alpha=0.7)

# （可选）画出每 5 天边界的垂直线（按原来需求）
for day in pd.date_range(start=dates.min(), end=dates.max(), freq='5D'):
    ax.axvline(day, color='gray', linestyle='--', alpha=0.25, linewidth=0.6)

# 如果你想保持原来在竖线下方写侯编号的做法，可以移除（现在用 top axis 显示）
# ax.text(day, y_text, ...)

# ===== 5. top x-axis：以侯区间中点为 ticks，并标注侯编号 =====
ax_top = ax.twiny()
# 使 top 轴与主轴 x 轴范围一致
ax_top.set_xlim(ax.get_xlim())

# 取得 mid 日期作为 ticks（matplotlib 支持 datetime objects）
mid_dates = [t[3] for t in pentad_ranges]
pentad_labels = [str(t[0]) for t in pentad_ranges]

ax_top.set_xticks(mid_dates)
ax_top.set_xticklabels(pentad_labels, fontsize=10, rotation=0)
ax_top.xaxis.set_ticks_position('top')
ax_top.xaxis.set_label_position('top')
ax_top.set_xlabel('Pentad', fontsize=11)

# （可选）在 top 轴上画小的分隔线或延伸显示每个侯的范围：
# 例如画半透明的矩形背景表示侯区间
for (_, s, e, mid) in pentad_ranges:
    ax_top.axvspan(s, e + pd.Timedelta(days=1), ymin=0.0, ymax=0.12, color='gray', alpha=0.06, zorder=0, clip_on=False)

# ===== 6. 美化主轴刻度为年份格式（可按需要改） =====
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1,4,7,10]))  # 例如每季或按需
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
fig.autofmt_xdate(rotation=45)

ax.set_ylabel('SCSSM value')
ax.set_title('SCSSM 2005 with Pentad labels (top axis)')

# 若需要在 top axis 上只显示 pentad 起始日期作为次要说明，可在 top axis 下方添加注释
# 例如：
for p, s, e, mid in pentad_ranges:
    ax_top.text(mid, 0.2, f'{s.strftime("%m-%d")}', transform=ax_top.get_xaxis_transform(), ha='center', va='bottom', fontsize=8)

plt.tight_layout()
plt.show()

#%%
# ncep_doe = pd.read_excel('../AOSL_toLYN/AOSLv2025-07/reconstruct/STT_ENSO_SAM_OD(2017CD).xlsx',header=None)
# ncep_doe_y = ncep_doe[4].values
# #年份
# ncep_doe_years = ncep_doe[0].values
# #赋予年份
# ncep_doe_y = pd.Series(ncep_doe_y, index=ncep_doe_years)
# #添加2025为30
# ncep_doe_y.loc[2025] = 30
#计算相关系数
# corr = scssm_onset_values.loc['1979':'2024'].corr(ncep_doe_y)

#%%
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Calibri']  # 设置字体为Calibri
plt.rcParams['axes.linewidth'] = 2
#刻度设置粗细
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

fig, ax1 = plt.subplots(figsize=(8, 4), dpi=600)

# your data
years = np.arange(1979, 2026)                      # length 86
days_idx = np.arange(scssm_each_year.shape[1])     # 0..66 (67 days)

# contourf 背景
c1 = ax1.contourf(years, days_idx, scssm_each_year.T,levels=np.arange(-12, 13, 1),cmap=cold_warm, extend='both')
ax1.contour(years, days_idx, scssm_each_year.T, levels=[0], colors="#B9B8B8", linewidths=1, linestyles='--')


# colorbar
cax = fig.add_axes([0.95, 0.18, 0.015, 0.6])
cb = fig.colorbar(c1, cax=cax, aspect=20, shrink=0.8, extendrect='both', drawedges=True, extendfrac='auto')
cb.ax.tick_params(labelsize=10)


# ---- 左轴：只在部分索引打日期标签（物理位置基于 days_idx） ----
left_ticks = np.arange(0,66,5)# 这些是 contourf 的行坐标（物理位置）
dates = pd.date_range(start='2025-04-26', end='2025-06-30', freq='5D')
day_labels = dates.strftime('%b %d').tolist()  # 你希望在这些位置显示的值
ax1.set_yticks(left_ticks)
ax1.set_yticklabels(day_labels, fontsize=14)

ax1.set_ylim(0, days_idx[-1])   # 确保主轴 y 范围为 0..66
ax1.tick_params(axis='y', labelsize=14)

# ---- 右轴：在相同物理位置显示 pentad 标签 ----
ax2 = ax1.twinx()
right_ticks = np.arange(2.5, 63.5, 5)
right_labels = np.arange(24, 37)  # 你希望在这些位置显示的值

# **关键**：把右轴刻度位置设为与左轴相同的物理位置
ax2.set_ylim(ax1.get_ylim())        # 使两边物理范围一致
ax2.set_yticks(right_ticks)          # 刻度位置（物理位置）用 right_ticks
ax2.set_yticklabels(right_labels, fontsize=14)  # 刻度标签显示 pentad 值
ax2.tick_params(axis='y', labelsize=14)

# ---- 绘曲线：先把 pentad 值映射到左轴索引（物理坐标），再画 ----
era5_mapped = np.interp(scssm_onset_values, right_labels, right_ticks)
# doe5_mapped = np.interp(ncep_doe_y.loc[1979:2025], right_labels, right_ticks)

ax2.plot(years, era5_mapped, label='ERA5', marker='o', c="#424040", lw=3, markersize=8,alpha=0.8)
# ax2.plot(years, doe5_mapped, label='DOE', marker='s', c="#4770ED", lw=3, markersize=8,alpha=0.8)

# x 轴、标题、图例、注释
ax1.set_xticks(np.arange(1979, 2016, 3))
ax1.set_xticklabels(np.arange(1979, 2016, 3), fontsize=14)
ax1.set_xlim(1979, 2015)
ax1.set_title('SCSSMO (For training)', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.25,lw = 1,color = "#8D8787")

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1.02))

#svg
# plt.savefig('./features_separate/u850.svg', bbox_inches='tight', dpi=600)
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

plt.rcParams['font.sans-serif'] = ['Calibri']  # 设置字体为Calibri
plt.rcParams['axes.linewidth'] = 2
#刻度设置粗细
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

fig, ax1 = plt.subplots(figsize=(12, 4), dpi=600)

# your data
years = np.arange(1979, 2026)                      # length 86
days_idx = np.arange(scssm_each_year.shape[1])     # 0..66 (67 days)

# contourf 背景
c1 = ax1.contourf(years, days_idx, scssm_each_year.T,levels=np.arange(-12, 13, 1),cmap=cold_warm, extend='both')
ax1.contour(years, days_idx, scssm_each_year.T, levels=[0], colors="#B9B8B8", linewidths=1, linestyles='--')

mid_color = cold_warm(0.5)  # 取cmap中间颜色
u850_proxy = Line2D([0], [0], marker='o', color='w', markerfacecolor=mid_color, 
                    markeredgecolor='black', markersize=10, label='U850')

# colorbar
cax = fig.add_axes([0.955, 0.12, 0.015, 0.6])
cb = fig.colorbar(c1, cax=cax, aspect=20, shrink=0.8, extendrect='both', drawedges=True, extendfrac='auto')
cb.ax.tick_params(labelsize=14)
# cb.set_label('850hPa U (m/s)', fontsize=14)

# ---- 左轴：只在部分索引打日期标签（物理位置基于 days_idx） ----
left_ticks = np.arange(0,66,5)# 这些是 contourf 的行坐标（物理位置）
dates = pd.date_range(start='2025-04-26', end='2025-06-30', freq='5D')
day_labels = dates.strftime('%b %d').tolist()  # 你希望在这些位置显示的值
ax1.set_yticks(left_ticks)
ax1.set_yticklabels(day_labels, fontsize=14)

ax1.set_ylim(0, days_idx[-1])   # 确保主轴 y 范围为 0..66
ax1.tick_params(axis='y', labelsize=14)

# ---- 右轴：在相同物理位置显示 pentad 标签 ----
ax2 = ax1.twinx()
right_ticks = np.arange(2.5, 63.5, 5)
right_labels = np.arange(24, 37)  # 你希望在这些位置显示的值

# **关键**：把右轴刻度位置设为与左轴相同的物理位置
ax2.set_ylim(ax1.get_ylim())        # 使两边物理范围一致
ax2.set_yticks(right_ticks)          # 刻度位置（物理位置）用 right_ticks
ax2.set_yticklabels(right_labels, fontsize=14)  # 刻度标签显示 pentad 值
ax2.tick_params(axis='y', labelsize=14)

# ---- 绘曲线：先把 pentad 值映射到左轴索引（物理坐标），再画 ----
era5_mapped = np.interp(scssm_onset_values, right_labels, right_ticks)
# doe5_mapped = np.interp(ncep_doe_y.loc[1979:2025], right_labels, right_ticks)

ax2.plot(years, era5_mapped, label='onset', marker='o', c="#424040", lw=3, markersize=8,alpha=0.8)
# ax2.plot(years, doe5_mapped, label='DOE', marker='s', c="#4770ED", lw=3, markersize=8,alpha=0.8)

# x 轴、标题、图例、注释
ax1.set_xticks(np.arange(1979, 2026, 3))
ax1.set_xticklabels(np.arange(1979, 2026, 3), fontsize=14)
ax1.set_xlim(1979, 2025)
# ax1.set_title('SCSSM Onset Pentad', fontsize=18)
ax1.grid(True, linestyle='--', alpha=0.25,lw = 1,color = "#8D8787")

lines1, labels1 = ax1.get_legend_handles_labels()
ax1.add_artist(u850_proxy)
legend_elements = [u850_proxy] + lines2
ax1.legend(handles=legend_elements, loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1.02))

# lines2, labels2 = ax2.get_legend_handles_labels()
# ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1.02))

# plt.savefig('./features_separate/scssm_onset_pentad_ERA5_u850.png', bbox_inches='tight', dpi=600)
plt.show()

#%%
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Ellipse
from matplotlib.legend_handler import HandlerBase
import matplotlib as mpl
import pandas as pd

plt.rcParams['font.sans-serif'] = ['Calibri']
plt.rcParams['axes.linewidth'] = 2
mpl.rcParams['xtick.major.width'] = 2
mpl.rcParams['ytick.major.width'] = 2

fig, ax1 = plt.subplots(figsize=(12, 4), dpi=600)

# your data
years = np.arange(1979, 2026)
days_idx = np.arange(scssm_each_year.shape[1])

# contourf 背景
c1 = ax1.contourf(years, days_idx, scssm_each_year.T, levels=np.arange(-12, 13, 1), cmap=cold_warm, extend='both')
ax1.contour(years, days_idx, scssm_each_year.T, levels=[0], colors="#B9B8B8", linewidths=1, linestyles='--')

# ========================== 三层同心椭圆图例（U850） ==========================
class TripleEllipseLegendHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x = width / 2
        y = height / 2
        
        # 外层：冷色
        e1 = Ellipse((x, y), width, height,
                     facecolor=cold_warm(0.15), edgecolor='black', linewidth=0.5, transform=trans)
        # 中层：白色
        e2 = Ellipse((x, y), width * 0.66, height * 0.66,
                     facecolor=cold_warm(0.5), edgecolor='black', linewidth=0.5, transform=trans,ls ='--')
        # 内层：暖色
        e3 = Ellipse((x, y), width * 0.33, height * 0.33,
                     facecolor=cold_warm(0.85), edgecolor='black', linewidth=0.5, transform=trans,ls ='--')
        
        return [e1, e2, e3]

# 创建一个 Ellipse 类型的 dummy handle 作为 U850 的图例入口
u850_legend = Ellipse((0, 0), 1, 1, facecolor='white', edgecolor='black', label='U850')

# colorbar
cax = fig.add_axes([0.955, 0.12, 0.015, 0.6])
cb = fig.colorbar(c1, cax=cax, aspect=20, shrink=0.8, extendrect='both', drawedges=True, extendfrac='auto')
cb.ax.set_ylabel(r'Zonal wind (m s$^{-1}$)', fontsize=14)
cb.ax.tick_params(labelsize=14)

# ---- 左轴：日期标签 ----
left_ticks = np.arange(0, 66, 5)
dates = pd.date_range(start='2025-04-26', end='2025-06-30', freq='5D')
day_labels = dates.strftime('%b %d').tolist()
ax1.set_xlabel('Year', fontsize=16)
ax1.set_ylabel('Date', fontsize=16)
ax1.set_yticks(left_ticks)
ax1.set_yticklabels(day_labels, fontsize=14)

ax1.set_ylim(0, days_idx[-1])
ax1.tick_params(axis='y', labelsize=14)

# ---- 右轴：pentad 标签 ----
ax2 = ax1.twinx()
right_ticks = np.arange(2.5, 63.5, 5)
right_labels = np.arange(24, 37)

ax2.set_ylim(ax1.get_ylim())
ax2.set_ylabel('Pentad', fontsize=16)
ax2.set_yticks(right_ticks)
ax2.set_yticklabels(right_labels, fontsize=14)
ax2.tick_params(axis='y', labelsize=14)

# ---- 绘曲线：pentad 映射到左轴物理坐标 ----
era5_mapped = np.interp(scssm_onset_values, right_labels, right_ticks)
# doe5_mapped = np.interp(ncep_doe_y.loc[1979:2025], right_labels, right_ticks)

line_onset = ax2.plot(years, era5_mapped, label='onset', marker='o', c="#424040", lw=3, markersize=8, alpha=0.8)[0]

# x 轴、网格
ax1.set_xticks(np.arange(1979, 2026, 3))
ax1.set_xticklabels(np.arange(1979, 2026, 3), fontsize=14)
ax1.set_xlim(1979, 2025)
ax1.grid(True, linestyle='--', alpha=0.25, lw=1, color="#8D8787")

# ---- 图例：U850 三层椭圆 + onset 线 ----
legend_elements = [u850_legend, line_onset]
ax1.legend(handles=legend_elements,
           handler_map={Ellipse: TripleEllipseLegendHandler()},
           loc='upper right', fontsize=12, bbox_to_anchor=(1.15, 1.02))


plt.savefig('./v1_code/scssm_onset_pentad_ERA5_u850.png', bbox_inches='tight', dpi=600)
plt.show()

#%%
sel_scssm_onset_values = scssm_onset_values.loc['1986':'2025']
sel_scssm_onset_values_mean = sel_scssm_onset_values.mean()

#%%整体侯变化趋势,一元线性拟合

from scipy.stats import linregress
slope, intercept, r_value, p_value, std_err = linregress(scssm_onset_values.index, scssm_onset_values.values)


import seaborn as sns
plt.figure(figsize=(10, 6), dpi=600)
sns.regplot(x=scssm_onset_values.index, y=scssm_onset_values.values, scatter=True, order=1, ci=None, scatter_kws={'s':50}, line_kws={'color':'red','lw':2})

plt.xticks(np.arange(1979, 2026, 3), fontsize=14)
plt.yticks(np.arange(20, 40, 2), fontsize=14)
plt.xlabel('Year', fontsize=16)
plt.ylabel('SCSSM Onset Pentad', fontsize=16)
plt.title('Trend of SCSSM Onset Pentad (ERA5 u850)', fontsize=18)
plt.show()


# %%
def pentad_to_doy_range(pentad_float):
    """
    将“第几侯(可为小数)”换算为一年中的日序(doy)范围（按你的算法：pentad=((doy-1)//5)+1）
    返回: (doy_start, doy_end, doy_center)
    """
    p = float(pentad_float)
    doy_start = 5*(p - 1) + 1
    doy_end   = 5*p
    doy_center = (doy_start + doy_end) / 2
    return doy_start, doy_end, doy_center

def doy_to_date(year, doy, drop_leap_day=True):
    """
    将doy换算为日期。
    若drop_leap_day=True，则按你代码的规则：闰年也当作365天，跳过2月29日。
    doy 可为小数，返回向下取整后的日期（也可自行round/ceil）。
    """
    doy_int = int(np.floor(doy))

    if not drop_leap_day:
        return (pd.Timestamp(year=year, month=1, day=1) + pd.Timedelta(days=doy_int-1)).date()

    # 按“去掉2月29日”的365天日历
    base = pd.Timestamp(year=year, month=1, day=1)
    date = base + pd.Timedelta(days=doy_int-1)

    # 如果闰年且结果日期落在2/29及之后，需要再加1天跳过2/29
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    if is_leap and date >= pd.Timestamp(year=year, month=3, day=1):
        date += pd.Timedelta(days=1)

    return date.date()

# 示例：28.55侯 -> doy范围与中心
p = 28.55
doy_s, doy_e, doy_c = pentad_to_doy_range(p)
print("pentad:", p)
print("doy range:", doy_s, "-", doy_e, "center:", doy_c)

# 示例：把中心doy换算到某一年日期（按“去掉2月29日”规则）
year = 2020  # 闰年/平年都可以测试
print("center date:", doy_to_date(year, doy_c, drop_leap_day=True))
# %%
