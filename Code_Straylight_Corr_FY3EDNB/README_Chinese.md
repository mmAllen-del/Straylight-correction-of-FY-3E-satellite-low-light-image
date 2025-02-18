### 软件使用说明：FY-3E MERSI L1B HDF图像处理工具

#### 1. 环境和依赖库要求

本软件依赖于以下环境和库，确保在运行前安装并配置好相应的环境。

**环境要求：**
- Python 3.9 或更高版本

**依赖库：**
- numpy
- pylab
- matplotlib
- h5py
- os
- pandas
- time
- re
- math
- datetime
- cv2
- shutil
- scipy
- copy
- PIL
- pywt
- cartopy

可以使用以下命令安装必要的库（推荐使用 `pip`）：
```bash
pip install numpy pylab matplotlib h5py pandas opencv-python scipy pillow pywt cartopy
```

#### 2. 需要的输入文件

- **FY-3E L1B HDF文件：** 包含卫星传感器数据的HDF文件（例如：`FY3E_L1B_20230511_1300.hdf`）。
- **FY-3E GEO HDF文件：** 包含地理定位数据的HDF文件（例如：`FY3E_GEO_20230511_1300.hdf`）。

#### 3. 主程序：`hdf5_fy3e_main_module.py`

该程序用于处理特定时间点的FY-3E L1B HDF数据，生成相应的图像和处理结果。

**运行流程：**
1. 打开主程序 `hdf5_fy3e_main_module.py`，并根据需求修改相关参数。
2. 主要参数设置在代码中的第87到93行：
   - `date`：需要处理的时刻（格式为：`YYYYMMDD_HHMM`，如 `"20230511_1300"`）。
   - `rad_file`：对应时刻的FY-3E L1B HDF文件的绝对路径。
   - `geo_file`：对应时刻的FY-3E GEO HDF文件的绝对路径。
   - `outdir`：结果保存的文件夹路径，程序会自动在该文件夹下创建时刻子文件夹。
   - `draw_flag`：设置图像绘制模式：
     - `0`：不绘制图像。
     - `1`：绘制地理图（带地图）。
     - `2`：绘制长方形流程图。
   - `dpi0`：地理图的图片像素值（仅适用于 `draw_flag=1`）。

**生成的文件：**

以 `20230511_1300` 为例，假设 `draw_flag=1` 时，生成的文件包括：
- `20230511_1300_geo.png`：带地图的原始FY-3E MERSI LLB原始辐射值图像。
- `20230511_1300_no_stray_light.png`：去除杂散光后的图像。
- `20230511_1300_enhanced.png`：经增强和去条纹处理后的图像。
- `FY3E_MERSI_LL_Clear_Image_20230511_1300.hdf5`：处理后的HDF5文件，包含以下数据：
  - 纬度（Latitude）
  - 经度（Longitude）
  - 太阳高度角（SolarZenith，SZA）
  - 去除杂散光后的灰度值或辐射值矩阵（No_stray_light_image）
  - 增强和去条纹后的图像（Enhanced_image）

**补充说明：**
1. `No_stray_light_image` 包含图像类型标注（`no stray light images`, `partial stray light images`, `common stray light images`），并根据图像类型分别进行处理。
2. 图像矩阵大小为 `2000x1536`，与原始文件保持一致。
3. 若图像在夜间（SZA > 100°）占比小于 30% 或全黑夜背景下无效0值占比大于 20%，则不会处理并提示：“night proportion less 0.3 !no handle!” 或 “too much 0 in all dark pict !no handle!”。
4. 对于杂散光图像，在最大迭代次数内未找到最佳拟合时，程序将跳过拟合步骤，继续后续处理，并提示：“too long time!!!!”。
5. 文件夹Output和fy3emersill_input放有测试数据

#### 4. 生成文件示例

假设 `20230511_1300` 时刻设置为 `draw_flag=2`（长方形流程图）时，生成的文件包括：
- `20230511_1300.png`：原始FY-3E MERSI LLB辐射值图像。
- `20230511_1300_gauss.png`：高斯拟合去除亮斑后的辐射值图像。
- `20230511_1300_brightness.png`：亮度均衡后的去杂散光图像。
- `20230511_1300_correction.png`：增强和去条纹后的图像。
- `FY3E_MERSI_LL_Clear_Image_20230511_1300.hdf5`：处理后的HDF5文件。

若 `SZA > 100°` 时出现0值，则会生成 `YYYYMMDD_HHMM_filldark.png`，并且针对部分杂散光图像，可能会生成去雾处理后的图像（例如 `YYYYMMDD_HHMM_dehaze.png`）。

#### 5. 注意事项

**读取HDF5文件时：**
- 原数据中，列 `[0, 1, 2, ..., 1535]` 存在无效值（0值），读取时需要去除这些无效值。
- 显示有效图像时，应确保 SZA > 100°，其他范围为空值。

#### 6. 其他辅助程序

- `draw_100aoi_pict.py`：绘制符合 SZA > 100° 条件的污染图。
- `no_stray_judge.py`：判断图像是否含有杂散光污染。
- `gauss_fitting.py`：对污染图像进行高斯拟合去除杂散光。
- `stray_light_handle.py`：处理不同类型杂散光的步骤。
- `algorithm_details.py`：处理杂散光图像的若干算法，包括增强和去条纹、去雾处理、光照校正等。
- `resize_pict.py`：调整图像尺寸为 `2000x1536`。
- `draw_geo_pict.py`：根据 `draw_flag=1` 绘制带地理信息的图像。

#### 7. 总结

本工具集成了卫星图像的处理与分析功能，适用于FY-3E卫星L1B数据的去杂散光、增强、去条纹等处理，生成高质量的图像以及详细的HDF5数据文件，方便后续的分析和研究。