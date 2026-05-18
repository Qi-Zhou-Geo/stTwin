## 📢 Welcome to **stTwin**
A global seismic record of channelized flow events.
---

This is a <br>
A digital twins for catchments sediments transport

### 🛠️ 0. Major Changes for v0.1
**1,** Initial the project.


### 💪 2. Contributors <br>
**[Qi Zhou](https://github.com/Qi-Zhou-Geo)** <br>
qi.zhou@gfz.de or qi.zhou.geo@gmail.com <br>


### Questions, 2025-09
(1) why one more column in [Error?](config/SedCas-main/climate.met) <br>
(2) what's the difference of func. ET_PM_PT and ET_PT in [Error?](config/SedCas-main/modules.py) <br>
(3) what's the EPT in ET_PT [Error?](config/SedCas-main/modules.py) <br>
(4) why close cond2? in  func [large_ls](config/SedCas-main/modules.py) <br>  
#cond2 = T_day_1 > 0                    # positive T days
(5) error? [large_ls](config/SedCas-main/modules.py) <br>  
lrg_ls = lrg_ls / area * 10.**-3

### Questions, 2025-10
(5) inactive storage in sedcas func. [large_ls](config/SedCas-main/modules.py) <br>
(6) why after 2018, there is no DF? need to recarlibrate it <br>
(7) what's this? one slice of the running case? [save_output](config/SedCas-main/SedCas.py)
"sedout['Qstl'] = self.sed.sopot[:, 0]" <br>
(8) what's the potential sediment output based on discharge or debris flows, sediment output above minimum threshold and concentration of debris flows
