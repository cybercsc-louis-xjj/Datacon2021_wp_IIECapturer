VPN加密代理流量分类(第一阶段)
参考这篇论文: [Context-aware Website Fingerprinting over Encrypted Proxies](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9488676)
# 按照协议分类
Group 1、label 0: UDP协议
Group 2、label 6: UDP+TCP/TLS
Group 3、label 8: wire gurard
Group 4、label 4，5，7，9: TCP
Group 5、label 1，2，3，10: TCP/TLS
# 针对4，5进行分类：
1、Group4提取
（1）窗口信息
（2）时间序列信息（packet length，payload前32个包的前32个字节）
（3）流统计特征信息（overall statistic，packet timing，packet size，max packet size）
2、Group5提取
  在（1）（2）（3）的基础上增加comment type序列

# 分类器
使用XGBoost