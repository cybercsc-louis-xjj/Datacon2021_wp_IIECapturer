# DNS 指令分析

## 一、分析DNS指令发送机制

1. 观察sample数据会发现，CS 使用dns作为通道发送指令的时候，是利用dns协议的TXT记录类型，将数据加密后拆分开在多个dns txt数据包里面返回, txt数据相当于从受害主机从CS服务器下载指令

   ![](https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027161047.png)

2. 而受害主机使用post.**.ns1.ssltestdomain.xyz的请求报文上传数据，数据被加载在域名中间，并且通过后面的序号(序号递增)指定为同一批序号

![](https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027161346.png)

3. 使用脚本将所有的DNS txt数据和A记录里面post的数据分析出来得到下面的文件(`dnsAllDataWithData.txt`)，

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027161523.png" style="zoom:50%;" />

4. 利用HTTP指令找到的不同指令的上传下载数据量不同的模式，从如上的文件中识别出不同指令。