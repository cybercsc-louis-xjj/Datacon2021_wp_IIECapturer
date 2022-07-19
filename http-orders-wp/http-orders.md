# HTTP 指令序列分析

1. 首先分析sample数据

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027154323.png" style="zoom:50%;" />

   使用wireshark follow 一条http流之后，发现指令任然是CS 木马发送的指令，因此使用tshark 过滤出所有请求的uri里面包含`jquery-3.3.1.min.js`的flow

2. 找到指令payload

   分析所有与CS通信的HTTP流，发现在服务器返回的js中有一段不是正常js的payload，确认为CS的指令数据

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027154703.png" style="zoom:50%;" />

3. 发现一些指令会向CS主机post数据，为受害主机向CS主机发送数据

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027155046.png" style="zoom:50%;" />

3. 找到所有这样的payload数据后，会发现不同的指令所发送的payload长度很不一样，并且不同的指令payload长度存在一定的模式，模式如下：
   + 心跳包：payload固定为6个字符
   + sleep：只有response有payload数据, 没有post数据
   + shell: response中携带的数据量与post的数据量大体相当，比值范围在0.5到2之间
   + file: post的数据量大于response中携带的数据量，比值一般超过2
   + hash: 存在大量的response数据，数据量非常大，并且post的数据量很小切非常固定
   + screen：response中和post中都存在大量的数据

5. 将所有flow中的post数据和解析处理后，按照时间排序得到如下的结果文档(allResult.txt)：

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027155606.png" style="zoom:50%;" />

   利用这个文档，根据之前发现的指令模式去找出所有的指令。