# HTTPS 指令分析

1. https指令理论上和http指令分析工作相同，只不过由于数据经过加密之后无法解析出具体的payload数据，分析长度的时候不是很好区别，但是本质上还是找不同指令发送数据量的却别。

2. 分析sample里面的几个文件，由于找不到具体的payload长度，因此这里的长度数据是tshark工具中通过`ssl.app_data`字段的字符长度，得到结果如下(详细结果在ipynb里面)，可以发现1775-18095是一对明显的心跳包，而不同于这种心跳包的就是发送的额外指令数据或者返回的数据。

   - ​	这里多出18095的数据可以看做CS服务器下发的任务数据，而下面额外返回的1631数据可以看做受害主机的响应数据。

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027162123.png" style="zoom:33%;" />

   + file指令和shell指令的区别在于响应数据量和下载的数据量的比值不同(和HTTP及DNS的特征相同)

     <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027162834.png" style="zoom:33%;" />

   + sleep指令只有下载数据，而没有返回响应数据

     <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027163001.png" style="zoom:33%;" />

   + hash和screen的模式非常明显，hash存在大量数据下载但是上传数据量不大，而screen同时存在大量上传和下载数据。

3. 将所有的数据处理成上面这样，得到文件`httpsAllResult.txt`，然后按照不同的模式去寻找指令。

4. 上面的分析实际验证的时候会有一些偏差。另外我们还发现如下的规律，使用wireshark工具导出conversation后，对比上面的模式，会发现所有存在指令的流的duration都比其他的短(1s左右)，这可以辅助定位指令流的位置

   <img src="https://ipic-picgo.oss-cn-beijing.aliyuncs.com/20211027163406.png" style="zoom:50%;" />

   

