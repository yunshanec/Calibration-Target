# -*- coding: utf-8 -*-
# @Time : 2021/08/09 13:43
# @Author : yunshan
# @File : run.py
import cv2
import numpy as np

img = cv2.imread('left_img/0.png')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

img_size = gray.shape[::-1]

image_position = np.array([[760.0396118164062, 1165.35498046875], [812.6103515625, 1159.9296875], [865.263427734375, 1154.4588623046875], [918.0045166015625, 1148.993896484375], [970.7626953125, 1143.524658203125], [1023.5965576171875, 1138.0333251953125], [1076.5220947265625, 1132.5582275390625], [765.6646728515625, 1217.9595947265625], [818.2499389648438, 1212.5640869140625], [870.8987426757812, 1207.13330078125], [923.6265258789062, 1201.723388671875], [976.38623046875, 1196.310791015625], [1029.1619873046875, 1190.90478515625], [1082.20361328125, 1185.4554443359375], [771.4293212890625, 1270.849609375], [824.0032348632812, 1265.4998779296875], [876.663818359375, 1260.115966796875], [929.3221435546875, 1254.7001953125], [982.0464477539062, 1249.2806396484375], [1034.8670654296875, 1243.889892578125], [1087.7308349609375, 1238.400390625], [777.1517333984375, 1323.7344970703125], [829.7005004882812, 1318.412353515625], [882.31689453125, 1313.037109375], [935.0123291015625, 1307.66259765625], [987.736572265625, 1302.2669677734375], [1040.559326171875, 1296.8631591796875], [1093.3734130859375, 1291.4603271484375], [782.7803344726562, 1376.509033203125], [835.3339233398438, 1371.187744140625], [887.959716796875, 1365.84228515625], [940.6233520507812, 1360.50830078125], [993.3636474609375, 1355.150390625], [1046.1678466796875, 1349.78173828125], [1098.98486328125, 1344.3992919921875], [788.42724609375, 1429.2117919921875], [840.97607421875, 1423.91796875], [893.5848999023438, 1418.6314697265625], [946.2374267578125, 1413.3099365234375], [998.968017578125, 1407.9852294921875], [1051.773193359375, 1402.6298828125], [1104.6107177734375, 1397.300048828125], [794.2391357421875, 1481.874755859375], [846.6229858398438, 1476.596923828125], [899.2445068359375, 1471.3504638671875], [951.9059448242188, 1466.095703125], [1004.6342163085938, 1460.7762451171875], [1057.47314453125, 1455.462158203125], [1110.2291259765625, 1450.080322265625], [790.1411743164062, 1308.1031494140625], [842.0465698242188, 1302.9864501953125], [894.076171875, 1297.8592529296875], [946.2186889648438, 1292.716796875], [998.4718627929688, 1287.5797119140625], [1050.7720947265625, 1282.3599853515625], [1103.2489013671875, 1277.251708984375], [795.681884765625, 1360.33251953125], [847.5743408203125, 1355.2611083984375], [899.638916015625, 1350.1683349609375], [951.74560546875, 1345.0885009765625], [1003.9827270507812, 1340.015869140625], [1056.2884521484375, 1334.96337890625], [1108.8892822265625, 1329.837890625], [801.3316040039062, 1412.771484375], [853.24853515625, 1407.80859375], [905.2833862304688, 1402.800537109375], [957.3756713867188, 1397.739013671875], [1009.5731201171875, 1392.684814453125], [1061.8974609375, 1387.6414794921875], [1114.3404541015625, 1382.5296630859375], [806.9403076171875, 1465.2431640625], [858.8257446289062, 1460.30078125], [910.8541870117188, 1455.34375], [962.945068359375, 1450.3740234375], [1015.1615600585938, 1445.36328125], [1067.49072265625, 1440.3358154296875], [1119.8692626953125, 1435.2926025390625], [812.4902954101562, 1517.5296630859375], [864.3865356445312, 1512.645751953125], [916.3797607421875, 1507.7459716796875], [968.4823608398438, 1502.8221435546875], [1020.6908569335938, 1497.865234375], [1073.006103515625, 1492.907958984375], [1125.41015625, 1487.9224853515625], [818.0349731445312, 1569.7845458984375], [869.9343872070312, 1564.96630859375], [921.9265747070312, 1560.1002197265625], [974.0208740234375, 1555.22705078125], [1026.200927734375, 1550.3553466796875], [1078.5360107421875, 1545.437255859375], [1130.961181640625, 1540.5341796875], [823.7257080078125, 1622.078857421875], [875.4721069335938, 1617.2564697265625], [927.4776611328125, 1612.4346923828125], [979.60205078125, 1607.6324462890625], [1031.760986328125, 1602.7581787109375], [1084.0867919921875, 1597.9110107421875], [1136.4915771484375, 1592.99658203125], [744.309326171875, 1302.2950439453125], [795.4269409179688, 1297.423828125], [846.7089233398438, 1292.5516357421875], [898.1400146484375, 1287.6522216796875], [949.6866455078125, 1282.7586669921875], [1001.396240234375, 1277.797607421875], [1053.326904296875, 1272.849365234375], [749.7258911132812, 1354.1654052734375], [800.8239135742188, 1349.39111328125], [852.1419677734375, 1344.575927734375], [903.5610961914062, 1339.7666015625], [955.1351318359375, 1334.9559326171875], [1006.8080444335938, 1330.1270751953125], [1058.8656005859375, 1325.2581787109375], [755.2474365234375, 1406.326904296875], [806.3599853515625, 1401.6456298828125], [857.6773071289062, 1396.93603515625], [909.0670166015625, 1392.1397705078125], [960.6063232421875, 1387.3902587890625], [1012.3319091796875, 1382.606689453125], [1064.2364501953125, 1377.7763671875], [760.7634887695312, 1458.4324951171875], [811.856201171875, 1453.8272705078125], [863.1162109375, 1449.199951171875], [914.543212890625, 1444.5230712890625], [966.1058349609375, 1439.827392578125], [1017.8375244140625, 1435.1180419921875], [1069.7113037109375, 1430.3839111328125], [766.213134765625, 1510.4359130859375], [817.3035888671875, 1505.8868408203125], [868.5771484375, 1501.33740234375], [919.9861450195312, 1496.7431640625], [971.55322265625, 1492.134765625], [1023.2620239257812, 1487.49267578125], [1075.132568359375, 1482.8358154296875], [771.6329956054688, 1562.3822021484375], [822.7289428710938, 1557.8968505859375], [874.0087280273438, 1553.41943359375], [925.4244995117188, 1548.90283203125], [976.9901733398438, 1544.3731689453125], [1028.7125244140625, 1539.8226318359375], [1080.5777587890625, 1535.257080078125], [777.1928100585938, 1614.36962890625], [828.1689453125, 1609.9393310546875], [879.465576171875, 1605.5238037109375], [930.9068603515625, 1601.0618896484375], [982.4539794921875, 1596.64990234375], [1034.1983642578125, 1592.1700439453125], [1086.0926513671875, 1587.6427001953125], [716.2703247070312, 1237.27392578125], [769.40478515625, 1231.827880859375], [822.581787109375, 1226.3563232421875], [875.8322143554688, 1220.9044189453125], [929.08203125, 1215.4307861328125], [982.41015625, 1209.9556884765625], [1035.8822021484375, 1204.4638671875], [721.6176147460938, 1290.5821533203125], [774.7476806640625, 1285.1298828125], [827.9638671875, 1279.6663818359375], [881.2501831054688, 1274.238037109375], [934.5003051757812, 1268.8394775390625], [987.8357543945312, 1263.414794921875], [1041.415771484375, 1257.9820556640625], [727.0833129882812, 1344.13037109375], [780.2247924804688, 1338.76513671875], [833.4762573242188, 1333.35498046875], [886.7086791992188, 1327.9122314453125], [940.0057983398438, 1322.4862060546875], [993.3661499023438, 1317.067138671875], [1046.8370361328125, 1311.5777587890625], [732.497802734375, 1397.751220703125], [785.6431884765625, 1392.4053955078125], [838.9093627929688, 1387.0225830078125], [892.1820678710938, 1381.632568359375], [945.5029907226562, 1376.211669921875], [998.9073486328125, 1370.810791015625], [1052.342529296875, 1365.38427734375], [737.89599609375, 1451.2877197265625], [791.0614624023438, 1445.9622802734375], [844.3028564453125, 1440.616455078125], [897.5972900390625, 1435.2447509765625], [950.9545288085938, 1429.8782958984375], [1004.3818969726562, 1424.4656982421875], [1057.8240966796875, 1419.076171875], [743.245849609375, 1504.83056640625], [796.4556274414062, 1499.5125732421875], [849.7053833007812, 1494.2025146484375], [903.0313110351562, 1488.8729248046875], [956.3983154296875, 1483.509521484375], [1009.841552734375, 1478.149169921875], [1063.3310546875, 1472.7607421875], [748.7821655273438, 1558.4027099609375], [801.8624267578125, 1553.0816650390625], [855.1764526367188, 1547.7960205078125], [908.4921264648438, 1542.501953125], [961.8742065429688, 1537.175048828125], [1015.367919921875, 1531.8431396484375], [1068.8453369140625, 1526.4989013671875], [684.4423217773438, 1231.5396728515625], [736.307373046875, 1226.3509521484375], [788.3030395507812, 1221.1317138671875], [840.409912109375, 1215.9188232421875], [892.5542602539062, 1210.722412109375], [944.8546142578125, 1205.470458984375], [997.328857421875, 1200.2349853515625], [689.5343627929688, 1283.9794921875], [741.4005737304688, 1278.8128662109375], [793.4765014648438, 1273.62744140625], [845.5838623046875, 1268.4732666015625], [897.7833251953125, 1263.3382568359375], [950.0301513671875, 1258.1907958984375], [1002.6676635742188, 1252.9886474609375], [694.7083740234375, 1336.6649169921875], [746.6431884765625, 1331.6031494140625], [798.7001953125, 1326.509765625], [850.8187255859375, 1321.3603515625], [903.04052734375, 1316.2325439453125], [955.3436889648438, 1311.12060546875], [1007.8245239257812, 1305.8907470703125], [699.905517578125, 1389.4520263671875], [751.8475952148438, 1384.4400634765625], [803.89208984375, 1379.3760986328125], [856.0479736328125, 1374.3055419921875], [908.3019409179688, 1369.2176513671875], [960.649169921875, 1364.112548828125], [1013.0995483398438, 1359.0032958984375], [705.0474243164062, 1442.1439208984375], [756.9882202148438, 1437.177490234375], [809.0657348632812, 1432.162353515625], [861.2523193359375, 1427.14208984375], [913.5286254882812, 1422.12255859375], [965.890380859375, 1417.0692138671875], [1018.3825073242188, 1412.0023193359375], [710.1630859375, 1494.84033203125], [762.152099609375, 1489.923095703125], [814.2371215820312, 1484.9676513671875], [866.4417114257812, 1480.0023193359375], [918.745361328125, 1475.0115966796875], [971.1483154296875, 1470.0244140625], [1023.6571044921875, 1465.01123046875], [715.4364624023438, 1547.583740234375], [767.32080078125, 1542.6656494140625], [819.4539184570312, 1537.7686767578125], [871.6849975585938, 1532.887451171875], [923.9862670898438, 1527.953369140625], [976.450439453125, 1522.9822998046875], [1028.96240234375, 1518.0166015625], [772.7057495117188, 1207.664306640625], [824.8578491210938, 1202.582763671875], [877.2041625976562, 1197.46728515625], [929.7177124023438, 1192.31591796875], [982.349853515625, 1187.1729736328125], [1035.152099609375, 1181.99365234375], [1088.13037109375, 1176.8258056640625], [777.6842651367188, 1260.644775390625], [829.9119873046875, 1255.636962890625], [882.3374633789062, 1250.60888671875], [934.814697265625, 1245.5611572265625], [987.5188598632812, 1240.512451171875], [1040.28662109375, 1235.4520263671875], [1093.43994140625, 1230.33935546875], [782.8262329101562, 1313.9991455078125], [835.0730590820312, 1309.0751953125], [887.488037109375, 1304.1240234375], [940.0255126953125, 1299.1097412109375], [992.7029418945312, 1294.0947265625], [1045.52001953125, 1289.092041015625], [1098.55322265625, 1284.00146484375], [787.9299926757812, 1367.408203125], [840.2015991210938, 1362.5439453125], [892.6200561523438, 1357.640869140625], [945.2071533203125, 1352.727294921875], [997.9093627929688, 1347.7957763671875], [1050.793701171875, 1342.8399658203125], [1103.7906494140625, 1337.87841796875], [792.9631958007812, 1420.718505859375], [845.2682495117188, 1415.92822265625], [897.7171020507812, 1411.10009765625], [950.31201171875, 1406.2664794921875], [1003.0634155273438, 1401.4100341796875], [1055.9840087890625, 1396.50732421875], [1109.005126953125, 1391.6195068359375], [798.0199584960938, 1474.0469970703125], [850.3322143554688, 1469.315673828125], [902.8212280273438, 1464.5928955078125], [955.4429931640625, 1459.82373046875], [1008.21533203125, 1455.032958984375], [1061.1600341796875, 1450.2137451171875], [1114.2425537109375, 1445.3914794921875], [803.2301635742188, 1527.3822021484375], [855.4434204101562, 1522.738525390625], [907.9456176757812, 1518.06640625], [960.61376953125, 1513.407958984375], [1013.3876953125, 1508.7115478515625], [1066.402587890625, 1503.9677734375], [1119.506103515625, 1499.165771484375], [731.6129150390625, 1326.9920654296875], [784.8848266601562, 1321.58984375], [838.21728515625, 1316.1728515625], [891.6128540039062, 1310.76806640625], [945.0025634765625, 1305.347412109375], [998.4297485351562, 1299.9071044921875], [1051.9957275390625, 1294.487060546875], [736.5955810546875, 1380.498046875], [789.9093017578125, 1375.0821533203125], [843.3093872070312, 1369.736328125], [896.734619140625, 1364.325927734375], [950.1627197265625, 1358.9932861328125], [1003.6301879882812, 1353.564208984375], [1057.351318359375, 1348.1912841796875], [741.731689453125, 1434.2923583984375], [795.0942993164062, 1428.97900390625], [848.5116577148438, 1423.633056640625], [901.9320068359375, 1418.239013671875], [955.4080200195312, 1412.8665771484375], [1008.9457397460938, 1407.47119140625], [1062.580322265625, 1402.009033203125], [746.8300170898438, 1488.2506103515625], [800.21240234375, 1482.9166259765625], [853.6471557617188, 1477.5924072265625], [907.157958984375, 1472.2431640625], [960.689208984375, 1466.886474609375], [1014.2537841796875, 1461.4998779296875], [1067.87841796875, 1456.1065673828125], [751.8723754882812, 1542.124755859375], [805.2899169921875, 1536.8408203125], [858.7904052734375, 1531.51806640625], [912.3062133789062, 1526.1983642578125], [965.8797607421875, 1520.8671875], [1019.5150756835938, 1515.495849609375], [1073.17919921875, 1510.122802734375], [756.9003295898438, 1596.1231689453125], [810.368896484375, 1590.8074951171875], [863.9044799804688, 1585.4984130859375], [917.47119140625, 1580.202392578125], [971.1151733398438, 1574.8795166015625], [1024.7584228515625, 1569.547607421875], [1078.501708984375, 1564.21044921875], [762.0734252929688, 1650.218505859375], [815.4701538085938, 1644.899169921875], [869.08447265625, 1639.6104736328125], [922.6959838867188, 1634.327880859375], [976.3538208007812, 1629.000732421875], [1030.1165771484375, 1623.651611328125], [1083.8336181640625, 1618.3287353515625], [821.312255859375, 1262.9730224609375], [874.342041015625, 1257.7567138671875], [927.4329223632812, 1252.5216064453125], [980.6475830078125, 1247.27197265625], [1033.9337158203125, 1242.029052734375], [1087.290771484375, 1236.734375], [1140.7998046875, 1231.457763671875], [826.1613159179688, 1316.4293212890625], [879.200927734375, 1311.22412109375], [932.3713989257812, 1306.0264892578125], [985.6203002929688, 1300.8135986328125], [1038.958251953125, 1295.68359375], [1092.34228515625, 1290.45458984375], [1146.0462646484375, 1285.172119140625], [831.0943603515625, 1370.1761474609375], [884.212646484375, 1365.081787109375], [937.4193115234375, 1359.93798828125], [990.6755981445312, 1354.7503662109375], [1044.052490234375, 1349.5726318359375], [1097.513427734375, 1344.4000244140625], [1151.0989990234375, 1339.1436767578125], [836.0301513671875, 1424.0936279296875], [889.1627807617188, 1419.0087890625], [942.3989868164062, 1413.91650390625], [995.7241821289062, 1408.8001708984375], [1049.167724609375, 1403.64794921875], [1102.6739501953125, 1398.4957275390625], [1156.278076171875, 1393.3262939453125], [840.89990234375, 1477.9666748046875], [894.0704956054688, 1472.9281005859375], [947.3567504882812, 1467.869873046875], [1000.7263793945312, 1462.788818359375], [1054.2115478515625, 1457.6845703125], [1107.789306640625, 1452.5609130859375], [1161.4375, 1447.4373779296875], [845.7654418945312, 1531.861328125], [898.9900512695312, 1526.8746337890625], [952.3189697265625, 1521.8602294921875], [1005.7423706054688, 1516.8465576171875], [1059.2794189453125, 1511.782958984375], [1112.890380859375, 1506.7060546875], [1166.6474609375, 1501.63671875], [850.8040161132812, 1585.8763427734375], [903.9349975585938, 1580.893310546875], [957.3340454101562, 1575.948974609375], [1010.807373046875, 1570.97216796875], [1064.36279296875, 1565.976806640625], [1118.1055908203125, 1560.9591064453125], [1171.811767578125, 1555.8603515625], [744.9337768554688, 1388.8248291015625], [797.254638671875, 1383.997802734375], [849.6883544921875, 1379.130126953125], [902.320556640625, 1374.2708740234375], [955.0679321289062, 1369.38134765625], [1007.941162109375, 1364.46630859375], [1061.04443359375, 1359.564208984375], [749.5054321289062, 1442.1136474609375], [801.83203125, 1437.31396484375], [854.3934936523438, 1432.5819091796875], [907.0425415039062, 1427.825439453125], [959.8397827148438, 1423.026611328125], [1012.7330932617188, 1418.15673828125], [1065.9881591796875, 1413.3607177734375], [754.1676635742188, 1495.7493896484375], [806.56201171875, 1491.091796875], [859.150390625, 1486.395751953125], [911.8390502929688, 1481.64111328125], [964.6439208984375, 1476.918701171875], [1017.6240234375, 1472.1531982421875], [1070.802001953125, 1467.2923583984375], [758.7982177734375, 1549.4898681640625], [811.2447509765625, 1544.9073486328125], [863.852783203125, 1540.26171875], [916.6008911132812, 1535.61669921875], [969.506103515625, 1530.928955078125], [1022.5276489257812, 1526.2322998046875], [1075.712890625, 1521.4927978515625], [763.3797607421875, 1603.255859375], [815.8811645507812, 1598.7210693359375], [868.5534057617188, 1594.1300048828125], [921.341796875, 1589.5391845703125], [974.28564453125, 1584.926513671875], [1027.3538818359375, 1580.291259765625], [1080.5994873046875, 1575.6419677734375], [767.9407958984375, 1657.0933837890625], [820.5177001953125, 1652.59619140625], [873.2185668945312, 1648.0758056640625], [926.0968017578125, 1643.520263671875], [979.0828857421875, 1638.9739990234375], [1032.2342529296875, 1634.406494140625], [1085.4996337890625, 1629.8648681640625], [772.656005859375, 1711.0428466796875], [825.1605834960938, 1706.563720703125], [877.9506225585938, 1702.1104736328125], [930.8633422851562, 1697.646484375], [983.9129028320312, 1693.125], [1037.109619140625, 1688.607421875], [1090.46044921875, 1684.1214599609375], [1099.05029296875, 1217.4136962890625], [1142.5252685546875, 1186.005126953125], [1186.0816650390625, 1154.572509765625], [1229.6732177734375, 1123.0927734375], [1273.304443359375, 1091.598876953125], [1317.0347900390625, 1059.9920654296875], [1360.8695068359375, 1028.3648681640625], [1130.3092041015625, 1261.0518798828125], [1173.84912109375, 1229.6839599609375], [1217.4490966796875, 1198.2440185546875], [1261.0953369140625, 1166.8087158203125], [1304.7906494140625, 1135.3294677734375], [1348.5386962890625, 1103.7979736328125], [1392.51318359375, 1072.1497802734375], [1161.860595703125, 1304.92626953125], [1205.4654541015625, 1273.5963134765625], [1249.125244140625, 1242.1639404296875], [1292.7398681640625, 1210.7049560546875], [1336.4588623046875, 1179.228759765625], [1380.25048828125, 1147.705810546875], [1424.10205078125, 1116.0589599609375], [1193.4697265625, 1348.8636474609375], [1237.0484619140625, 1317.5693359375], [1280.7314453125, 1286.152587890625], [1324.4671630859375, 1254.7257080078125], [1368.2127685546875, 1223.2327880859375], [1412.0155029296875, 1191.7122802734375], [1455.902099609375, 1160.1246337890625], [1224.92626953125, 1392.766845703125], [1268.5599365234375, 1361.473876953125], [1312.2821044921875, 1330.0830078125], [1356.0423583984375, 1298.6783447265625], [1399.8253173828125, 1267.2138671875], [1443.6884765625, 1235.701416015625], [1487.5823974609375, 1204.1392822265625], [1256.3941650390625, 1436.683349609375], [1300.0738525390625, 1405.4044189453125], [1343.837158203125, 1374.0579833984375], [1387.6162109375, 1342.634765625], [1431.459228515625, 1311.197265625], [1475.3690185546875, 1279.69677734375], [1519.329833984375, 1248.1474609375], [1288.03369140625, 1480.55126953125], [1331.6109619140625, 1449.36474609375], [1375.41552734375, 1418.03076171875], [1419.26318359375, 1386.643798828125], [1463.1409912109375, 1355.231689453125], [1507.0927734375, 1323.7197265625], [1551.117919921875, 1292.219970703125], [1018.8826293945312, 1238.6715087890625], [1061.88232421875, 1207.5264892578125], [1105.066650390625, 1176.317138671875], [1148.3209228515625, 1145.0146484375], [1191.6378173828125, 1113.7080078125], [1235.0439453125, 1082.2757568359375], [1278.6522216796875, 1050.7501220703125], [1049.7451171875, 1282.2413330078125], [1092.863037109375, 1251.1514892578125], [1136.095947265625, 1219.9498291015625], [1179.4464111328125, 1188.7021484375], [1222.8531494140625, 1157.416015625], [1266.2987060546875, 1126.0732421875], [1310.0576171875, 1094.54833984375], [1080.9395751953125, 1326.1026611328125], [1124.1351318359375, 1295.0367431640625], [1167.46630859375, 1263.87646484375], [1210.785888671875, 1232.640869140625], [1254.230224609375, 1201.3154296875], [1297.7908935546875, 1169.9979248046875], [1341.447021484375, 1138.504638671875], [1112.19482421875, 1370.050537109375], [1155.440673828125, 1339.010009765625], [1198.79541015625, 1307.89892578125], [1242.2359619140625, 1276.6571044921875], [1285.761474609375, 1245.383056640625], [1329.38134765625, 1214.0491943359375], [1373.0667724609375, 1182.642578125], [1143.3538818359375, 1413.9791259765625], [1186.645751953125, 1382.957763671875], [1230.067138671875, 1351.8729248046875], [1273.574951171875, 1320.6912841796875], [1317.1925048828125, 1289.4212646484375], [1360.879150390625, 1258.0938720703125], [1404.6431884765625, 1226.70361328125], [1174.5345458984375, 1457.9287109375], [1217.9049072265625, 1426.9681396484375], [1261.37109375, 1395.9019775390625], [1304.9798583984375, 1364.7369384765625], [1348.6456298828125, 1333.512939453125], [1392.3966064453125, 1302.2110595703125], [1436.223876953125, 1270.8394775390625], [1205.896240234375, 1501.9000244140625], [1249.20849609375, 1471.0126953125], [1292.79052734375, 1439.9725341796875], [1336.4339599609375, 1408.85498046875], [1380.1348876953125, 1377.6640625], [1423.9930419921875, 1346.3582763671875], [1467.907470703125, 1315.0491943359375], [1011.4180297851562, 1356.794189453125], [1054.6497802734375, 1325.63720703125], [1098.031494140625, 1294.38720703125], [1141.560791015625, 1263.0316162109375], [1185.193603515625, 1231.583251953125], [1228.9747314453125, 1200.0496826171875], [1272.953125, 1168.3582763671875], [1042.3367919921875, 1401.0802001953125], [1085.704345703125, 1369.993896484375], [1129.203125, 1338.757568359375], [1172.8157958984375, 1307.478515625], [1216.5489501953125, 1276.041015625], [1260.4210205078125, 1244.59423828125], [1304.571533203125, 1212.9532470703125], [1073.624267578125, 1445.6383056640625], [1117.0584716796875, 1414.6187744140625], [1160.677978515625, 1383.436767578125], [1204.3414306640625, 1352.1424560546875], [1248.1365966796875, 1320.771240234375], [1292.1190185546875, 1289.302978515625], [1336.227783203125, 1257.6627197265625], [1104.9700927734375, 1490.3448486328125], [1148.496337890625, 1459.312744140625], [1192.1815185546875, 1428.215576171875], [1235.9483642578125, 1396.9578857421875], [1279.8778076171875, 1365.620849609375], [1323.9521484375, 1334.1544189453125], [1368.102783203125, 1302.621337890625], [1136.2574462890625, 1535.02197265625], [1179.879150390625, 1504.0491943359375], [1223.6376953125, 1472.9345703125], [1267.51220703125, 1441.77978515625], [1311.5235595703125, 1410.459228515625], [1355.662353515625, 1379.056640625], [1399.934814453125, 1347.5167236328125], [1167.5721435546875, 1579.739501953125], [1211.277099609375, 1548.8060302734375], [1255.138916015625, 1517.7625732421875], [1299.12353515625, 1486.609130859375], [1343.205078125, 1455.354736328125], [1387.4576416015625, 1423.9979248046875], [1431.81787109375, 1392.5186767578125], [1199.0858154296875, 1624.468505859375], [1242.7630615234375, 1593.649169921875], [1286.7301025390625, 1562.6395263671875], [1330.8250732421875, 1531.52880859375], [1375.00048828125, 1500.3350830078125], [1419.3468017578125, 1468.9814453125], [1463.78369140625, 1437.5582275390625]])
object_position = np.array([[0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0], [0.0, 0.0, 0.0], [0.0, 2.5, 0.0], [0.0, 5.0, 0.0], [0.0, 7.5, 0.0], [0.0, 10.0, 0.0], [0.0, 12.5, 0.0], [0.0, 15.0, 0.0], [2.5, 0.0, 0.0], [2.5, 2.5, 0.0], [2.5, 5.0, 0.0], [2.5, 7.5, 0.0], [2.5, 10.0, 0.0], [2.5, 12.5, 0.0], [2.5, 15.0, 0.0], [5.0, 0.0, 0.0], [5.0, 2.5, 0.0], [5.0, 5.0, 0.0], [5.0, 7.5, 0.0], [5.0, 10.0, 0.0], [5.0, 12.5, 0.0], [5.0, 15.0, 0.0], [7.5, 0.0, 0.0], [7.5, 2.5, 0.0], [7.5, 5.0, 0.0], [7.5, 7.5, 0.0], [7.5, 10.0, 0.0], [7.5, 12.5, 0.0], [7.5, 15.0, 0.0], [10.0, 0.0, 0.0], [10.0, 2.5, 0.0], [10.0, 5.0, 0.0], [10.0, 7.5, 0.0], [10.0, 10.0, 0.0], [10.0, 12.5, 0.0], [10.0, 15.0, 0.0], [12.5, 0.0, 0.0], [12.5, 2.5, 0.0], [12.5, 5.0, 0.0], [12.5, 7.5, 0.0], [12.5, 10.0, 0.0], [12.5, 12.5, 0.0], [12.5, 15.0, 0.0], [15.0, 0.0, 0.0], [15.0, 2.5, 0.0], [15.0, 5.0, 0.0], [15.0, 7.5, 0.0], [15.0, 10.0, 0.0], [15.0, 12.5, 0.0], [15.0, 15.0, 0.0]])

image_position = np.array(image_position).reshape(12, 49, 1, 2).astype(np.float32)
object_position = np.array(object_position).reshape(12, 49, 1, 3).astype(np.float32)

ret,mtx,dist,rvecs,tvecs = cv2.calibrateCamera(object_position,image_position,img_size,None,None)
print("mtx 内参数矩阵:\n", mtx)  # 内参数矩阵
print(
    "dist 畸变系数:\n", dist
)  # 畸变系数   distortion cofficients = (k_1,k_2,p_1,p_2,k_3)
print("rvecs 旋转向量:\n", rvecs)  # 旋转向量  # 外参数
print("tvecs 平移向量:\n", tvecs)  # 平移向量  # 外参数
