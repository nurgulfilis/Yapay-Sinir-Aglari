# Bu repo Yapay Sinir Ağları dersi adı altında yapmış olduğum çalışmalardan oluşur.

Burada 3 adet yapay sinir ağı oluşturulmuştur. Bunlardan ilki;

Feed Forward:
•Forward propagation için, input olarak şu X matrisini verin (tensöre çevirmeyi unutmayın):
X =    1    2    3    Satırlar veriler (sample’lar), kolonlar öznitelikler (feature’lar).
4  5   6
•Bir adet hidden layer olsun ve içinde tanh aktivasyon fonksiyonu olsun
•Hidden layer’da 50 nöron olsun
•Bir adet output layer olsun, tek nöronu olsun ve içinde sigmoid aktivasyon fonksiyonu olsun Tanh fonksiyonu:
f (x) = exp(x)−exp(−x)
exp(x)+exp(−x)

Sigmoid fonksiyonu:
f (x) =      1     
1+exp(−x)

Pytorch  kütüphanesi  ile,  ama  kutuphanenin  hazır  aktivasyon  fonksiyonlarını  kullanmadan,  formulunu  verdigim  iki  aktivasyon  fonksiyonunun  kodunu  ikinci  haftada  yaptıgımız gibi  kendiniz  yazarak  bu  yapay  sinir  ağını  oluşturun 

5 Multilayer Perceptron (MLP):
Bu  bolumdeki  sorularda  benim  vize  ile  beraber  paylastıgım  Prensesi  I˙yile¸stir  (Cure  The Princess) Veri Seti par¸caları kullanılacak. Hikaye ¸s¨oyle (soruyu ¸c¨ozmek i¸cin hikaye kısmını okumak  zorunda  de˘gilsiniz):

“Bir  zamanlar,  ¸cok  uzaklarda  bir  u¨lkede,  a˘gır  bir  hastalı˘ga  yakalanmı¸s  bir  prenses  ya¸sarmı¸s.  U¨ lkenin kralı ve krali¸cesi onu iyile¸stirmek i¸cin ellerinden gelen her ¸seyi yapmı¸slar, ancak denedikleri hi¸cbir ¸care i¸se yaramamı¸s.
Yerel  bir  grup  k¨oylu¨,  herhangi  bir  hastalı˘gı  iyile¸stirmek  i¸cin  gu¨cu¨  oldu˘gu  s¨oylenen  bir  dizi  sihirli
malzemeden bahsederek kral ve krali¸ceye yakla¸smı¸s. Ancak, k¨oylu¨ler kral ile krali¸ceyi, bu malzemelerin etkilerinin  patlayıcı  olabilece˘gi  ve  son  zamanlarda  ya¸sanan  kuraklıklar  nedeniyle  bu  malzemelerden sadece birkac¸ının herhangi bir zamanda bulunabilece˘gi konusunda uyarmı¸slar. Ayrıca, sadece deneyimli bir  simyacı  bu  ¨ozelliklere  sahip  patlayıcı  ve  az  bulunan  malzemelerin  belirli  bir  kombinasyonunun prensesi iyile¸stirece˘gini belirleyebilecekmi¸s.
Kral  ve  krali¸ce  kızlarını  kurtarmak  i¸cin  umutsuzlar,  bu  yu¨zden  u¨lkedeki  en  iyi  simyacıyı  bulmak  i¸cin yola  ¸cıkmı¸slar.  Da˘gları  tepeleri  a¸smı¸slar  ve  nihayet  ”Yapay  Sinir  A˘gları  Uzmanı”  olarak  bilinen  yeni bir sihirli sanatın ustası olarak u¨n yapmı¸s bir simyacı bulmu¸slar.
Simyacı  ¨once  k¨oylu¨lerin  iddialarını  ve  her  bir  malzemenin  alınan  miktarlarını,  ayrıca  iyile¸smeye  yol a¸cıp  a¸cmadı˘gını  incelemi¸s.  Simyacı  biliyormu¸s  ki  bu  prensesi  iyile¸stirmek  i¸cin  tek  bir ¸sansı  varmı¸s  ve bunu  do˘gru  yapmak  zorundaymı¸s.  (Original  source:  https://www.kaggle.com/datasets/unmoved/ cure-the-princess)
(Buradan itibaren ChatGPT ve Dr. Ulya Bayram’a ait hikayenin devamı)
Simyacı,  bu¨yu¨lu¨  bile¸senlerin  farklı  kombinasyonlarını  analiz  etmek  ve  denemek  i¸cin  gu¨nler  harcamı¸s. Sonunda  birkac¸  denemenin  ardından  prensesi  iyile¸stirecek  c¸e¸sitli  karı¸sım  kombinasyonları  bulmu¸s  ve bunları bir veri setinde toplamı¸s. Daha sonra bu veri setini e˘gitim, validasyon ve test setleri olarak u¨¸c par¸caya  ayırmı¸s  ve  bunun  u¨zerinde  bir  yapay  sinir  a˘gı  e˘giterek  kendi  y¨ontemi  ile  prensesi  iyile¸stirme ihtimalini  hesaplamı¸s  ve  ikna  olunca  kral  ve  kraliceye  haber  vermi¸s.  Heyecanlı  ve  umutlu  olan  kral ve  krali¸ce,  simyacının  prensese  hazırladı˘gı  ilacı  vermesine  izin  vermi¸s  ve  ila¸c  i¸se  yaramı¸s  ve  prenses hastalı˘gından kurtulmu¸s.
Kral ve krali¸ce, kızlarının hayatını kurtardı˘gı icin simyacıya krallıkta kalması ve ¸calı¸smalarına devam etmesi i¸cin bu¨yu¨k bir ara¸stırma bu¨tcesi ve ¸cok sayıda GPU’su olan bir server vermi¸s. I˙yile¸sen prenses de kendisini iyile¸stiren y¨ontemleri ¨o˘grenmeye merak salıp, krallıktaki u¨niversitenin bilgisayar mu¨hendisli˘gi b¨olu¨mu¨ne  girmi¸s  ve  mezun  olur  olmaz  da  simyacının  yanında,  onun  ara¸stırma  grubunda  c¸alı¸smaya ba¸slamı¸s.  Uzun  yıllar  birlikte  krallıktaki  insanlara,  hayvanlara  ve  do˘gaya  faydalı  olacak  yazılımlar geli¸stirmi¸sler,  ve  simyacı  emekli  oldu˘gunda  prenses  hem  ara¸stırma  grubunun  hem  de  krallı˘gın  lideri olarak hayatına devam etmi¸s.
Prenses,  kendisini  iyile¸stiren  veri  setini  de,  gelecekte  onların  izinden  gidecek  bilgisayar  mu¨hendisi prensler  ve  prensesler  ba¸skalarına  faydalı  olabilecek  yapay  sinir  a˘gları  olu¸sturmayı  ¨o˘grensinler  diye halka ac¸mı¸s ve sınavlarda kullanılmasını salık vermi¸s.”
I˙ki  hidden  layer’lı  bir  Multilayer  Perceptron  (MLP)  olu¸sturun  be¸sinci  ve  altıncı  haf- talarda  yaptı˘gımız  gibi.  Hazır  aktivasyon  fonksiyonlarını  kullanmak  serbest.  I˙lk  hidden layer’da  100,  ikinci  hidden  layer’da  50  n¨oron  olsun.  Hidden  layer’larda  ReLU,  output layer’da sigmoid aktivasyonu olsun.
Output  layer’da  ka¸c  n¨oron  olaca˘gını  veri  setinden  bakıp  bulacaksınız.  Elbette  bu  veriye uygun Cross Entropy loss y¨ontemini uygulayacaksınız. Optimizasyon i¸cin Stochastic Gra- dient Descent yeterli. Epoch sayınızı ve learning rate’i validasyon seti u¨zerinde denemeler yaparak (loss’lara overfit var mı diye bakarak) kendiniz belirleyeceksiniz. Batch size’ı 16 se¸cebilirsiniz.

Ve sonuncusu;

Bir  önceki  sorudaki  Prensesi İyilestir  problemindeki  yapay  sinir  ağına  seçtiğimiz  herhangi  iki  farklı  regülarizasyon  yöntemi  ekleme
