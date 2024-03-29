
"Level" sütunu, sınıflandırma için kullanılabilecek olan 'risk seviyelerini' (etiketleri) içerirken, 

"EventID" sütunu ise özellikler (bağımsız değişkenler) olarak kullanılacak. 

```Log Source
https://securitydatasets.com/notebooks/atomic/windows/intro.html
```

![image](https://github.com/mrrsayarr/example/assets/64076325/5e7e9c5b-f3c3-4d02-85e0-5f1bd29af6ef)

https://ieeexplore.ieee.org/abstract/document/9678773/references#references

![image](https://github.com/mrrsayarr/example/assets/64076325/de4e9f31-37fd-428b-b26b-3eb2e03493b1)

-

https://dl.acm.org/doi/abs/10.1145/3338906.3338931

Günlük verilerinin istikrarsızlığı iki kaynaktan gelir: 1) günlük ifadelerinin evrimi ve 2) günlük verilerindeki işleme gürültüsü. Bu makalede, LogRobust adı verilen yeni bir log tabanlı anomali tespit yaklaşımı öneriyoruz. LogRobust, günlük olaylarının anlamsal bilgilerini çıkarır ve bunları anlamsal vektörler olarak temsil eder. Ardından, günlük dizilerindeki bağlamsal bilgileri yakalama ve farklı günlük olaylarının önemini otomatik olarak öğrenme yeteneğine sahip dikkat tabanlı bir Bi-LSTM modeli kullanarak anormallikleri tespit eder. Bu şekilde, LogRobust kararsız günlük olaylarını ve dizilerini tanımlayabilir ve işleyebilir. LogRobust'ı Hadoop sisteminden ve Microsoft'un gerçek bir çevrimiçi hizmet sisteminden toplanan günlükleri kullanarak değerlendirdik. Deneysel sonuçlar, önerilen yaklaşımın günlük istikrarsızlığı sorununu iyi bir şekilde ele alabildiğini ve gerçek dünyada sürekli değişen günlük verileri üzerinde doğru ve sağlam sonuçlar elde edebildiğini göstermektedir.

https://www.sciencedirect.com/science/article/pii/S0957417421015724

Bu makale, anormallik tespitine odaklanarak bilgisayar sistemleri tarafından yayılan günlüklerin analizini ele almaktadır. AutoLog olarak adlandırılan önerilen yaklaşım, günlüklerin düzenli aralıklarla örneklenmesini ve sayısal puanların hesaplanmasını içermektedir. Normatif işlemler altında toplanan puanlar, gelecekteki puanları sınıflandırmak için bir temel görevi gören yarı denetimli bir derin oto kodlayıcıyı eğitmek için kullanılır. Yaklaşım, altta yatan günlüklerin yapısı tarafından kısıtlanmaz ve eğitim zamanında anormalliklere ihtiyaç duymaz. İki endüstriyel sistemin anormalliklerini tespit etmede elde edilen sonuçlar ve yaygın olarak karşılaştırma ölçütü olarak kullanılan halka açık BG/L ve Hadoop veri kümeleri, AutoLog'un geri çağırma oranının 0,96 ile 0,99 arasında değiştiğini, hassasiyetin ise 0,93 ile 0,98 arasında olduğunu göstermektedir. Önerinin geçerliliğini göstermek için izolasyon ormanı, tek sınıf DVM, karar ağacı, vanilya otoenkoder ve varyasyonel otoenkoder ile karşılaştırmalı bir çalışma yapılmıştır.

https://ieeexplore.ieee.org/document/10126899 (CVE Kullanılarak yapılmış)

Microsoft Windows 10 ve 11'i piyasaya sürmesine rağmen, dünya çapında birçok kişisel bilgisayar hala güvenlik yamalarını yüklemeden eski Windows 7 sürümünü çalıştırıyor. Bu da saldırganların bu bilgisayarları istismar edebilmesine yol açmaktadır. Bu makalede, Windows olay günlüklerinden Windows saldırılarını tespit etmek için SHIRO adlı hafif bir sistem öneriyoruz. Sistem, CVE 2017-0143 (EternalBlue), CVE 2017-0199 (HTA) ve CVE 2019-0708 (BlueKeep) olmak üzere en kritik üç Ortak Güvenlik Açıklarına (CVE'ler) odaklanarak Windows 7 istemcilerine yönelik saldırıları tespit etmeyi amaçlamaktadır. Önerilen sistemimizi doğrulamak için çeşitli saldırıları taklit ediyor ve her saldırı türü için veri kümeleri oluşturuyoruz. Ardından günlük sunucusu her istemciden Windows olay günlüklerini toplar. Saldırılar sırasında elde edilen günlükler ile normal işlemler sırasında elde edilen günlükleri karşılaştırarak saldırıları tespit ediyoruz. Ardından, belirli olay kimliklerinden her CVE için algılama imzaları geliştiriyoruz. SHIRO kayıtlarda saldırı imzalarını bulduğunda, saldırı türünü tanımlar ve yöneticiyi uyarır. Hem önceden oluşturulmuş veri kümelerine hem de gerçek zamanlı saldırılara dayanan deneylerimiz, SHIRO'nun üç tür saldırıyı doğru bir şekilde tespit edebildiğini doğrulamaktadır. Deney sonuçları, SHIRO'nun yöneticinin güvenliği ihlal edilmiş Windows makinelerini verimli bir şekilde bulması için yararlı olduğunu kanıtlamaktadır.
