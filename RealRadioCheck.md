# Результаты проверки работы с реальной рацией #

Автор: Йолаф

### Главный вывод ###

Всё работает! :)

### Тестовый стенд ###

Использованы рации [Yaesu FT-60R](http://1mhz.ru/shop/index.php?productID=131), 5 Вт и [Midland GXT-650](http://1mhz.ru/shop/index.php?productID=126) GMRS 5 Вт.

Подключение осуществлялось к разъёму задней панели встроенной звуковой карты материнской платы [ASUS P5WD2 Premium/ WiFi-TV Edition](http://www.asus.com/product.aspx?P_ID=RGSwH7V139hRZ34V).

Для подключения использован стандартный кабель стерео-миниджек-папа/стерео-миниджек-папа длиной 1.5 метра.

Подключение к компьютеру и рации Midland выполнялось напрямую, подключение к рации Yaesu - через стандартный переходник [CT-44](http://1mhz.ru/shop/index.php?productID=239).

Стандартная для данной материнской платы программа [Realtek HD Audio Manager](http://www.realtek.com.tw/downloads/downloadsCheck.aspx?langid=1&pfid=24&level=4&conn=3&downtypeid=3) (RTHDCPL.exe) автоматически распознала подключение и предложила выбрать тип подключённого устройста. Были опробованы варианты Line In и Mic In, никакой разницы в работе замечено не было.

На компьютере была установлена операционная система [Microsoft Windows XP](http://www.microsoft.com/windows/windows-xp/default.aspx) Professional 2002 32-bit SP3 English.

Для контроля уровня сигнала на входе использовалась программа [Adobe Audition](http://www.adobe.com/products/audition/) 2.0.

### Проблемы ###

#### 1 ####

Как и предполагалось, граница "уровня шума", выставленная в 0, работает плохо, ложные срабатывания случаются достаточно часто, так как микрофонный вход сам по себе шумит достаточно сильно. Необходима возможность ручной установки "уровня шума".

#### 2 ####

"Хвост" тишины в конце каждого файла действительно весьма велик и создаёт неудобства при прослушивании. Желательно его обрезать.

#### 3 ####

Визуальный контроль уровня записи и возможность её подстройки (возможно штатными средствами системы) необходим, так как на слух очень легко ошибиться и выставить уровень либо слишком низкий (что приведёт к ухудшению качества сигнала и увеличению относительного уровня шумов), либо слишком высокий (что приведёт к искажению сигнала и возможному повреждению звуковой карты).

### Тонкости ###

#### 1 ####

Разъёмы в рациях - моно (2-контактные) с длинным первым контактом, разъёмы на кабеле - стерео (3-контактные), разъём в компьютере - стерео (3-контактный), с землёй на среднем контакте, левом канале на концевом контакте, правом канале на первом контакте. Как следствие, при подключении сигнал оказывается только в левом канале, а правый оказывается замкнут на землю.

**Решение 1:** подготовить специальный кабель, в котором перепаять "на перекрест" жилы концевого и среднего контактов. В итоге одинаковый сигнал окажется в обоих каналах.

**Решение 2:** не обращать внимание, так как программа выдаёт монофонические файлы, и видимо вполне адекватно обходится с отсутствием сигнала в правом канале.

#### 2 ####

Уровень сигнала на выходе рации, предназначенный для наушника/гарнитуры, явно гораздо выше, чем уровень сигнала, ожидающийся звуковой картой. Как следствие, нужно очень аккуратно регулировать громкость выхода рации, так как "зашкалить" ничего не стоит - это может привести как минимум к искажениям звука, как максимум к повреждению звуковой карты.

**Важно:** во избежание повреждения звуковой карты перед подключением громкость выставить на минимум, потом после подключения перевести рацию в режим приёма (заблокировав шумодав или начав передачу с другой рации) и постепенно добавлять громкость, контролируя уровень сигнала.

#### 3 ####

Регулировку уровня записи можно прозводить как регулятором громкости рации, так и регулятором уровня записи с конкретного входа (в тесте использовался вход Rear Pink In) в штатном микшере.

**Важно:** для регулировки микшер должен находиться в режиме записи: Options - Properties - Mixer Device - Realtek HD Audio **Input** - Recording - убедиться, что нужный канал (Mic Volume или Line Volume) включён - OK, поставить галочку Select у нужного канала.

#### 4 ####

Вывод входящего сигнала на колонки (мониторинг) осуществляется средствами штатного микшера.

**Важно:** для регулировки громкости воспроизведения сигнала в колонках микшер должен находиться в режиме воспроизведения: Options - Properties - Mixer Device - Realtek HD Audio **Output** - Playback - убедиться, что нужный канал (в случае данного теста - Rear Pink In) включён - OK, снять галочку Mute у нужного канала.

**Важно:** таким способом регулируется именно громкость воспроизведения звука со входа через колонки, уровень записи от этого не меняется.

**Важно:** не включайте воспроизведение через колонки, если тестируете систему путём передачи с одной рации на другую, находясь перед компьютером. Передающая рация может ловить звук из колонок, что приводит к "заводке" (характерному писку) и искажению сигнала.

#### 5 ####

Ввиду предыдущего, функцию вывода записываемого звука в колонки в программе нужно либо убрать вовсе, либо как минимум сделать опциональной, в частности для избежания "двоящегося звука" в колонках - так как программный мониторинг идёт с задержкой.

#### 6 ####

Нужно дополнительное тестирование на компьютере с простой звуковой картой и/или на ноутбуке - там возможны ограничения дуплексного режима и штатный микшер может не справляться с регулированием записи и воспроизведения одновременно. В этом случае программный мониторинг может оказаться полезен.

#### 7 ####

Посторонние звуки, воспроизводящиеся на компьютере хоть приглушённо и искаженно, но всё же попадают в записываемые файлы. Как следствие, слушать музыку на компьютере, ведущем запись, противопоказано. Различные системные звуки также следует отключать.

#### 8 ####

Программа создаёт файлы с именами вида `2010-04-28_02.49.48.wav`.

Предлагаю использовать вместо точек дефисы - `2010-04-28_02-49-48.wav`, дабы минимизировать возможные проблемы из-за "множественных расширений" имени файла.

Как вариант - вообще убрать разделители в дате и времени, дабы укоротить имя файла: `20100428-024948.wav`