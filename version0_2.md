Ante Scriptum: с ключом --message при коммите пока не разобрался :)

# Что появилось/изменилось в версии 0.2 (9 ревизия) #
  * Самое главное: добавилось асинхронное управление. Реализовано запуском в самом начале бесконечного цикла  запроса raw\_input в параллельном треде.
  * За счет этого появилась возможность:
    1. аккуратно выйти из программ без потери данных
    1. сказать выйти после заврешения текущего трека
    1. изменить уровень порога сигнала
    1. узнать среднее значение сигнала за последнюю секунду.
    1. включать-выключать коммутацию входа с выходом
  * В качестве рабочей библиотеки пока оставил PyAudio. PyMedia для сборки запросил gcc, его мне было ставить лениво ну итд.
  * Устранена лишня тишина в конце трека
  * Впрочем, добавлен дополнительный параметр на fadeout, потому что если отрезать всю тишину, то глатаются окончания.
  * седлан шаг к искользованию конфиг-файлов. В смысле, все параметры уже кучкуются в начале кода.