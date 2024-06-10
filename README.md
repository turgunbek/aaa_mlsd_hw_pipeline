# aaa_mlsd_hw_pipeline
Домашнее задание по MLSD "Вычислительные ресурсы"

Запуск производится из командной строки для случаев:

-**baseline** (LFM модель с параметрами из заданного файла):
```console
python main.py
```
-**tuning_baseline** (LFM модель с перебором параметров и возможностью выбора оптимизатора (по умолчанию стоит Adam)):
```console
python main.py --run_name tuning_baseline --model_name tuning_baseline
```
-**best_ALS** (ALS модель с параметрами, которые дают лучший скор по предыдущей домашке):
```console
python main.py --run_name best_ALS --model_name best_ALS
```

Докер образ собран и запушен в докерхаб: https://hub.docker.com/repository/docker/omurkanov/hw_pipelines
