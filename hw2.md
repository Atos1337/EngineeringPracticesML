# Установка пакетного менеджера

Кажется, что pip идёт вместе с питоном (я не устанавливал отдельно), но вот так можно установить: 
```shell
sudo apt install python3-pip
```

# Развёртывание окружения
```shell
pip install -r requirements/prod.txt
```
или в виртуальном:
```shell
python -m venv venv
source venv/bin/activate
pip install -r requirements/prod.txt
```
# Сборка пакета
```shell
python -m pip install build
python -m build
```
# Ссылка на PyPI
https://pypi.org/project/vsamoilov-eng-prac/1.0.1/
# Установка из PyPI
```shell
pip install vsamoilov-eng-prac==1.0.1
```
