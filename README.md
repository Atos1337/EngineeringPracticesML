# EngineeringPracticesML

To make experiment, run in shell:
```shell
python3 decision_tree/main.py
```

# HW 3

* Форматеры: `black` и `isort`
* Линтеры: `flake8`
* Плагины:
  * `flake8-quotes`
  * `flake8-comprehensions`
  * `flake8-return`
  * `flake8-literal`
  * `flake8-variables-names`

# HW 5

Использую DVC.

## DAG
```bash
              +--------------+
              | prepare_data |
              +--------------+
           ***        *      ****
        ***           *          ***
      **              *             ***
+-----+               *                **
| fit |*              *                 *
+-----+ ***           *                 *
    *      ***        *                 *
    *         ***     *                 *
    *            **   *                 *
    **          +---------+            **
      ***       | predict |         ***
         ***    +---------+      ***
            **        *      ****
              ***     *   ***
                 **   * **
                 +-------+
                 | plots |
                 +-------+
+--------------------+
| plots/tree.png.dvc |
+--------------------+
+-------------------------+
| plots/roc_curve.png.dvc |
+-------------------------+
+------------------+
| plots/2d.png.dvc |
+------------------+
```

С помощью команды `dvc dag`

## CLI и запуск пайплайна
`dvc repro`
