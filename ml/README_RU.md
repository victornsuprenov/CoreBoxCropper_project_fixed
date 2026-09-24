# ML: проверка и первое обучение

## Сейчас — только проверка DTM

На этом этапе **не запускайте обучение**. Сначала проверяем разметку.

В PowerShell из корня проекта:

```powershell
python .\ml\validate_dataset.py D:\DTM
```

Если `DTM` лежит в другом месте, укажите настоящий путь.

Скрипт создаст:

```text
DTM\_validation\validation_report.txt
DTM\_validation\validation_report.json
```

В ответ пришлите мне **содержимое `validation_report.txt`**. Архив фотографий для этого первого этапа не нужен.

Код возврата:
- `0` — датасет формально готов;
- `2` — найдены ошибки.

Проверяются:
- соответствие image/label;
- корректность YOLO-координат;
- ровно один target_corebox на фото;
- класс `0`;
- выход bbox за границы;
- битые изображения;
- дубликаты изображений;
- структура `dataset.yaml`, `classes.txt`, `annotations.json`;
- статистика ориентаций и размеров bbox.

## После нашего одобрения

Подготовка воспроизводимого train/val:

```powershell
python .\ml\prepare_training.py D:\DTM
```

По умолчанию 80%/20%, seed фиксирован. Это создаст `DTM\yolo_dataset`.

## Обучение

Пока не запускайте. После проверки мы подберём параметры и установим отдельные ML-зависимости на машине разработчика.

```powershell
python .\ml\train_yolo.py D:\DTM\yolo_dataset\dataset.yaml
```

**Важно:** `yolo11n.pt` — только стартовые предобученные веса. Это ещё не наша модель. Наша модель появится после обучения на DTM. В конечный корпоративный EXE попадёт только обученный вес/ONNX, а не весь тренировочный стек.

## Обучение keypoint-модели V3

Редактор разметки CoreBoxCropper сохраняет четыре нормализованные точки в
`annotations.json` в порядке `ЛВ, ПВ, ПН, ЛН`. Для V3 используется Ultralytics
Pose с одной целью и четырьмя keypoints. Точки имеют visibility `2`.

### 1. Подготовить исходный датасет

Структура каталога должна быть такой:

```text
D:\DTM_KEYPOINT_V1\
	images\
		photo_001.jpg
	annotations.json
```

Разметить все фотографии кнопкой **«Разметка данных для ML»**. После этого
создать pose-датасет:

```powershell
python .\ml\prepare_keypoint_training.py D:\DTM_KEYPOINT_V1
```

По умолчанию результат появится в `D:\DTM_KEYPOINT_V1\keypoint_dataset`.
Для воспроизводимого другого split можно указать `--seed` и `--val-ratio`.

### 2. Проверить pose-датасет

```powershell
python .\ml\validate_keypoint_dataset.py D:\DTM_KEYPOINT_V1\keypoint_dataset
```

Ожидаемый результат: `RESULT: OK`. Скрипт проверяет битые изображения,
соответствие labels, ровно 17 значений в каждой строке и нахождение bbox/точек
в диапазоне `[0, 1]`.

### 3. Установить зависимости для обучения

На машине обучения:

```powershell
python -m pip install --upgrade ultralytics
```

Скачайте или положите рядом с проектом стартовые веса `yolo11n-pose.pt`.
Это базовые веса, а не готовая модель проекта.

### 4. Обучить модель

Для видеокарты NVIDIA:

```powershell
python .\ml\train_keypoints.py D:\DTM_KEYPOINT_V1\keypoint_dataset\dataset.yaml --weights .\yolo11n-pose.pt --epochs 100 --imgsz 960 --batch 8 --device 0 --export-onnx
```

Для CPU:

```powershell
python .\ml\train_keypoints.py D:\DTM_KEYPOINT_V1\keypoint_dataset\dataset.yaml --weights .\yolo11n-pose.pt --epochs 100 --imgsz 640 --batch 2 --device cpu --export-onnx
```

Результаты будут в:

```text
runs\pose\corebox_keypoints_v3\weights\best.pt
runs\pose\corebox_keypoints_v3\weights\best.onnx
```

Перед использованием в приложении проверьте `best.pt` на отдельном наборе,
который не входил в train/val. Для production нужен именно ONNX-файл; `.pt`
оставляйте только для дальнейшего обучения и диагностики.

### 5. Требования к качеству

- минимум 100-200 разнообразных размеченных фотографий для первого прототипа;
- в train и val должны попасть разные сцены, а не соседние кадры одной серии;
- все точки должны быть в одном порядке: `ЛВ, ПВ, ПН, ЛН`;
- не включать в разметку изображения, где ящик закрыт или видны не все четыре угла;
- отдельно собрать hard cases: блики, поворот, тени, частичное перекрытие и
	несколько похожих ящиков.

Текущий production detector не переключается на V3 автоматически. После
обучения нужно прогнать regression harness на 118 фото, сравнить crop с
`bbox_crops`, а затем подключить `best.onnx` через V3 inference adapter.

После отдельной проверки ONNX V3 включается автоматически для GUI и CLI, если
в проекте существует актуальный файл `runs/pose/corebox_keypoints_v2/weights/best.onnx`.
При необходимости его можно отключить настройкой:

```json
{
	"v3_enabled": false,
	"v3_keypoint_model": "C:/path/to/runs/pose/corebox_keypoints_v2/weights/best.onnx"
}
```

При провале quality gate приложение автоматически продолжает через текущий
YOLO/legacy fallback.

### Regression V3 на 118 фотографиях

Проверка V3 выполняется отдельным harness и не смешивается с bbox regression:

```powershell
python .\ml\regression_v3.py `
	"C:\Users\victornsuprenov\Pictures\тесты\ВЫБОРКА\Левые фотки_2" `
	--v3 ".\runs\pose\corebox_keypoints_v2\weights\best.onnx" `
	--baseline ".\runs\detect\ml\runs\corebox_v2-2\weights\best.pt"
```

Успешным считается только кадр, где одновременно прошли quality gate,
получены 4 keypoints и создан perspective crop. Результаты сохраняются в
`_regression_v3\predictions`, `_regression_v3\crops`, `_regression_v3\debug` и
`_regression_v3\report.json`.

### Как добавить новые фотографии в текущий датасет

Исходный датасет — это каталог `DTM_POINT`, а не папка
`DTM_POINT\keypoint_dataset`. Для каждого нового изображения:

1. Скопируйте оригинал в `DTM_POINT\images`.
2. Откройте в приложении разметку ML и укажите папку датасета
	`C:\Users\victornsuprenov\Pictures\тесты\DTM_POINT`.
3. Выберите новые изображения. Нажмите **«Очистить точки»**, если нужна
	ручная разметка с нуля, затем щёлкните `P1`, `P2`, `P3`, `P4` по порядку.
	Для уже найденной рамки можно перетащить всю рамку внутри области.
4. Нажмите **«Сохранить текущую»**. Редактор добавит запись в
	`annotations.json` и совместимый bbox в `labels`.
5. После разметки всех новых фото пересоберите pose-датасет:

```powershell
python .\ml\prepare_keypoint_training.py "C:\Users\victornsuprenov\Pictures\тесты\DTM_POINT"
python .\ml\validate_keypoint_dataset.py "C:\Users\victornsuprenov\Pictures\тесты\DTM_POINT\keypoint_dataset"
```

Скрипт подготовки очищает только `keypoint_dataset\images\train|val` и
`labels\train|val`, затем создаёт новый воспроизводимый split. Исходные фото,
`annotations.json` и исходные `labels` не удаляются.
