# Системный промпт: Visual Odometry с Loop Closure SLAM

## Контекст проекта

Вы работаете над проектом стерео-визуальной одометрии на C++/OpenCV с обнаружением замыкания петли (loop closure) для реализации SLAM. Проект находится в `/home/selbizo/CV/StabAndSLAM/visual_odom`.

## Архитектура системы

### Основные компоненты

- **`src/main.cpp`** - Главный цикл выполнения, координация VO и loop closure
- **`src/loopclosure.cpp/h`** - Модуль обнаружения и коррекции замыкания петли
- **`src/visualOdometry.cpp/h`** - Оценка визуальной одометрии (отслеживание позы)
- **`src/feature.cpp/h`** - Детекция (FAST), отслеживание (KLT) и бакетирование фич
- **`src/utils.cpp`** - Визуализация траектории, вывод координат
- **`src/videoStabilization/`** - Опциональный модуль стабилизации видео
- **`src/Frame.cpp/h`** - Структура данных кадра
- **`src/rgbd_standalone.cpp/h`** - Поддержка RGBD камеры Intel

### Зависимости

- OpenCV 3.0+ (с поддержкой CUDA по опции `USE_CUDA`)
- C++17
- Датасет KITTI odometry (или RGBD камера)

## Ключевые структуры данных

### KeyFrame (loopclosure.h:19-35)

```cpp
struct KeyFrame {
    int id;
    cv::Mat image;           // Полное изображение
    cv::Mat descriptor;      // Deep learning фичи (1000-мерные)
    cv::Mat orb_descriptor;  // ORB дескрипторы (256-мерные)
    std::vector<int> desc_feat_indx;
    cv::Mat rotation, translation;  // Преобразование кадр-к-кадру
    std::vector<cv::Point2f> keypoints;
    std::vector<cv::Point3f> points3D;
    bool is_keyframe;
    cv::Mat full_pose;       // 4x4 накопленная мировая поза
};
```

## Конвейер loop closure (loopclosure.cpp)

### 1. Детекция ключевых кадров (isKeyframe)

- Проверяет, если `points3D.rows >= min_keypoints` (по умолчанию: 50)
- Добавляется как ключевой кадр только если превышает порог ИЛИ `force_keyframe=true`

### 2. Извлечение фич

- **Deep фичи**: Модель MobileNetV2 ONNX (вход 224x224, L2-нормализованный вывод)
- **Дескрипторы точек**: ORB (200 фич, размер keypoint 7.0f)
- **Сопоставление дескрипторов**: BruteForce-Hamming с фильтрацией по расстоянию

### 3. Обнаружение замыкания (detectLoop)

- Вычисляется сходство deep фич между текущим и всеми предыдущими ключевыми кадрами
- Порог: сходство > `strong_threshold` (по умолчанию: 0.82f)
- Валидация через ORB дескрипторы (> `min_match_count` = 20)
- Оценка коррекции позы через PnP (solvePnPRansac)

**КРИТИЧНО**: Конвертировать результат PnP из локальной системы координат кандидата в мировую коррекцию:

- `T_pnp`: Текущая поза в локальных координатах кандидата
- `T_current_estimated_world = candidate_kf.full_pose * T_pnp.inv()` - где кадр ДОЛЖЕН быть в мировых координатах
- `T_current_naive_world = current_kf.full_pose` - одометрия (с накопленным дрейфом)
- `T_delta = T_current_estimated_world * T_current_naive_world.inv()` - правильная коррекция

### 4. Применение коррекции

- Применяется интерполированная коррекция к ключевым кадрам между кандидатом и текущим
- Кадры до/включая кандидата остаются нетронутыми (доверенный якорь)
- Коррекция интерполируется от 0 (кандидат) до полного дельта (текущий кадр)

### 5. Коррекция траектории (utils.cpp:42-92)

- Поддерживает отдельный буфер траектории `corrected_poses_`
- Применяет интерполированную коррекцию к точкам траектории
- Вызывается через `setLoopClosureCorrection()` из main.cpp

## Системы координат и преобразования

### Критическое различие

- `rotation`/`translation` из `trackingFrame2Frame()`: **инкрементное** преобразование кадр-к-кадру
- `frame_pose`: **накопленная** мировая поза (4x4 матрица, накопленная от начала)
- `full_pose` в KeyFrame: ДОЛЖЕН быть `frame_pose` (мировая поза), НЕ инкрементная

### Правильный поток в main.cpp (строка 591-594)

```cpp
loopClosure.addFrame(frame_id, imageLeft_t1, imageRight_t1,
                    pointsLeft_t0, pointsRight_t0,
                    rotation, translation, points3D_t0, frame_pose, true);
```

## Текущее состояние реализации

### Работает:
- ✅ Оценка VO кадр-к-кадру через алгоритм Nister's 5-point + solvePnPRansac
- ✅ Deep feature loop detection с MobileNetV2
- ✅ Сопоставление ORB дескрипторов для валидации loop closure
- ✅ Интерполированная коррекция для ключевых кадров между loop candidate и текущим
- ✅ Логирование координат траектории в `trajectory_coordinates.txt`

## Сборка и запуск

### Сборка:
```bash
cd /home/selbizo/CV/StabAndSLAM/visual_odom
mkdir -p build && cd build
cmake ..
make -j4
```

### Запуск:
```bash
./run /home/selbizo/CV/dataset/sequences/00/ ../calibration/kitti00.yaml
```

### Параметры (main.cpp:65-76):
- Путь к датасету: `filepath` (захардкожен на KITTI sequence 00)
- Калибровка: `strSettingPath` (kitti00.yaml)
- Пропуск кадров: настраивается клавишами 'w'/'s' во время выполнения
- Локальный потолок loop: `local_loop_ceiling = 4449` (для повторяющихся последовательностей)

## Параметры и пороги (loopclosure.cpp:7-37)

```cpp
min_keypoints_ = 50
weak_threshold_ = 0.8f
strong_threshold_ = 0.82f
max_weak_candidates_ = 5
min_match_count_ = 20
max_pose_distance_between_loop_keyframes_ = 50.0 градусов
max_pose_differnece_between_old_new_ = 10.0
```

## Известные проблемы и ограничения

- Нет оптимизации pose-graph (g2o/Sophus) - используется упрощенная линейная интерполяция
- Нет viewer/debug визуализации для ключевых кадров
- Логирование координат траектории может быть неполным (см. проблему буфера траектории выше)
- Ускорение CUDA для KLT отслеживания присутствует, но loop closure только CPU

## Важные замечания

- **НЕ** изменяйте архитектуру или добавляйте новые библиотеки (g2o, Sophus, viewer)
- **НЕ** изменяйте логику стабилизации видео, если не обязательно
- **НЕ** меняйте алгоритм детекции/отслеживания фич (FAST + KLT)

## Этап развития: от debug к настройке параметров ключевых кадров

Loop closure уже срабатывает на нужных парах кадров (130↔1576, 131↔1577, ..., 182↔1623).
Задача — перейти от жёсткой привязки к априорной информации о парах кадров к **адаптивной
добавке ключевых кадров на основе настраиваемых параметров**.

### Подход к настройке ключевых кадров (keyframe selection)

1. **Порог качества трекинга** — добавлять ключевой кадр не только по количеству 3D-точек,
   но и по доле успешных KLT-треков (`good_tracks / total_tracks`), качеству ре-проекции,
   или средней длине трека.

2. **Дистанционный порог** — добавлять ключевой кадр, если пройденное расстояние между
   последними ключевыми кадрами превышает `min_keyframe_distance` (в метрах).

3. **Временной/кадровый интервал** — минимальное число кадров между ключевыми кадрами
   `min_frames_between_keyframes`.

4. **Динамический порог 3D-точек** — вместо фиксированного `min_keypoints_ = 50` использовать
   адаптивный порог, зависящий от средней освещённости/контраста сцены.

5. **Регулярная диспансация (thinning)** — если ключевых кадров слишком много, применять
   downsample по дистанции или по времени.

### Переменные конфигурации (вынести в loopclosure.h как публичные параметры)

```cpp
// Пороги для добавления ключевых кадров
float min_keypoints_ = 50;              // Мин. количество 3D-точек
float min_tracking_quality_ = 0.3f;     // Мин. доля хороших треков (0..1)
float min_keyframe_distance_ = 0.5f;    // Мин. дистанция между КФ (метры)
int min_frames_between_keyframes_ = 2;  // Мин. кадров между КФ

// Пороги loop closure
float weak_threshold_ = 0.8f;
float strong_threshold_ = 0.82f;
int max_weak_candidates_ = 5;
int min_match_count_ = 20;
float max_pose_distance_between_loop_keyframes_ = 50.0f; // градусы
float max_pose_difference_between_old_new_ = 10.0f;     // градусы
```

### Процесс отладки

- Отлаживать на **паре 130↔1576** как валид-сете.
- Логировать: **момент добавления каждого ключевого кадра**, количество 3D-точек,
  качество трекинга, дистанцию от предыдущего КФ.
- Подбирать параметры так, чтобы КФ покрывали всю петлю равномерно, но не избыточно.
- Исключить hardcoded reference to specific frame IDs — вся логика на параметрах.

## Файлы проекта

```
visual_odom/
├── CMakeLists.txt
├── README.md
├── LOOP_CLOSURE_FIX_INSTRUCTIONS.md
├── .gitignore
├── calibration/
│   ├── kitti00.yaml
│   ├── zed.yaml
│   └── rgbd.yaml
├── src/
│   ├── main.cpp
│   ├── loopclosure.cpp
│   ├── loopclosure.h
│   ├── visualOdometry.cpp
│   ├── visualOdometry.h
│   ├── feature.cpp
│   ├── feature.h
│   ├── utils.cpp
│   ├── utils.h
│   ├── Frame.cpp
│   ├── Frame.h
│   ├── bucket.cpp
│   ├── bucket.h
│   ├── rgbd_standalone.cpp
│   ├── rgbd_standalone.h
│   ├── camera_object.h
│   └── videoStabilization/
│       ├── basicFunctions.cpp/h
│       ├── stabilizationFunctions.cpp/h
│       ├── wienerFilter.cpp/h
│       ├── ConfigVideoStab.h
│       └── basicStructs.h
└── build/
```