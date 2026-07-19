#!/bin/bash
# ============================================================================
# Диагностика и автоматическое исправление сборки StereoVision-SLAM
# Запуск: chmod +x diagnose_and_fix.sh && ./diagnose_and_fix.sh
# ============================================================================

set -e  # остановка при первой ошибке

PROJECT_DIR="$(pwd)"  # текущая папка (корень проекта)
BUILD_DIR="$PROJECT_DIR/build"
CMAKE_EXE="/opt/cmake-3.22.1/bin/cmake"  # используем ваш cmake

# Цветной вывод
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}   Диагностика окружения StereoVision-SLAM   ${NC}"
echo -e "${GREEN}==============================================${NC}"

# ----------------------------------------------------------------------------
# 1. Проверка основных инструментов
# ----------------------------------------------------------------------------
echo -e "\n${YELLOW}1. Проверка компиляторов и CMake...${NC}"
if ! command -v g++ &> /dev/null; then
    echo -e "${RED}Ошибка: g++ не найден. Установи build-essential.${NC}"
    exit 1
fi
if ! command -v $CMAKE_EXE &> /dev/null; then
    echo -e "${RED}Ошибка: CMake не найден по пути $CMAKE_EXE. Укажи правильный путь.${NC}"
    exit 1
fi
echo -e "${GREEN}OK: g++ и CMake найдены.${NC}"

# ----------------------------------------------------------------------------
# 2. Проверка зависимостей (библиотеки)
# ----------------------------------------------------------------------------
echo -e "\n${YELLOW}2. Проверка установленных библиотек...${NC}"

# Список критических библиотек с командами проверки
declare -A DEPS=(
    ["Eigen3"]="pkg-config --modversion eigen3"
    ["Sophus"]="pkg-config --modversion Sophus 2>/dev/null || echo 'не найден (pkg-config)'"
    ["g2o"]="pkg-config --modversion g2o 2>/dev/null || echo 'не найден (pkg-config)'"
    ["OpenCV"]="pkg-config --modversion opencv4"
    ["fmt"]="pkg-config --modversion fmt"
    ["PCL"]="pkg-config --modversion pcl_common-1.15 || pkg-config --modversion pcl_common-1.12 || echo 'не найден'"
    ["CSparse"]="pkg-config --modversion suitesparse 2>/dev/null || echo 'не найден'"
    ["Flann"]="pkg-config --modversion flann"
    ["Qhull"]="pkg-config --modversion qhull_r"
    ["Boost"]="dpkg -s libboost-system-dev 2>/dev/null | grep Version | cut -d' ' -f2 || echo 'не установлен'"
)

for dep in "${!DEPS[@]}"; do
    cmd="${DEPS[$dep]}"
    version=$(eval $cmd 2>/dev/null || echo "НЕ УСТАНОВЛЕН")
    if [[ "$version" == "НЕ УСТАНОВЛЕН" || "$version" == *"не найден"* ]]; then
        echo -e "${RED}✗ $dep: $version${NC}"
    else
        echo -e "${GREEN}✓ $dep: $version${NC}"
    fi
done

# Специальная проверка для CUDA (используем 12.4)
echo -e "\n${YELLOW}Проверка CUDA 12.4...${NC}"
if [ -f /usr/local/cuda-12.4/bin/nvcc ]; then
    CUDA_VER=$(/usr/local/cuda-12.4/bin/nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
    echo -e "${GREEN}✓ CUDA: $CUDA_VER (по пути /usr/local/cuda-12.4)${NC}"
else
    echo -e "${RED}✗ CUDA 12.4 не найдена в /usr/local/cuda-12.4${NC}"
fi

# ----------------------------------------------------------------------------
# 3. Поиск путей к PCL и fmt (пригодятся для CMake)
# ----------------------------------------------------------------------------
echo -e "\n${YELLOW}3. Поиск путей к PCL и fmt...${NC}"
PCL_CMAKE_DIR=$(find /usr/local -name "PCLConfig.cmake" 2>/dev/null | head -1)
if [ -z "$PCL_CMAKE_DIR" ]; then
    PCL_CMAKE_DIR=$(find /usr -name "PCLConfig.cmake" 2>/dev/null | head -1)
fi
if [ -n "$PCL_CMAKE_DIR" ]; then
    echo -e "${GREEN}✓ PCL Config найден: $PCL_CMAKE_DIR${NC}"
else
    echo -e "${RED}✗ PCLConfig.cmake не найден. Возможно, PCL не установлен или не в /usr/local.${NC}"
fi

FMT_CMAKE_DIR=$(find /usr -name "fmt-config.cmake" 2>/dev/null | head -1)
if [ -z "$FMT_CMAKE_DIR" ]; then
    FMT_CMAKE_DIR=$(find /usr/local -name "fmt-config.cmake" 2>/dev/null | head -1)
fi
if [ -n "$FMT_CMAKE_DIR" ]; then
    echo -e "${GREEN}✓ fmt Config найден: $FMT_CMAKE_DIR${NC}"
else
    echo -e "${RED}✗ fmt-config.cmake не найден. Установи libfmt-dev.${NC}"
    exit 1
fi

# ----------------------------------------------------------------------------
# 4. Исправление CMakeLists.txt
# ----------------------------------------------------------------------------
echo -e "\n${YELLOW}4. Проверка и исправление CMakeLists.txt...${NC}"

CMAKELIST="$PROJECT_DIR/CMakeLists.txt"
if [ ! -f "$CMAKELIST" ]; then
    echo -e "${RED}Ошибка: файл $CMAKELIST не найден. Запусти скрипт из корня проекта.${NC}"
    exit 1
fi

# Создаём резервную копию
cp "$CMAKELIST" "$CMAKELIST.bak"
echo -e "${GREEN}Резервная копия создана: $CMAKELIST.bak${NC}"

# Проверяем, есть ли find_package(fmt)
if grep -q "find_package(fmt" "$CMAKELIST"; then
    echo -e "${GREEN}✓ find_package(fmt) уже присутствует.${NC}"
else
    echo -e "${YELLOW}Добавляем find_package(fmt REQUIRED) в CMakeLists.txt...${NC}"
    # Ищем строку с "pcl" или "rerun" и вставляем перед ней
    sed -i '/find_package(PCL REQUIRED)/i find_package(fmt REQUIRED)' "$CMAKELIST"
    # Альтернативный вариант: вставить после "rerun"
    # sed -i '/FetchContent_MakeAvailable(rerun_sdk)/a find_package(fmt REQUIRED)' "$CMAKELIST"
fi

# Проверяем, есть ли fmt::fmt в списке THIRD_PARTY_LIBS
if grep -q "THIRD_PARTY_LIBS.*fmt::fmt" "$CMAKELIST"; then
    echo -e "${GREEN}✓ fmt::fmt уже добавлен в THIRD_PARTY_LIBS.${NC}"
else
    echo -e "${YELLOW}Добавляем fmt::fmt в THIRD_PARTY_LIBS...${NC}"
    # Ищем строку set(THIRD_PARTY_LIBS ... ) и добавляем fmt::fmt в конец списка
    sed -i 's/set(THIRD_PARTY_LIBS \([^)]*\))/set(THIRD_PARTY_LIBS \1 fmt::fmt)/' "$CMAKELIST"
    # Также добавим ${PCL_LIBRARIES}, если его нет
    if ! grep -q "THIRD_PARTY_LIBS.*\${PCL_LIBRARIES}" "$CMAKELIST"; then
        sed -i 's/set(THIRD_PARTY_LIBS \([^)]*\))/set(THIRD_PARTY_LIBS \1 ${PCL_LIBRARIES})/' "$CMAKELIST"
        echo -e "${GREEN}✓ ${PCL_LIBRARIES} добавлен в THIRD_PARTY_LIBS.${NC}"
    fi
fi

# ----------------------------------------------------------------------------
# 5. Очистка и пересборка
# ----------------------------------------------------------------------------
echo -e "\n${YELLOW}5. Очистка папки build и пересборка...${NC}"
if [ -d "$BUILD_DIR" ]; then
    rm -rf "$BUILD_DIR"
    echo -e "${GREEN}Папка build удалена.${NC}"
fi
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo -e "${YELLOW}Запуск CMake...${NC}"
$CMAKE_EXE -DCMAKE_BUILD_TYPE=Debug ..
if [ $? -ne 0 ]; then
    echo -e "${RED}Ошибка при конфигурации CMake. Проверь вывод выше.${NC}"
    exit 1
fi

echo -e "${YELLOW}Запуск сборки (make -j$(nproc))...${NC}"
make -j$(nproc)
if [ $? -ne 0 ]; then
    echo -e "${RED}Ошибка сборки. Проверь вывод выше.${NC}"
    exit 1
fi

echo -e "${GREEN}==============================================${NC}"
echo -e "${GREEN}   Сборка успешно завершена!   ${NC}"
echo -e "${GREEN}==============================================${NC}"
echo -e "Исполняемые файлы находятся в папке ${YELLOW}$BUILD_DIR/../bin/${NC}"
echo -e "Можешь запускать: ${YELLOW}./bin/run_stereo_visual_SLAM${NC} (или run_dense_reconstruction)"