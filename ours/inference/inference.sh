#!/usr/bin/env bash

# 當任何指令回傳非 0 值（錯誤）時，立即停止腳本
set -e


# --- 1. 全域設定區 (Global Configuration) ---
EXP_NAME="UNet_baseline"
ALPHA=0.4

# 專案根目錄與資料根目錄
PROJECT_ROOT="/d/YYYeh/ToothSegmentation"
# DATA_ROOT="${PROJECT_ROOT}/data"
DATA_ROOT="data"

echo "========================================"
echo "      Starting Batch Inference          "
echo "      Experiment: $EXP_NAME             "
echo "========================================"

# --- 2. Fold 迴圈 (從 1 到 4) ---
for FOLD in 1 2 3 4; do

  # 根據 FOLD 選擇對應的病人清單
	case "$FOLD" in
		1)
			PATIENT_LIST=(
			"57969132"  # 植體
			"2188726"   # 植體
			"21800298"  # 植體
			"28643177"  # 植體
			"40657603"  # 植體
			"52730449"  # 植體
			
			"24937292"  # data_6
			"33802992"  # data_9
			"49095613"  # data_14
			"73141798"  # data_18
			)
			;;
		2)
			PATIENT_LIST=(
			"15235753"  # data_3
			"33208616"  # data_8
			"56872376"  # data_16
			)
			;;
		3)
			PATIENT_LIST=(
			"18234781"  # data_4
			"35290820"  # data_10
			)
			;;
		4)
			PATIENT_LIST=(
			"4333498"   # data_2
			"36719405"  # data_11
			"37460134"  # data_12
			)
			;;
		*)
			echo "Error: Invalid Fold $FOLD"
			exit 1
			;;
	esac

	echo ""
	echo ">>> Processing FOLD: $FOLD with ${#PATIENT_LIST[@]} patients"
	echo "----------------------------------------"

	# --- 3. 病人迴圈 (Batch Processing Loop) ---
	for PATIENT_ID in "${PATIENT_LIST[@]}"; do

		INPUT_DIR="${DATA_ROOT}/${PATIENT_ID}/dcm_to_png"
		OUTPUT_ROOT="${DATA_ROOT}/${PATIENT_ID}"
		OUTPUT_MASK="${OUTPUT_ROOT}/mask"
		OUTPUT_OVERLAY="${OUTPUT_ROOT}/overlay"

		echo "========================================"
		echo "Processing Patient: $PATIENT_ID (FOLD=$FOLD)"

		if [ ! -d "$INPUT_DIR" ]; then
			echo "Error: Directory not found: $INPUT_DIR"
			echo "Skipping..."
			continue
		fi

		mkdir -p "$OUTPUT_MASK" "$OUTPUT_OVERLAY"

		# (A) 產生 Mask (PNG + Binary NPY + Multi NPY)
		echo "[1/3] Generating Masks & Dual NPYs..."
		python -m ours.inference.inference_mask \
			--exp "$EXP_NAME" \
			--input "$INPUT_DIR" \
			--output "$OUTPUT_MASK" \
			--output_npy "$OUTPUT_ROOT" \
			--fold "$FOLD"

		# (B) 產生 Overlay (PNG)
		echo "[2/3] Generating Overlays..."
		python -m ours.inference.inference_overlay \
			--exp "$EXP_NAME" \
			--input "$INPUT_DIR" \
			--output "$OUTPUT_OVERLAY" \
			--fold "$FOLD" \
			--alpha "$ALPHA"

		# (C) 產生 Origin NPY
		echo "[3/3] Stacking Origin NPY..."
		python -m ours.inference.save_origin_npy \
			--input "$INPUT_DIR" \
			--output "$OUTPUT_ROOT"

		echo "Done: $PATIENT_ID"
		echo "========================================"
	done
done

echo ""
echo "All Folds and Patients Processed Successfully!"
