#!/bin/bash

################################################################################
# VADTree Automatic Execution Script
# This script automates the 7-step workflow described in the README.md
################################################################################

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_step() {
    echo -e "${BLUE}===================================================${NC}"
    echo -e "${GREEN}$1${NC}"
    echo -e "${BLUE}===================================================${NC}"
}

print_warning() {
    echo -e "${YELLOW}WARNING: $1${NC}"
}

print_error() {
    echo -e "${RED}ERROR: $1${NC}"
}

################################################################################
# Configuration Variables - MODIFY THESE PATHS ACCORDING TO YOUR SETUP
################################################################################

# Dataset selection (ucf_crime, xd_violence, msad)
DATASET="ucf_crime"

# Video directory path
VIDEO_DIR="/path/to/UCF_CRIME_TEST_VIDEO_DIR"

# Model paths
GEBD_MODEL_WEIGHT="/path/to/GEBD_MODEL_WEIGHT"
GEBD_MODEL_CONFIG="/path/to/MODEL_CONFIG"
VLM_MODEL_DIR="/path/to/VLM_MODEL_DIR"
LLM_MODEL_DIR="/path/to/LLM_MODEL_DIR"

# HGTree parameters
THRESHOLD="kmeans"
GAMMA="0.4"

# Correlation parameters
BETA="0.2"

# Conda environments (modify if your environment names are different)
GEBD_ENV="EfficientGEBD"
VADTREE_ENV="VADTree"
LLAVA_ENV="llava"

# Script paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Dataset-specific paths
if [ "$DATASET" == "ucf_crime" ]; then
    DATASET_NAME="UCF_Crime_test"
elif [ "$DATASET" == "xd_violence" ]; then
    DATASET_NAME="XD_Violence_test"
elif [ "$DATASET" == "msad" ]; then
    DATASET_NAME="MSAD_test"
else
    print_error "Unknown dataset: $DATASET"
    exit 1
fi

# Result paths (will be generated during execution)
GEBD_OUTPUT_DIR="$SCRIPT_DIR/result/${DATASET_NAME}/EGEBD_x2x3x4_r50_eff_split_out_th0.5"
HGTREE_OUTPUT_DIR="$SCRIPT_DIR/result/${DATASET_NAME}/EGEBD_x2x3x4_r50_eff_split_out_th0.5_peak_dfs_${THRESHOLD}_1_${GAMMA}"

################################################################################
# Validation
################################################################################

print_step "Validating configuration..."

if [ "$VIDEO_DIR" == "/path/to/UCF_CRIME_TEST_VIDEO_DIR" ]; then
    print_warning "VIDEO_DIR is not configured. Please edit this script and set the correct path."
    print_warning "You can skip steps that require video data (1, 3, 5) by setting SKIP_VIDEO_STEPS=true"
fi

if [ "$GEBD_MODEL_WEIGHT" == "/path/to/GEBD_MODEL_WEIGHT" ]; then
    print_warning "GEBD_MODEL_WEIGHT is not configured. Step 1 will be skipped."
fi

if [ "$VLM_MODEL_DIR" == "/path/to/VLM_MODEL_DIR" ]; then
    print_warning "VLM_MODEL_DIR is not configured. Step 3 will be skipped."
fi

if [ "$LLM_MODEL_DIR" == "/path/to/LLM_MODEL_DIR" ]; then
    print_warning "LLM_MODEL_DIR is not configured. Step 4 will be skipped."
fi

# Option to skip steps that require video data
SKIP_VIDEO_STEPS=${SKIP_VIDEO_STEPS:-false}

echo "Configuration:"
echo "  Dataset: $DATASET"
echo "  Video Directory: $VIDEO_DIR"
echo "  Skip Video Steps: $SKIP_VIDEO_STEPS"
echo ""

################################################################################
# Step 1: GEBD boundary extraction
################################################################################

if [ "$SKIP_VIDEO_STEPS" == "false" ] && [ "$GEBD_MODEL_WEIGHT" != "/path/to/GEBD_MODEL_WEIGHT" ]; then
    print_step "Step 1: GEBD boundary extraction"
    
    cd "$SCRIPT_DIR/EfficientGEBD"
    
    eval "$(conda shell.bash hook)"
    conda activate $GEBD_ENV
    
    python GEBD_split100.py \
        --video_dir "$VIDEO_DIR" \
        --resume "$GEBD_MODEL_WEIGHT" \
        --config-file "$GEBD_MODEL_CONFIG"
    
    print_step "Step 1 completed successfully!"
else
    print_warning "Skipping Step 1: GEBD boundary extraction"
    print_warning "Using existing results from: $GEBD_OUTPUT_DIR"
fi

################################################################################
# Step 2: Build HGTree
################################################################################

print_step "Step 2: Build HGTree"

cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate $VADTREE_ENV

python HGTree_generation.py \
    --json_path "$GEBD_OUTPUT_DIR/pred_scenes_th0.5.json" \
    --threshold "$THRESHOLD" \
    --gamma "$GAMMA"

print_step "Step 2 completed successfully!"

################################################################################
# Step 3: Node-wise VLM captioning (Coarse and Fine)
################################################################################

if [ "$SKIP_VIDEO_STEPS" == "false" ] && [ "$VLM_MODEL_DIR" != "/path/to/VLM_MODEL_DIR" ]; then
    print_step "Step 3: Node-wise VLM captioning (Coarse and Fine)"
    
    cd "$SCRIPT_DIR/LLaVA-NeXT"
    
    eval "$(conda shell.bash hook)"
    conda activate $LLAVA_ENV
    
    # Coarse-grained captioning
    echo "Processing coarse-grained nodes..."
    python infer_VAD.py \
        --pretrained "$VLM_MODEL_DIR" \
        --video_root "$VIDEO_DIR" \
        --json_path "$HGTREE_OUTPUT_DIR/dfs_coarse_scenes.json"
    
    # Fine-grained captioning
    echo "Processing fine-grained nodes..."
    python infer_VAD.py \
        --pretrained "$VLM_MODEL_DIR" \
        --video_root "$VIDEO_DIR" \
        --json_path "$HGTREE_OUTPUT_DIR/dfs_fine_scenes.json"
    
    print_step "Step 3 completed successfully!"
else
    print_warning "Skipping Step 3: Node-wise VLM captioning"
    print_warning "Using existing results from: $HGTREE_OUTPUT_DIR/LLaVA-Video-7B-Qwen2_*"
fi

################################################################################
# Step 4: Node-wise LLM reasoning (Coarse and Fine)
################################################################################

if [ "$LLM_MODEL_DIR" != "/path/to/LLM_MODEL_DIR" ]; then
    print_step "Step 4: Node-wise LLM reasoning (Coarse and Fine)"
    
    cd "$SCRIPT_DIR/DeepSeek-R1"
    
    eval "$(conda shell.bash hook)"
    conda activate $LLAVA_ENV  # Uses same environment as LLaVA
    
    # Find the VLM output directories
    COARSE_VLM_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_coarse/maxf64_*_Here is a .json" | head -1)
    FINE_VLM_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_fine/maxf64_*_Here is a .json" | head -1)
    
    if [ -z "$COARSE_VLM_JSON" ] || [ -z "$FINE_VLM_JSON" ]; then
        print_error "Could not find VLM output JSON files. Please run Step 3 first or check the paths."
        exit 1
    fi
    
    # Coarse-grained reasoning
    echo "Processing coarse-grained reasoning..."
    python deepseek_batch_infer.py \
        --video_root "$VIDEO_DIR" \
        --ckpt_dir "$LLM_MODEL_DIR" \
        --input_json "$COARSE_VLM_JSON"
    
    # Fine-grained reasoning
    echo "Processing fine-grained reasoning..."
    python deepseek_batch_infer.py \
        --video_root "$VIDEO_DIR" \
        --ckpt_dir "$LLM_MODEL_DIR" \
        --input_json "$FINE_VLM_JSON"
    
    print_step "Step 4 completed successfully!"
else
    print_warning "Skipping Step 4: Node-wise LLM reasoning"
    print_warning "Using existing results from: $HGTREE_OUTPUT_DIR/*/DeepSeek-R1-Distill-Qwen-14B_think/"
fi

################################################################################
# Step 5: Feature similarity (Coarse and Fine)
################################################################################

if [ "$SKIP_VIDEO_STEPS" == "false" ]; then
    print_step "Step 5: Feature similarity (Coarse and Fine)"
    
    cd "$SCRIPT_DIR/ImageBind"
    
    eval "$(conda shell.bash hook)"
    conda activate $VADTREE_ENV
    
    # Find the VLM output directories
    COARSE_VLM_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_coarse/maxf64_*_Here is a .json" | head -1)
    FINE_VLM_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_fine/maxf64_*_Here is a .json" | head -1)
    
    if [ -z "$COARSE_VLM_JSON" ] || [ -z "$FINE_VLM_JSON" ]; then
        print_error "Could not find VLM output JSON files. Please run Step 3 first or check the paths."
        exit 1
    fi
    
    # Coarse-grained similarity
    echo "Processing coarse-grained similarity..."
    python imagebind_sim.py \
        --video_summary_json "$COARSE_VLM_JSON" \
        --video_root "$VIDEO_DIR"
    
    # Fine-grained similarity
    echo "Processing fine-grained similarity..."
    python imagebind_sim.py \
        --video_summary_json "$FINE_VLM_JSON" \
        --video_root "$VIDEO_DIR"
    
    print_step "Step 5 completed successfully!"
else
    print_warning "Skipping Step 5: Feature similarity"
    print_warning "Using existing results from: $HGTREE_OUTPUT_DIR/*/sim_maxf64_*_Here is a .pkl"
fi

################################################################################
# Step 6: Intra-cluster refinement & eval (Coarse and Fine)
################################################################################

print_step "Step 6: Intra-cluster refinement & eval (Coarse and Fine)"

cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate $VADTREE_ENV

# Find the LLM reasoning output files
COARSE_SCORE_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_coarse/DeepSeek-R1-Distill-Qwen-14B_think/maxf64_*_Here is a .json" | head -1)
FINE_SCORE_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_fine/DeepSeek-R1-Distill-Qwen-14B_think/maxf64_*_Here is a .json" | head -1)

if [ -z "$COARSE_SCORE_JSON" ] || [ -z "$FINE_SCORE_JSON" ]; then
    print_error "Could not find LLM reasoning output JSON files. Please run Step 4 first or check the paths."
    exit 1
fi

# Coarse-grained refinement
echo "Processing coarse-grained refinement..."
python refinement_eval.py \
    --score_json "$COARSE_SCORE_JSON"

# Fine-grained refinement
echo "Processing fine-grained refinement..."
python refinement_eval.py \
    --score_json "$FINE_SCORE_JSON"

print_step "Step 6 completed successfully!"

################################################################################
# Step 7: Inter-cluster correlation & eval
################################################################################

print_step "Step 7: Inter-cluster correlation & eval"

cd "$SCRIPT_DIR"

eval "$(conda shell.bash hook)"
conda activate $VADTREE_ENV

# Find the refined score files
COARSE_REFINED_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_coarse/DeepSeek-R1-Distill-Qwen-14B_think_VxV10_nn10_tao0.1/refine_maxf64_*_Here is a .json" | head -1)
FINE_REFINED_JSON=$(find "$HGTREE_OUTPUT_DIR" -path "*/LLaVA-Video-7B-Qwen2_*_fine/DeepSeek-R1-Distill-Qwen-14B_think_VxV10_nn10_tao0.1/refine_maxf64_*_Here is a .json" | head -1)

if [ -z "$COARSE_REFINED_JSON" ] || [ -z "$FINE_REFINED_JSON" ]; then
    print_error "Could not find refined score JSON files. Please run Step 6 first or check the paths."
    exit 1
fi

python correlation_eval.py \
    --coarse_scores_json "$COARSE_REFINED_JSON" \
    --fine_scores_json "$FINE_REFINED_JSON" \
    --beta "$BETA"

print_step "Step 7 completed successfully!"

################################################################################
# Final Summary
################################################################################

print_step "VADTree pipeline completed successfully!"

echo -e "${GREEN}Results are saved in:${NC}"
echo "  $HGTREE_OUTPUT_DIR"
echo ""
echo -e "${GREEN}Final output:${NC}"
FINAL_OUTPUT=$(find "$HGTREE_OUTPUT_DIR" -path "*/ense_refine_maxf64_*_Here is a .json" | head -1)
if [ -n "$FINAL_OUTPUT" ]; then
    echo "  $FINAL_OUTPUT"
else
    echo "  Check the directory above for the final results"
fi
echo ""
echo -e "${YELLOW}Note: If you skipped any steps due to missing configurations,${NC}"
echo -e "${YELLOW}please update the configuration variables at the top of this script.${NC}"
