export HMS_DATA_DIR="${HMS_DATA_DIR:-$(pwd)/data}"
export HMS_THIRD_PARTY_DIR="${HMS_THIRD_PARTY_DIR:-$(pwd)/third_party}"
export OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}"
export HMS_VLM_BASE_URL="${HMS_VLM_BASE_URL:-http://localhost:7698/v1}"

python -m hms.generation \
    exp_name="barbell_lift" \
    exp_group="sketchfab_spso" \
    exp_tags="[sketchfab]" \
    seed=0 \
    reuse_exp_dir="" \
    reuse_subdirs="[affordance]" \
    interaction="a person lifting up a barbell from the ground with both hands" \
    humans_0.human_name="person 1" \
    scenes_0.obj_name="barbell" \
    scenes_0.obj_path="${HMS_DATA_DIR}/sketchfab_objects/barbell/barbell.obj"
