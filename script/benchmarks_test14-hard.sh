PLANNER="instruct_driver"
BENCHMARK='test14-hard'
CHALLENGES=$1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_simulation.py \
    +simulation=$CHALLENGES \
    planner=instruct_driver \
    worker.threads_per_node=8 \
    scenario_builder=nuplan_challenge \
    scenario_filter=$BENCHMARK \
    experiment_uid=$BENCHMARK/$PLANNER \
    verbose=true 
