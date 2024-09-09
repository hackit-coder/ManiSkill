# generate 10k trajectories for one task
for i in {0..9}
do
    python -m mani_skill.examples.motionplanning.panda.run --env-id "PickCube-v1" \
      --traj-name="trajectory$i" -n 1000 --only-count-success --seed $((i * 10000)) &
done

python -m mani_skill.trajectory.merge_trajectory -i demos/PickCube-v1/motionplanning/ -o demos/PickCube-v1/motionplanning/trajectory.h5 -p "trajectory*.h5

python -m mani_skill.trajectory.replay_trajectory \
  --traj-path demos/PickCube-v1/motionplanning/trajectory.h5 \
  --use-first-env-state -c pd_joint_delta_pos -o state \
  --save-traj --num-procs 10 
  
#   --num-procs 1 -b gpu --count 1000

seed=0
demos=9000
python train.py --env-id PickCube-v1 \
  --demo-path ../../../demos/PickCube-v1/motionplanning/trajectory.state.pd_joint_delta_pos.cpu.h5 \
  --control-mode "pd_joint_delta_pos" --sim-backend "cpu" --num-demos ${demos} --max_episode_steps 100 \
  --total_iters 100000 --exp-name diffusion_policy-PickCube-v1-state-${demos}_motionplanning_demos-${seed}