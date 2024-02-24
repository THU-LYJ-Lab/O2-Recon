# python ./exp_runner.py --mode train --conf ./confs/sofa.conf --gpu 4

# python ./exp_runner.py --mode validate_mesh --conf ./confs/scene0616_00-obj9-newfillednvt-64-normal-init-bias0.3.conf --gpu 3 --scene_name scene0616_00_scannet_obj_9_newfillednvt

# run the same model on different objects
scene_name=scene0005_00
gpu_id=5
# for obj_id in 1 2 4 5 6 8 9 10 11 17
for obj_id in 9
do
    for exp_name in newfilled_5
    do
        python ../exp_runner.py --mode train --conf ./confs/paper/${scene_name}-obj${obj_id}-${exp_name}-default-stage1.conf --gpu ${gpu_id} --scene_name ${scene_name}_scannet_obj_${obj_id}_${exp_name}

        python ../exp_runner.py --mode train --conf ./confs/paper/${scene_name}-obj${obj_id}-${exp_name}-default-stage2.conf --gpu ${gpu_id} --scene_name ${scene_name}_scannet_obj_${obj_id}_${exp_name} --is_continue

        python ../exp_runner.py --mode validate_mesh --conf ./confs/paper/${scene_name}-obj${obj_id}-${exp_name}-default-stage2.conf --gpu ${gpu_id} --scene_name ${scene_name}_scannet_obj_${obj_id}_${exp_name} --is_continue

    done
done
